"""Functional ops, mirroring `torch.nn.functional`."""

from __future__ import annotations

import math
import os

import torch


# Mark every public wrapper as `allow_in_graph` so torch.compile (Dynamo)
# treats each call as an opaque node instead of graph-breaking when it
# can't trace through our Python autograd.Function. Doesn't enable fusion
# into the surrounding compiled region (the node stays a black box), but
# eliminates the cache-invalidation / recompile cost of a graph break and
# lets PyTorch ops on either side keep compiling normally.
try:
    _allow_in_graph = torch.compiler.allow_in_graph
except AttributeError:  # PyTorch < 2.0

    def _allow_in_graph(fn):
        return fn


# Cache opt-in env vars at module load. The previous code did
# `os.environ.get("NANOOPS_LOOKUP_SORTED")` inside `embedding()` on every
# forward call — dict lookup is fast but it's still ~20 layers × 256
# accum × N iter calls; doing it once at import is cleaner. The trade-off:
# env-var changes AFTER `import nanoops.functional` no longer take effect.
# nanchat sets env vars before any import, so this is fine in practice.
_LOOKUP_SORTED = bool(os.environ.get("NANOOPS_LOOKUP_SORTED"))


class Mm(torch.autograd.Function):
    """2D-only matrix multiply, mirroring `torch.mm` semantics.

    Higher-rank inputs must be flattened to 2D by the caller (see `linear`).

    Complexity (left (M,K) @ right (K,N) -> out (M,N)):
      forward:  O(M*K*N) FLOPs; allocates out (M*N). ctx holds refs to
                left and right (O(M*K + K*N) memory) until backward runs.
      backward: O(M*K*N) FLOPs each for grad_left and grad_right -> 2x forward;
                allocates grad_left (M*K) and grad_right (K*N).
      total:    ~3x forward FLOPs end-to-end (standard matmul rule of thumb).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        assert left.ndim == 2, f"left must be 2D, got {left.ndim}D"
        assert right.ndim == 2, f"right must be 2D, got {right.ndim}D"
        ctx.save_for_backward(left, right)
        output = left @ right
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        grad_left = grad_output @ right.T
        grad_right = left.T @ grad_output
        return grad_left, grad_right


def unbroadcast(grad: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    # Sum out broadcasted dimensions to match target shape. Two phases:
    #   (1) any extra leading dims (grad.ndim > len(target_shape)) get
    #       summed away without keepdim.
    #   (2) any size-1 target dim that got broadcast up gets summed with
    #       keepdim=True.
    # Both phases are batched into a SINGLE sum() call each (tuple of dims),
    # so the total dispatch cost is at most 2 kernel launches regardless of
    # how many dims need reducing. The original loop-form did one launch per
    # dim — Add/Mul/Where backward called from every elementwise op in the
    # graph, so the savings add up.
    n_extra = grad.ndim - len(target_shape)
    if n_extra > 0:
        grad = grad.sum(dim=list(range(n_extra)))
    # grad.ndim == len(target_shape) now.
    keepdim_axes = [
        i for i, (g, t) in enumerate(zip(grad.shape, target_shape)) if g != t
    ]
    if keepdim_axes:
        grad = grad.sum(dim=keepdim_axes, keepdim=True)
    return grad


class Add(torch.autograd.Function):
    """Elementwise add with NumPy/PyTorch broadcasting semantics."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        # Fail fast with a clear error if shapes are not broadcastable,
        # instead of relying on the downstream `+` op's message.
        torch.broadcast_shapes(left.shape, right.shape)
        ctx.left_shape = left.shape
        ctx.right_shape = right.shape
        return left + right

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return unbroadcast(grad_output, ctx.left_shape), unbroadcast(
            grad_output, ctx.right_shape
        )


class Mul(torch.autograd.Function):
    """Elementwise multiply with NumPy/PyTorch broadcasting semantics.

    Memory note: unlike `Add` whose backward only needs operand shapes,
    `Mul`'s backward needs each operand multiplied by the OTHER one
    (dL/da = g * b, dL/db = g * a) — so we must save both tensors. For
    (..., D) bf16 inputs that's 2 x (..., D) tensors held in ctx until
    backward, roughly 4 bytes/elem. This is the cost of a non-constant
    Jacobian even for an op as simple as elementwise multiply.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        # Fail fast with a clear error if shapes are not broadcastable.
        torch.broadcast_shapes(left.shape, right.shape)
        ctx.save_for_backward(left, right)
        return left * right

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        # "Multiply by the other operand", then unbroadcast for shape.
        grad_left = unbroadcast(grad_output * right, left.shape)
        grad_right = unbroadcast(grad_output * left, right.shape)
        return grad_left, grad_right


@_allow_in_graph
def mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.mul` / the `*` operator (elementwise with broadcasting)."""
    return Mul.apply(left, right)


@_allow_in_graph
def linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    # weight: (out, in); input: (..., in) -> (..., out)
    # Layout: storing weight as (out, in) keeps the contracted dim (`in`) as
    # the fast axis in BOTH operands of the GEMM below. `weight.T` is a free
    # view (stride flip, no copy); BLAS then sees both operands with `in`
    # contiguous — the cache-friendliest pattern for inner dot products.
    # With tuned BLAS this is mostly absorbed by internal packing, but the
    # advantage becomes real in tensor-parallel sharding (row-blocks of W
    # are physically contiguous) and in any non-BLAS / small-matrix path.
    input_shape = input.shape
    new_input = input.reshape(-1, input_shape[-1])
    out = Mm.apply(new_input, weight.T)
    out = Add.apply(out, bias) if bias is not None else out
    return out.reshape(*input_shape[:-1], -1)


class Lookup(torch.autograd.Function):
    """1D-only index select, mirroring `torch.nn.functional.embedding` semantics.

    Complexity (indices (N,), weight (V,D) -> out (N,D)):
      forward:  O(N*D) - copy N rows. No V factor: this is the whole point of
                using indexing over `one_hot(indices) @ weight`, which would
                cost O(N*V*D) compute and O(N*V) memory for the one-hot matrix.
                Allocates out (N*D).
      backward: O(V*D) to zero-init grad_weight + O(N*D) for index_add_,
                i.e. O(V*D) when V >> N (the typical case: V=50k, N=batch*seq).
                The dense (V*D) grad allocation is what `sparse=True` avoids
                in production paths - only N rows are actually touched.

    Key takeaway: embedding's asymptotic win is on the forward pass; backward
    still materializes a full (V*D) gradient tensor unless sparse.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        indices: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        # (N,E)
        assert indices.ndim == 1, f"indices must be 1D, got {indices.ndim}D"
        # (E,D)
        assert weight.ndim == 2, f"weight must be 2D, got {weight.ndim}D"
        # Backward only needs weight's shape + dtype + device, not its data.
        # Saving a fresh empty-shape buffer instead of the (V, D) weight
        # decouples ctx lifetime from the embedding parameter. In practice
        # the weight is alive anyway (it's a model param), so peak memory is
        # unchanged — this is just hygiene.
        ctx.save_for_backward(indices)
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device
        # (N, D)
        return weight[indices]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        (indices,) = ctx.saved_tensors
        N = indices.shape[0]
        grad_weight = torch.zeros(
            ctx.weight_shape,
            dtype=ctx.weight_dtype,
            device=ctx.weight_device,
        )
        if N == 0:
            return None, grad_weight  # nothing to scatter, mirror LookupSorted
        #  for i in range(N):
        #      grad_weight[indices[i]] += grad_output[i]
        grad_weight.index_add_(0, indices, grad_output)
        return None, grad_weight


class LookupSorted(torch.autograd.Function):
    """Same forward as Lookup, backward uses sort + cumsum-based segmented sum.

    ⚠ NOT USED BY DEFAULT — kept as an instructive counter-example. The
    textbook "atomic-add is slow on GPUs, sort first" rewrite turned out to
    be a regression on three separate axes (correctness in bf16, isolated
    speed, end-to-end training speed). Opt in via `NANOOPS_LOOKUP_SORTED=1`
    if you want to reproduce the comparison; the env-var-gated wrapper in
    `embedding()` routes to a `torch.library.custom_op` variant below
    (`_lookup_sorted_op`) which is the only form that survives
    `torch.compile(model, dynamic=False)`. This bare autograd.Function is
    kept for eager-mode parity tests and for reading the algorithm clearly.

    Algorithm (atomic-free segmented sum):

        1. sort indices to put identical row-targets next to each other
        2. permute grad_output the same way
        3. unique_consecutive(counts) finds the group boundaries
        4. cumsum + boundary-diff gives the per-group sum (purely parallel)
        5. one dense write to grad_weight at the unique row positions

    ─────────────────────────────────────────────────────────────────────
    Empirical findings (RTX 3090 ×2, nanchat d20, V=32768, D=1280):
    ─────────────────────────────────────────────────────────────────────

    Issue 1 — bf16 cumsum drift (fixed):
      Cumsum over 4096 rows in bf16 (7-bit mantissa) accumulates >10%
      relative error in `grad_weight`. cumsum[end] - cumsum[start] then
      propagates the truncation into every group sum.
      Fix: promote the accumulator to fp32 when input is bf16/fp16, cast
      back at the end. Same trick PyTorch's internal embedding-backward
      CUDA kernel uses. See the FP32 ACCUMULATOR comment in the impl.

    Issue 2 — microbench slowdown:
      At nanchat-realistic duplicate rates (Zipf, max 15 repeats, mean
      1.79), LookupSorted is ~4× SLOWER than Lookup in isolation
      (1.84 ms vs 0.45 ms / iter on 3090).
      Root cause: sort + permute + (N, D) cumsum overhead exceeds the
      atomicAdd contention Lookup pays. Ampere's hardware atomic-add is
      cheap when collisions are few; "GPUs hate atomics" was Pascal-era
      wisdom that no longer holds at low contention.

    Issue 3 — torch.compile incompatibility (fixed via custom_op):
      `unique_consecutive` returns a data-dependent output shape. nanchat
      does `torch.compile(model, dynamic=False)`, so aot_autograd
      fake-traces the backward and raises DynamicOutputShapeException at
      compile time.
      Workarounds that DIDN'T help:
        - @torch._dynamo.disable on a helper: aot_autograd traces via fake
          tensors, separate from Dynamo's symbolic tracing
        - `capture_dynamic_output_shape_ops=True`: got past
          unique_consecutive but immediately hit `if unique_rows.numel() > 1`
          as a GuardOnDataDependentSymNode
      The fix is `torch.library.custom_op` for BOTH forward AND backward
      (see `_lookup_sorted_op` / `_lookup_sorted_bw_op` below). Registering
      backward as its own custom_op + register_fake stub makes it a
      dispatcher-level opaque node, so aot_autograd uses the shape stub and
      never fake-traces into it. `register_autograd` alone is insufficient
      because its callback is still a plain Python function aot_autograd
      will wrap and trace.

    Issue 4 — end-to-end slowdown survives the compile fix:
      Once the compile-compat issue is fixed via custom_op, the slowdown
      from Issue 2 still shows up in real training:
        Case A (Lookup):              tok/s 28,790,  MFU 58.5%, dt 36.4s
        Case B (LookupSorted custom): tok/s 27,694,  MFU 56.3%, dt 37.8s
                                      → ~3.8% slower end-to-end
      Microbench's 4× ratio is diluted because embedding backward is a
      small fraction of total per-iter time, but the regression is
      consistent and measurable.

    Where the textbook argument WOULD have held: very heavy duplicate
    counts (one row hit hundreds of times). Doesn't happen in LLM training
    where vocab is large and any single token has at most O(seq_len) hits.

    ─────────────────────────────────────────────────────────────────────
    Lessons (educational, since this op exists for teaching, not speed):
      1. GPU atomicAdd is fast on Ampere/Hopper at low collision rates;
         Pascal-era "atomics are slow" intuition no longer applies broadly.
         Benchmark before assuming the "more parallel" rewrite wins.
      2. Mixed-precision reductions need fp32 accumulators or you silently
         train wrong gradients. Always verify gradients in the target dtype.
      3. Python autograd.Function backwards CAN be traced by aot_autograd
         when the forward is inside `torch.compile`. `register_autograd` on
         a custom_op is not enough on its own — wrap the backward as its
         own custom_op + fake stub to make it truly opaque.
    ─────────────────────────────────────────────────────────────────────
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        indices: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        assert indices.ndim == 1, f"indices must be 1D, got {indices.ndim}D"
        assert weight.ndim == 2, f"weight must be 2D, got {weight.ndim}D"
        ctx.save_for_backward(indices, weight)
        return weight[indices]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor]:
        indices, weight = ctx.saved_tensors
        # Delegate to a Dynamo-disabled helper. unique_consecutive returns
        # a data-dependent output shape, which torch.compile(dynamic=False)
        # (the mode nanchat uses) refuses to trace — it raises
        # DynamicOutputShapeException at compile time. Pulling the body out
        # of LookupSorted.backward (which aot_autograd CAN see when the
        # forward is inside a compiled region) into a @torch._dynamo.disable
        # function makes the whole backward opaque to Dynamo, so it runs
        # eagerly and unique_consecutive works as expected.
        return _lookup_sorted_backward_impl(indices, weight, grad_output)


@torch._dynamo.disable
def _lookup_sorted_backward_impl(
    indices: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor
) -> tuple[None, torch.Tensor]:
    grad_weight = torch.zeros_like(weight)
    N = indices.shape[0]
    if N == 0:
        return None, grad_weight

    # 1. Sort indices to co-locate duplicates.
    sorted_indices, perm = indices.sort()
    # 2. Permute grad rows so each group of identical targets is contiguous.
    sorted_grads = grad_output[perm]
    # 3. Group structure: which row each group targets, and how many
    #    sorted entries fall into each group.
    unique_rows, counts = sorted_indices.unique_consecutive(return_counts=True)

    # 4. Per-group sum via cumsum + boundary diff (no atomics):
    #    cumsum[k] = sum of sorted_grads[:k+1]
    #    group i ends at index group_ends[i] (last entry of group i in
    #    sorted order). Then:
    #      group_sum[i] = cumsum[group_ends[i]] - cumsum[group_ends[i-1]]
    #    (group_sum[0] is just cumsum[group_ends[0]]; no prior group.)
    #
    # FP32 ACCUMULATOR (only for low-precision inputs): bf16/fp16 cumsum
    # over thousands of rows accumulates >10% relative error because
    # each partial sum gets truncated to 7/10 mantissa bits and the
    # `cumsum[end] - cumsum[start]` subtraction propagates that error
    # into every group sum. Promote to fp32 in those cases; for fp32/
    # fp64 inputs, run in input dtype to avoid the opposite problem
    # (downgrading fp64 → fp32 would introduce its own error). Same
    # trick as PyTorch's internal embedding-backward kernel.
    cumsum_dtype = (
        torch.float32
        if sorted_grads.dtype in (torch.bfloat16, torch.float16)
        else sorted_grads.dtype
    )
    cumsum = sorted_grads.cumsum(0, dtype=cumsum_dtype)
    group_ends = counts.cumsum(0).sub_(1)  # last index per group
    group_sums = cumsum.index_select(0, group_ends)  # alloc (G, D)
    if unique_rows.numel() > 1:
        group_sums[1:].sub_(cumsum.index_select(0, group_ends[:-1]))

    # 5. Dense write — each unique_rows[i] is unique, so this is a
    #    contention-free scatter (one row touched at most once). Cast
    #    only happens when we promoted above; otherwise it's a no-op.
    grad_weight[unique_rows] = group_sums.to(grad_weight.dtype)
    return None, grad_weight


# -----------------------------------------------------------------------------
# torch.library.custom_op-registered version of LookupSorted.
#
# Why a second registration on top of LookupSorted (the autograd.Function):
#   The autograd.Function path fails under nanchat's torch.compile(model,
#   dynamic=False). aot_autograd fake-traces the user-defined backward and
#   hits unique_consecutive / data-dependent guards (`u0 > 1`), crashing
#   the compile. Even @torch._dynamo.disable on the helper didn't save it
#   because aot_autograd's fake-tracing happens through a different code
#   path than Dynamo's symbolic tracing.
#
#   torch.library.custom_op is the canonical escape: registers the forward
#   into PyTorch's dispatcher as a first-class op (with register_fake giving
#   aot_autograd the shape stub), and register_autograd installs the backward
#   as an autograd-only Python callback. The compiled forward sees the op as
#   opaque (looks up the fake stub for shapes), and the backward runs in
#   eager outside the compiled graph — no fake-tracing into it.
#
# Only registered if torch.library.custom_op exists (PyTorch >= 2.4).
try:
    _HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")
except AttributeError:
    _HAS_CUSTOM_OP = False

if _HAS_CUSTOM_OP:

    @torch.library.custom_op("nanoops::lookup_sorted", mutates_args=())
    def _lookup_sorted_op(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Forward: identical to Lookup.forward (just weight[indices])."""
        assert indices.ndim == 1
        assert weight.ndim == 2
        return weight[indices]

    @_lookup_sorted_op.register_fake
    def _lookup_sorted_op_meta(
        indices: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Shape stub for aot_autograd: output is (N, D)."""
        return weight.new_empty(indices.shape[0], weight.shape[-1])

    # The BACKWARD is also registered as its own custom_op + fake stub.
    # Without this, `register_autograd`'s backward callback is still a
    # plain Python function — aot_autograd builds a wrapping
    # `GeneratedBackwardFor_*` node and fake-traces it, hitting
    # unique_consecutive's DynamicOutputShapeException all over again.
    # Registering as a dispatcher op makes the backward a single opaque
    # node (whose meta is `torch.empty_like(weight)`), so aot_autograd
    # uses the fake stub and never looks inside.
    @torch.library.custom_op("nanoops::lookup_sorted_bw", mutates_args=())
    def _lookup_sorted_bw_op(
        indices: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor
    ) -> torch.Tensor:
        """Backward impl wrapped as a custom_op for aot_autograd opacity."""
        _, grad_weight = _lookup_sorted_backward_impl(indices, weight, grad_output)
        return grad_weight

    @_lookup_sorted_bw_op.register_fake
    def _lookup_sorted_bw_op_meta(
        indices: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor
    ) -> torch.Tensor:
        """Shape stub for the backward: grad_weight has weight's shape."""
        return torch.empty_like(weight)

    def _lookup_sorted_op_setup_context(ctx, inputs, output):
        indices, weight = inputs
        ctx.save_for_backward(indices, weight)

    def _lookup_sorted_op_backward(ctx, grad_output):
        indices, weight = ctx.saved_tensors
        grad_weight = _lookup_sorted_bw_op(indices, weight, grad_output)
        # custom_op autograd returns one grad per input (indices, weight).
        # indices is int → non-differentiable → None.
        return None, grad_weight

    _lookup_sorted_op.register_autograd(
        _lookup_sorted_op_backward,
        setup_context=_lookup_sorted_op_setup_context,
    )


@_allow_in_graph
def embedding(
    indices: torch.Tensor,
    weight: torch.Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    # Signature matches torch.nn.functional.embedding so that
    # nn.Embedding.forward() (which always passes all 7 args from its config)
    # can route through this function via monkey-patching. nanoops doesn't
    # implement padding_idx / max_norm / scale_grad_by_freq / sparse — accept
    # only the (None/False/default) values that nanchat actually uses.
    if padding_idx is not None:
        raise NotImplementedError("nanoops.embedding: padding_idx not supported")
    if max_norm is not None:
        raise NotImplementedError("nanoops.embedding: max_norm not supported")
    if scale_grad_by_freq:
        raise NotImplementedError("nanoops.embedding: scale_grad_by_freq not supported")
    if sparse:
        raise NotImplementedError("nanoops.embedding: sparse not supported")
    # weight: (V, D); input: (...) -> (..., D)
    # Layout (why (V, D), not (D, V)):
    #   Under row-major storage, `weight[i]` reads D contiguous floats
    #   (~1-2 cache lines). `index_add_` along dim 0 likewise writes D
    #   contiguous floats. A (D, V) layout would do strided column gathers
    #   with V*4-byte stride (~200KB jumps at V=50k), turning each lookup
    #   into ~D cache misses instead of ~D/16. This op doesn't go through
    #   BLAS, so the layout-locality advantage shows up directly —
    #   typically a 10-100x gap, not a wash.
    indices_shape = indices.shape
    new_indices = indices.reshape(-1)
    # Default to `Lookup` (naive index_add_); set `NANOOPS_LOOKUP_SORTED=1`
    # to swap to the sorted variant. See LookupSorted's docstring for why
    # sorted is NOT the default (bf16 cumsum accumulation error + slower
    # on Ampere atomicAdd at realistic duplicate rates).
    #
    # When opt-in, prefer the torch.library.custom_op registration
    # (`_lookup_sorted_op`) over the bare autograd.Function (`LookupSorted`).
    # The custom_op variant is compile-safe — its backward is opaque to
    # aot_autograd via register_autograd, so nanchat's
    # `torch.compile(model, dynamic=False)` no longer chokes on the
    # data-dependent unique_consecutive / numel comparisons inside the
    # backward. The autograd.Function variant is kept for the eager-mode
    # parity tests (no torch.compile needed there).
    if _LOOKUP_SORTED:
        if _HAS_CUSTOM_OP:
            out = _lookup_sorted_op(new_indices, weight)
        else:
            out = LookupSorted.apply(new_indices, weight)
    else:
        out = Lookup.apply(new_indices, weight)
    return out.reshape(*indices_shape, -1)


class RMSNorm(torch.autograd.Function):
    """Root-mean-square normalization, mirroring `torch.nn.functional.rms_norm` semantics."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        assert input.ndim == 2, f"input must be 2D, got {input.ndim}D"
        if weight is not None:
            assert weight.ndim == 1, f"weight must be 1D, got {weight.ndim}D"
            assert input.shape[-1] == weight.shape[0], (
                f"last dim of input ({input.shape[-1]}) must match weight shape ({weight.shape[0]})"
            )
        # Use rsqrt + multiply instead of sqrt + divide — throughput, not
        # precision: rsqrt + mul and sqrt + div are numerically equivalent
        # for finite positive inputs (rsqrt is exp(-0.5*log(x)) internally
        # vs sqrt's exp(0.5*log(x))). The win:
        # - rsqrt() is one CUDA instruction (SFU-issued, ~throughput parity
        #   with mul)
        # - fp32 mul throughput is ~30x fp32 div on Ampere/Hopper
        # So `mean.add_(eps).rsqrt_()` + `input * rsqrt` is ~2 fast ops
        # vs `sqrt(mean + eps)` + `input / sqrt_val` = 1 fast + 1 slow.
        #
        # The `.add_(eps).rsqrt_()` tail uses in-place forms — once the
        # (..., 1) mean tensor exists, the add and rsqrt steps have no
        # other refs, so writing in-place skips two tiny allocations.
        rsqrt = input.pow(2).mean(dim=-1, keepdim=True).add_(eps).rsqrt_()
        y = input * rsqrt
        ctx.save_for_backward(weight, rsqrt, y)
        output = y * weight if weight is not None else y
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
        weight, rsqrt, y = ctx.saved_tensors
        # grad_weight uses original grad_output (= g); grad_input uses
        # g_eff = g·w (or just g when weight is None) — see README appendix.
        if weight is not None:
            grad_weight = (grad_output * y).sum(dim=0)
            g_eff = grad_output * weight
        else:
            grad_weight = None
            g_eff = grad_output
        # Build grad_input = rsqrt * (g_eff - y * inner) with one alloc + a
        # chain of in-place ops instead of `rsqrt * (g_eff - y * (g_eff*y).mean)`
        # which churns 4 input-sized temps via Python operator chain. Same
        # math, fewer alloc/free cycles.
        inner = (g_eff * y).mean(dim=-1, keepdim=True)  # tiny (..., 1)
        grad_input = y * inner  # alloc 1 (broadcasts)
        grad_input.neg_().add_(g_eff)  # in-place: g_eff - y*inner
        grad_input.mul_(rsqrt)  # in-place: rsqrt * (above)
        return grad_input, grad_weight, None


@_allow_in_graph
def rms_norm(
    input: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert len(normalized_shape) == 1, (
        f"nanoops rms_norm only supports 1D normalized_shape, got {normalized_shape}"
    )
    D = normalized_shape[0]
    assert input.shape[-1] == D, (
        f"input last dim {input.shape[-1]} != normalized_shape {D}"
    )
    input_shape = input.shape
    input_flat = input.reshape(-1, D)
    out_flat = RMSNorm.apply(input_flat, weight, eps)
    return out_flat.reshape(input_shape)


@_allow_in_graph
def outer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.outer`. Both inputs must be 1D.

    Implemented as a 2D matmul of column x row: (M, 1) @ (1, N) -> (M, N).
    No new autograd Function needed — Mm's backward plus autograd's view
    backward together give the right (M,) and (N,) gradients.
    """
    assert left.ndim == 1, f"left must be 1D, got {left.ndim}D"
    assert right.ndim == 1, f"right must be 1D, got {right.ndim}D"
    return Mm.apply(left.unsqueeze(-1), right.unsqueeze(0))


class Cat(torch.autograd.Function):
    """Concatenation along a dim, mirroring `torch.cat` semantics.

    Signature note: `dim` is the FIRST positional parameter (before `*tensors`)
    because Python requires `*args` to be the last positional parameter. The
    functional `cat()` wrapper re-orders to PyTorch's `(tensors, dim)` convention.

    Why `*tensors` and not `tensors: list[Tensor] | tuple[Tensor, ...]`:
    `autograd.Function.apply()` only tracks gradients through tensors that are
    individual positional arguments — it does NOT recurse into list/tuple
    containers to find tensors. Passing a list as a single arg makes autograd
    treat it as an opaque non-differentiable constant, so the backward graph
    never gets built and the output ends up with no grad_fn. Splatting the
    tensors via `*tensors` makes each one its own positional arg, which is
    what autograd needs to set up the chain. (Type hints are static-analysis
    only; they don't influence apply's runtime behavior.)
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        dim: int,
        *tensors: torch.Tensor,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.sizes = [t.shape[dim] for t in tensors]
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> "tuple[None, *tuple[torch.Tensor, ...]]":
        grads = torch.split(grad_output, ctx.sizes, dim=ctx.dim)
        return (None, *grads)  # None for dim (int, non-differentiable)


@_allow_in_graph
def cat(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    dim: int = 0,
) -> torch.Tensor:
    """Mirrors `torch.cat`. Reorders args so dim goes to Cat's first slot."""
    return Cat.apply(dim, *tensors)


@_allow_in_graph
def stack(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    dim: int = 0,
) -> torch.Tensor:
    """Mirrors `torch.stack` using `cat` and `unsqueeze`."""
    assert all(t.shape == tensors[0].shape for t in tensors), (
        f"all tensors must have the same shape; got {[tuple(t.shape) for t in tensors]}"
    )
    unsqueezed = [t.unsqueeze(dim) for t in tensors]
    return cat(unsqueezed, dim=dim)


class ReluSquare(torch.autograd.Function):
    """Fused ReLU-squared: y = max(x, 0)**2 — nanchat's MLP activation.

    Backward simplifies beautifully:
        d(relu(x)^2) / dx = 2 * relu(x) * 1[x > 0] = 2 * relu(x)
    The mask is redundant because relu(x) is already 0 wherever x ≤ 0,
    so we don't need to save or apply it — backward is just `2 * y * g`,
    one fused mul chain. Saves the relu output y (same memory as input).

    Compared to composing `square(relu(x))`: this fuses both ops' backwards
    into one analytic expression, ~3x fewer backward FLOPs and one fewer
    intermediate tensor. The classic "fusion-as-optimization" pattern,
    minus the GPU kernel — same idea as fused GELU / SwiGLU in production.

    Subgradient at x=0 taken as 0 (PyTorch convention).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        y = x.clamp(min=0)
        ctx.save_for_backward(y)
        return y * y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        # `grad_output * 2 * y` would alloc twice (2*grad_output, then *y).
        # Compute (2*y) into a fresh buf, then in-place multiply by grad_output.
        # Saves one input-sized alloc per call. y is `relu(x)` so the mask
        # is already absorbed (zero where x ≤ 0).
        buf = y * 2
        buf.mul_(grad_output)
        return buf


@_allow_in_graph
def relu_square(input: torch.Tensor) -> torch.Tensor:
    """Fused relu(x)**2 — nanchat's MLP activation."""
    return ReluSquare.apply(input)


class Softmax(torch.autograd.Function):
    """Softmax along a dim, mirroring `torch.nn.functional.softmax` semantics."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        input_max = input.amax(dim=dim, keepdim=True)
        input_exp = (input - input_max).exp()
        sum_exp = input_exp.sum(dim=dim, keepdim=True)
        # In-place divide reuses input_exp's storage as the output. Safe because
        # input_exp is a fresh tensor (just created by .exp()) with no other
        # references — saves one input-sized allocation vs `input_exp / sum_exp`.
        output = input_exp.div_(sum_exp)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (output,) = ctx.saved_tensors
        dim = ctx.dim
        # Compact: y * (g - <g,y>) = g*y - y*<g,y>. Compute g*y once into gy,
        # then in-place subtract y * <g,y> via addcmul_(... value=-1), which
        # is a single fused kernel — no `output * inner` intermediate tensor.
        # Saves one input-sized transient + one kernel launch vs the naive form.
        gy = grad_output * output  # 1 alloc
        inner = gy.sum(dim=dim, keepdim=True)  # tiny <g,y>
        gy.addcmul_(output, inner, value=-1)  # in-place: gy -= y * inner
        return gy, None  # None for dim (int, non-differentiable)


@_allow_in_graph
def softmax(
    input: torch.Tensor,
    dim: int = -1,
    _stacklevel: int = 3,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Mirrors `torch.nn.functional.softmax`. Default dim=-1 (PyTorch's
    default is None and warns; we pick the common case).

    `_stacklevel` and `dtype` are accepted for signature compatibility with
    `torch.nn.functional.softmax` (so `nn.Softmax.forward()` can route here
    via monkey-patching); `_stacklevel` is ignored, `dtype` triggers a cast
    of the input before softmax (matching PyTorch's behavior).
    """
    if dtype is not None:
        input = input.to(dtype)
    return Softmax.apply(input, dim)


def chunked_logsumexp(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Memory-efficient logsumexp via online softmax.

    `torch.logsumexp(x, dim)` materializes a full `exp(x - max)` tensor the
    same shape as `x` during compute. For (B*T, V) logits at LLM scale this
    is a transient ~13 GB allocation. This chunked version processes `x`
    along `dim` in slices of `chunk_size`, maintaining a running
    `(max, sum_of_exp_shifted_by_max)` pair. Transient peak memory drops to
    roughly `chunk_size / size_along_dim` times what `torch.logsumexp` uses.

    Algorithm (online softmax — the same trick at the heart of Flash Attention):

        for each chunk c along dim:
            m'      = max(running_max, c.amax)
            new_sum = running_sum * exp(running_max - m')        # rebase to m'
            new_sum += (c - m').exp().sum                         # add chunk
            running_max, running_sum = m', new_sum
        return running_max + log(running_sum)

    Numerically identical to `torch.logsumexp` up to fp rounding (each rebase
    preserves the running LSE in shifted form).

    Performance: roughly memory-bandwidth bound; smaller chunks add only
    kernel-launch overhead. On a 3090, chunk_size=4096 is ~5% slower than
    `torch.logsumexp` with 1/4 the transient peak; chunk_size=1024 cuts the
    peak by 16x at ~14% extra time. Anything in [1024, 8192] is reasonable.
    """
    chunks = input.split(chunk_size, dim)
    # Initialize running stats from the first chunk: no rebase needed yet.
    # `.exp_()` (in-place) on the (chunks[0] - running_max) temporary saves
    # one chunk-sized allocation — the temp has no other references, so
    # writing exp directly into its storage is safe.
    running_max = chunks[0].amax(dim, keepdim=True)
    running_sum = (chunks[0] - running_max).exp_().sum(dim, keepdim=True)
    # Fold in each remaining chunk: shift the basis to the new max, then add
    # the chunk's own exp-sum contribution computed in that same basis.
    for chunk in chunks[1:]:
        new_max = torch.maximum(running_max, chunk.amax(dim, keepdim=True))
        rebase = (running_max - new_max).exp()  # rescale old sum
        chunk_sum = (
            (chunk - new_max).exp_().sum(dim, keepdim=True)
        )  # same .exp_() trick
        running_sum.mul_(rebase).add_(chunk_sum)  # in-place chain
        running_max = new_max
    result = running_max + running_sum.log()
    return result if keepdim else result.squeeze(dim)


class CrossEntropy(torch.autograd.Function):
    """Cross-entropy loss, mirroring `torch.nn.functional.cross_entropy` semantics.

    Fused log-softmax + NLL with memory optimizations in both directions.
    Empirical: at NT=16K, V=32K, bf16 (logits ~1 GB), the total peak GPU
    memory is roughly HALF of PyTorch's `F.cross_entropy` (2x input vs 4x
    input).

    Forward optimizations:
      1. ctx saves `(input, log_sum_exp, target)`, NOT softmax — `input`
         is already alive as a function argument (saved reference is free)
         and `log_sum_exp` is `(..., 1)`. Backward recomputes softmax via
         `(input - log_sum_exp).exp_()`. Saves ~1x logits of long-lived
         memory vs PyTorch (which keeps log_softmax in ctx until backward).
      2. LSE uses `chunked_logsumexp` (online softmax in Python), which
         keeps the transient exp tensor bounded by `chunk_size / V` of
         full logits (~1/8 at default chunk_size). A/B'd against plain
         `torch.logsumexp` on nanchat d20 — peak memory and wall time
         were byte-identical: backward's `(input - log_sum_exp).exp_()`
         already allocates the full softmax tensor, so it dominates the
         peak regardless of forward's choice. Kept chunked for the
         online-softmax algorithm demonstration (same trick Flash
         Attention uses for tile-wise normalization).

    Backward optimizations (all in-place to keep peak at 1x softmax):
      3. `(input - log_sum_exp).exp_()` — in-place exp on the sub temp;
         using `.exp()` would briefly hold both sub_temp and exp_result
         simultaneously (2x peak).
      4. `grad_input *= ...` — in-place scaling by upstream * mask; `a*b`
         would create a *new* (B*T, V) tensor, another 2x peak transient.

    See README appendix for the derivation; the boxed result is
    `dL/dx = softmax(x) - one_hot(target)`.

    Honest performance note: nanoops wins on memory but loses on time.
    At the same NT=16K, V=32K, bf16 benchmark on a 3090, one fwd+bwd takes:

        PyTorch F.cross_entropy (eager)        :  9.2 ms   (baseline)
        torch.compile (default, fused Triton)  :  4.1 ms   (2.3x faster)
        nanoops CrossEntropy (this class)      : 16.3 ms   (1.8x SLOWER)

    The slowness is structural: nanoops's Python-level autograd Function
    dispatches ~15-20 individual PyTorch ops per fwd+bwd, each paying
    ~30-50 us of dispatch + autograd machinery (the chunked LSE loop alone
    is ~8 op-dispatches per chunk). PyTorch eager calls a single fused
    C++ `cross_entropy_loss`; `torch.compile` further fuses everything
    into one Triton kernel. Closing this gap requires kernel-level fusion
    (Triton / CUDA), which is out of scope for nanoops.

    So: read this class to understand what fusion buys you in memory and
    why production code (Liger Kernel, Flash CE, `torch.compile`) does it
    at the kernel level. Don't use this class as a drop-in replacement
    expecting the same throughput.

    Returns per-sample loss (shape matches `target`); `'mean'` / `'sum'`
    reduction is handled by the `cross_entropy()` functional wrapper.

    See the README appendix for the full derivation (the boxed result is
    `dL/dx = softmax(x) - one_hot(target)`).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        # chunked_logsumexp (online softmax) keeps the transient exp
        # bounded by chunk_size/V of full logits. A/B'd vs torch.logsumexp
        # on nanchat d20: byte-identical peak memory + wall time — the
        # backward's full softmax recompute already sets the peak ceiling,
        # so forward's chunking choice is invisible end-to-end. Kept for
        # the algorithm demonstration (same online-softmax trick as Flash).
        log_sum_exp = chunked_logsumexp(input, dim=dim, keepdim=True)
        # ignore_index handling: gather() would crash on out-of-range target
        # values (e.g. -1), so replace those with 0 as a safe placeholder,
        # compute the per-sample loss as usual, then zero out those positions.
        valid_mask = target != ignore_index
        safe_target = torch.where(valid_mask, target, 0)
        target_idx = safe_target.unsqueeze(dim)
        per_sample = (log_sum_exp - input.gather(dim, target_idx)).squeeze(dim)
        per_sample.mul_(valid_mask)  # in-place: 0 at ignored positions
        # ─────────────────────────────────────────────────────────────────
        # CTX TRADE-OFF: valid_mask + target_idx are CHEAP to recompute in
        # backward (both derive from `target` + `ignore_index`):
        #     valid_mask   = target != ignore_index             # 1 op
        #     safe_target  = torch.where(valid_mask, target, 0) # 1 op
        #     target_idx   = safe_target.unsqueeze(dim)         # 1 op (view)
        # So an equally-valid implementation would save only `target` (+
        # `ignore_index` on ctx) and rebuild these in backward — that's the
        # canonical compute-over-memory choice.
        # We pick the OTHER side: cache valid_mask + target_idx (bool (N,) +
        # int64 (N, 1), few KB total) so backward skips 2 dispatches. Cheap
        # vs the giant `input` (N×V floats) we're already keeping alive — the
        # ctx-size delta is negligible, and the dispatch savings are free.
        # ─────────────────────────────────────────────────────────────────
        # Cache the -1 column vector once at forward — backward's scatter_add
        # always needs `full_like(target_idx, -1.0)`, and rebuilding it every
        # call is a fresh (N, 1) alloc per CE.backward. Tiny in absolute
        # terms (~N int64) but at zero extra ctx cost.
        neg_one_col = torch.full_like(target_idx, -1.0, dtype=input.dtype)
        ctx.save_for_backward(input, log_sum_exp, valid_mask, target_idx, neg_one_col)
        ctx.dim = dim
        return per_sample

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        # valid_mask, target_idx are pulled from ctx (computed in forward).
        # Could equivalently be rebuilt here from `target` + `ignore_index`;
        # see the forward's CTX TRADE-OFF block for why we cache instead.
        input, log_sum_exp, valid_mask, target_idx, neg_one_col = ctx.saved_tensors
        dim = ctx.dim
        # Recompute softmax = exp(input - LSE); stable since input - LSE <= 0.
        # `.exp_()` (in-place on the sub temp) keeps peak at 1x (..., V); using
        # `.exp()` would briefly hold sub_temp AND exp_result simultaneously
        # → 2x peak. Same one extra mul+exp we pay for the forward savings.
        grad_input = (input - log_sum_exp).exp_()  # = softmax(input)
        # softmax - one_hot(target): subtract 1 at the target column per row.
        # For ignored positions we used safe_target=0 in forward (the row
        # gets zeroed out by valid_mask below, so the spurious -1 at col 0
        # doesn't matter). neg_one_col was pre-built in forward.
        grad_input.scatter_add_(dim, target_idx, neg_one_col)
        # Scale by upstream grad_output AND zero out ignored rows in one step.
        # Use `*=` (in-place) — `a * b` would create a *new* (B*T, V) tensor,
        # briefly doubling backward peak to 2x the softmax size before the old
        # one becomes unreferenced. `*=` modifies grad_input directly.
        #   - grad_output (per-sample) broadcasts across the class dim
        #   - valid_mask (bool) → 0.0 at ignored positions kills those rows
        grad_input *= (grad_output * valid_mask).unsqueeze(dim)
        # Return one grad per forward input: (input, target, dim, ignore_index).
        # Only `input` is differentiable; the rest are ints / non-tensor.
        return grad_input, None, None, None


@_allow_in_graph
def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    size_average: bool | None = None,  # deprecated in PyTorch
    ignore_index: int = -100,
    reduce: bool | None = None,  # deprecated in PyTorch
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    dim: int = -1,
) -> torch.Tensor:
    """Mirrors `torch.nn.functional.cross_entropy`.

    Wraps the fused `CrossEntropy` autograd Function and applies reduction.
    See `CrossEntropy` for the algorithm and the memory / time trade-offs;
    this wrapper just adds reduction.

    Args:
      input:        Logits of shape (..., C); class dim defaults to -1.
      target:       Integer class indices, shape matching input minus dim.
                    Positions equal to `ignore_index` contribute 0 to both
                    loss and gradient.
      ignore_index: Target value to skip. Defaults to -100 (PyTorch default);
                    nanchat passes -1 explicitly.
      reduction:    "mean" (default) / "sum" / "none". "mean" divides by the
                    number of *non-ignored* positions, matching PyTorch
                    (returns NaN if all positions are ignored).
      dim:          Class dim. Defaults to -1 (PyTorch hardcodes class to
                    dim 1; we expose `dim` for flexibility).

    Not supported (vs PyTorch):
      - `weight` (per-class loss weighting).
      - `label_smoothing`.
      - Deprecated `size_average` / `reduce` flags.
    Signatures are accepted (for nn.CrossEntropyLoss.forward routing) but
    non-default values raise NotImplementedError.
    """
    if weight is not None:
        raise NotImplementedError("nanoops.cross_entropy: weight not supported")
    if label_smoothing != 0.0:
        raise NotImplementedError(
            "nanoops.cross_entropy: label_smoothing not supported"
        )
    if size_average is not None or reduce is not None:
        raise NotImplementedError(
            "nanoops.cross_entropy: deprecated size_average/reduce not supported"
        )
    per_sample = CrossEntropy.apply(input, target, dim, ignore_index)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        # PyTorch convention: divide by N_valid (excluding ignore_index), not
        # by total. Cast to float to avoid integer division. If everything is
        # ignored, valid_count is 0 and we return NaN (matching PyTorch).
        valid_count = (target != ignore_index).sum().to(per_sample.dtype)
        return per_sample.sum() / valid_count
    raise ValueError(f"unknown reduction: {reduction!r} (expected 'mean'/'sum'/'none')")


class Sigmoid(torch.autograd.Function):
    """Elementwise sigmoid: y = 1 / (1 + exp(-x)).

    Backward: dL/dx = g * y * (1 - y) — uses only y. Sigmoid is bijective,
    so y uniquely determines x; saving y (the output) is the natural choice
    (same memory as saving x, but lets backward use the y(1-y) form directly).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        # Use the tensor method `x.sigmoid()` rather than `torch.sigmoid(x)`:
        # `nanoops.integration` monkey-patches `torch.sigmoid` to *this*
        # function, so calling `torch.sigmoid(x)` here would infinitely
        # recurse. Tensor methods aren't shadowed by that patch.
        y = x.sigmoid()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        # y * (1 - y) = y - y². Build it via two in-place ops on a fresh
        # temp instead of `y * (1 - y)` (which is 2 allocs by itself: 1 for
        # (1 - y), 1 for the product). One alloc + two in-place ops; then
        # one final mul with grad_output.
        buf = y * y  # alloc 1
        buf.neg_().add_(y)  # in-place: buf = y - y² = y(1-y)
        return grad_output * buf  # alloc 2


@_allow_in_graph
def sigmoid(input: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.sigmoid`."""
    return Sigmoid.apply(input)


class Tanh(torch.autograd.Function):
    """Elementwise tanh: y = tanh(x).

    Backward: dL/dx = g * (1 - y^2) — uses only y. Like sigmoid, tanh is
    bijective; saving the output y is the natural choice.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        # See Sigmoid.forward for why we use the tensor method here instead
        # of the top-level `torch.tanh` function.
        y = x.tanh()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        # 1 - y² built via in-place neg + add (saves 1 alloc vs `1 - y * y`).
        buf = y * y  # alloc 1
        buf.neg_().add_(1)  # in-place: buf = 1 - y²
        return grad_output * buf  # alloc 2


@_allow_in_graph
def tanh(input: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.tanh`."""
    return Tanh.apply(input)


class ApplyRotaryEmb(torch.autograd.Function):
    """Apply rotary positional embedding to a 4D tensor's last-dim halves.

    Mirrors `nanchat/gpt.py:57` apply_rotary_emb. Treats the last dim as
    two halves (x1, x2) and rotates each pair (x1[i], x2[i]) by the angle
    encoded in (cos[i], sin[i]):

        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos

    In matrix form this is R(θ) applied to a 2-vector. Rotation matrices
    are orthogonal: R(θ)^T = R(-θ). So backward is the SAME forward shape
    with sin negated:

        grad_x1 = g1 * cos - g2 * sin
        grad_x2 = g1 * sin + g2 * cos

    Memory: ctx saves only (cos, sin) — precomputed lookup tables, usually
    shared across the entire attention call. NO x or y is saved (backward
    is purely linear in grad_output, with cos/sin as multipliers).

    cos and sin come from non-differentiable arange/outer/cos/sin
    precomputation (see `nanchat/gpt.py:_precompute_rotary_embeddings`),
    so backward returns None for both.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        assert x.ndim == 4, f"x must be 4D (B, T, n_head, head_dim), got {x.ndim}D"
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        # Pre-allocate output and write each half via slice assignment.
        # The original `torch.cat([y1, y2], dim=-1)` had to hold y1, y2, AND
        # the cat output alive at the same time (peak ~2x input). Slicing
        # into a pre-allocated buffer drops y1, y2 immediately after the
        # copy, lowering peak to ~1.5x input. Called 40+ times per d20 iter
        # (every Q and K of every layer), so the peak savings compound.
        out = torch.empty_like(x)
        out[..., :d] = x1 * cos + x2 * sin
        out[..., d:] = -x1 * sin + x2 * cos
        ctx.save_for_backward(cos, sin)
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        d = grad_output.shape[-1] // 2
        g1, g2 = grad_output[..., :d], grad_output[..., d:]
        # Same pre-allocate + slice-assign trick as forward; R(-θ) means
        # the formula is the forward with sin → -sin.
        grad_x = torch.empty_like(grad_output)
        grad_x[..., :d] = g1 * cos - g2 * sin
        grad_x[..., d:] = g1 * sin + g2 * cos
        # cos and sin are precomputed constants, not differentiable.
        return grad_x, None, None


@_allow_in_graph
def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotary positional embedding, mirroring `nanchat/gpt.py:57`."""
    return ApplyRotaryEmb.apply(x, cos, sin)


class Where(torch.autograd.Function):
    """Elementwise `torch.where(cond, a, b)` with grad routing by `cond`.

    Forward picks `a` where cond is True, else `b`. Backward routes the
    upstream grad to whichever operand was selected: dL/da is g masked by
    cond, dL/db is g masked by ~cond. `cond` is bool and non-differentiable.

    Broadcasting: cond/a/b can have different broadcastable shapes; the
    backward unbroadcasts each per-operand grad back to the operand's
    original shape (same `unbroadcast` helper as `Add` / `Mul`).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        condition: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(condition)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        return torch.where(condition, a, b)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, torch.Tensor]:
        (condition,) = ctx.saved_tensors
        # `torch.where(condition, grad_output, 0)` routes the grad to whichever
        # operand was selected, in one kernel. The naive
        # `grad_output * condition` / `grad_output * (~condition)` form needs
        # to allocate the bool `~condition` tensor and rely on bool*float
        # promotion. Using torch.where with a scalar fall-through skips both.
        grad_a = unbroadcast(torch.where(condition, grad_output, 0), ctx.a_shape)
        grad_b = unbroadcast(torch.where(condition, 0, grad_output), ctx.b_shape)
        return None, grad_a, grad_b  # condition is bool, non-differentiable


@_allow_in_graph
def where(condition: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mirrors `torch.where(condition, a, b)` (3-arg form)."""
    return Where.apply(condition, a, b)


class Roll(torch.autograd.Function):
    """Cyclic shift along given dim(s), mirroring `torch.roll`.

    Forward applies the shift; backward applies the inverse shift
    (negate shifts, keep dims). Bijective op — backward saves no tensor,
    just the integer shift/dim params on ctx.

    `shifts` can be `int` or tuple; `dims` can be `int`, tuple, or `None`
    (None flattens the tensor first, same as `torch.roll`).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        shifts: int | tuple[int, ...],
        dims: int | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        ctx.shifts = shifts
        ctx.dims = dims
        return torch.roll(input, shifts, dims)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        shifts = ctx.shifts
        if isinstance(shifts, (list, tuple)):
            neg_shifts = tuple(-s for s in shifts)
        else:
            neg_shifts = -shifts
        return torch.roll(grad_output, neg_shifts, ctx.dims), None, None


@_allow_in_graph
def roll(
    input: torch.Tensor,
    shifts: int | tuple[int, ...],
    dims: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Mirrors `torch.roll`. shifts: int or tuple; dims: int, tuple, or None."""
    return Roll.apply(input, shifts, dims)


class ScaledDotProductAttention(torch.autograd.Function):
    """Naive (non-Flash) scaled dot-product attention with explicit backward.

    Forward (row-vector convention, features on last dim):
        S = (Q @ K^T) * scale         # scores, shape (..., L, S)
        S += mask / -inf where masked # optional causal or attn_mask
        P = softmax(S, dim=-1)        # attention probs
        O = P @ V                     # output, shape (..., L, D_v)

    Backward (closed form, derivable from chain rule over the three ops):
        dV = P^T @ g
        dP = g @ V^T
        dS = P * (dP - sum(P * dP, dim=-1, keepdim=True))      # softmax_backward
        dQ = (dS * scale) @ K
        dK = (dS * scale)^T @ Q

    ctx saves Q, K, V (originals; pre-GQA-expand), and P. Saving P is the
    "naive" memory strategy — O(B*H*L*S). Flash Attention saves only the
    normalization stats (L = log-sum-exp, M = max), recomputes P from Q/K
    in backward at O(B*H*L) extra compute but O(B*H*L) memory. We do not.

    Args
    ----
    query, key, value : Tensor
        Shapes (..., L, D_k), (..., S, D_k), (..., S, D_v).
    attn_mask : Tensor or None
        bool tensor → True keeps the position, False masks (sets score to -inf).
        float tensor → added to scores (use -inf or large negative to mask).
        Mutually exclusive with is_causal.
    is_causal : bool
        Lower-triangular causal mask (tril with diagonal=S-L offset, so it
        works for both L==S training and L<S cached generation).
    scale : float or None
        Defaults to 1/sqrt(d_k).
    enable_gqa : bool
        Grouped-query attention. If True and Q has more heads than K/V,
        repeats K/V along the heads dim by factor G = H_q / H_kv.
        Forward expands K/V via repeat_interleave; backward sums grads
        from the G replicas back into H_kv slots (via unflatten + sum).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if is_causal and attn_mask is not None:
            raise ValueError("is_causal and attn_mask are mutually exclusive")

        d_k = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d_k)

        # GQA: repeat K/V heads to match Q heads.
        H_q = query.shape[-3] if query.ndim >= 3 else 1
        H_kv = key.shape[-3] if key.ndim >= 3 else 1
        if enable_gqa and H_q != H_kv:
            assert H_q % H_kv == 0, f"GQA needs H_q ({H_q}) divisible by H_kv ({H_kv})"
            G = H_q // H_kv
            key_e = key.repeat_interleave(G, dim=-3)
            value_e = value.repeat_interleave(G, dim=-3)
        else:
            G = 1
            key_e, value_e = key, value

        # scores = Q @ K^T * scale. The matmul output is a fresh P-sized
        # tensor with no other refs, so we apply scale in-place to avoid
        # allocating a second P-sized tensor. Same in-place pattern is used
        # for the mask steps below — each one keeps `scores` as the single
        # P-sized buffer rather than rolling over to a new tensor.
        scores = query @ key_e.transpose(-2, -1)
        scores.mul_(scale)

        if is_causal:
            L, S = scores.shape[-2], scores.shape[-1]
            # Upper-left aligned causal (matches PyTorch SDPA semantics).
            # For L==S this is standard lower-triangular. For L != S, the
            # causal triangle sits at the top-left of the L×S grid, NOT
            # right-aligned to the K edge — this is PyTorch's documented
            # convention. Cached-generation use cases (where you want
            # right-aligned causal) should pass an explicit attn_mask.
            causal_mask = torch.ones(
                L, S, dtype=torch.bool, device=scores.device
            ).tril()
            scores.masked_fill_(~causal_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(~attn_mask, float("-inf"))
            else:
                scores.add_(attn_mask)

        probs = torch.softmax(scores, dim=-1)
        output = probs @ value_e

        ctx.save_for_backward(query, key, value, probs)
        ctx.scale = scale
        ctx.G = G
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
        query, key, value, probs = ctx.saved_tensors
        scale = ctx.scale
        G = ctx.G

        # Re-expand K/V to match probs's head count (cheap; same op as forward).
        if G > 1:
            key_e = key.repeat_interleave(G, dim=-3)
            value_e = value.repeat_interleave(G, dim=-3)
        else:
            key_e, value_e = key, value

        # dL/dV_e = P^T @ g
        grad_value_e = probs.transpose(-2, -1) @ grad_output

        # dL/dP = g @ V_e^T
        grad_probs = grad_output @ value_e.transpose(-2, -1)

        # softmax_backward: dL/dS = P * (dP - sum(P*dP, dim=-1, keepdim))
        # Note: masked positions have P=0, so dS=0 there — masks naturally
        # don't propagate grad. Also: this is the same formula nanoops's
        # Softmax uses (the "save y, not x" trick).
        #
        # Compact form: P*(dP - <P,dP>) = P*dP - P*<P,dP>. Compute pg = P*dP
        # once, then subtract `P * <P,dP>` in-place via addcmul_(value=-1).
        # That's a single fused kernel — no `P * <P,dP>` intermediate. Saves
        # one P-sized transient per SDPA backward (the (B,H,L,S) attention
        # matrix is the dominant memory term, so cutting one is meaningful).
        grad_scores = probs * grad_probs  # 1 P-size alloc
        inner = grad_scores.sum(dim=-1, keepdim=True)  # tiny (B,H,L,1)
        grad_scores.addcmul_(
            probs, inner, value=-1
        )  # in-place: grad_scores -= probs * inner

        # Fold scale into dS once, used by both dQ and dK. In-place since
        # grad_scores has no other references.
        grad_scores.mul_(scale)

        # dL/dQ = dS @ K_e
        grad_query = grad_scores @ key_e

        # dL/dK_e = dS^T @ Q
        grad_key_e = grad_scores.transpose(-2, -1) @ query

        # GQA backward: collapse G replicas of K/V back into H_kv heads.
        # repeat_interleave([a, b], G, dim=-3) -> [a]*G ++ [b]*G; backward
        # sums each G-block back to a single element.
        if G > 1:
            # (..., H_q, S, D) -> (..., H_kv, G, S, D) -> sum over G (dim=-3).
            grad_key = grad_key_e.unflatten(-3, (-1, G)).sum(dim=-3)
            grad_value = grad_value_e.unflatten(-3, (-1, G)).sum(dim=-3)
        else:
            grad_key = grad_key_e
            grad_value = grad_value_e

        # query, key, value, attn_mask, is_causal, scale, enable_gqa
        return grad_query, grad_key, grad_value, None, None, None, None


@_allow_in_graph
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Mirrors `F.scaled_dot_product_attention` (causal + mask + GQA)."""
    return ScaledDotProductAttention.apply(
        query, key, value, attn_mask, is_causal, scale, enable_gqa
    )


def _sliding_chunk_mask(
    q_start: int, q_end: int, k_start: int, k_end: int, W: int, device: torch.device
) -> torch.Tensor:
    """Build the (q_sz, k_sz) sliding+causal bool mask for one chunk.

    Identical formula used in SlidingWindowSDPA's forward AND backward —
    factored here so the two stay in lockstep. Each query at absolute
    position i allows key positions j ∈ [i - W + 1, i] (W keys back,
    including itself). The chunk's local (i_idx, j_idx) grid uses
    absolute positions via the +q_start / +k_start offsets, so the
    `j <= i` (causal) and `j >= i - W + 1` (sliding) tests stay shape-
    agnostic.

    The per-chunk arange tensors are tiny (~q_sz + k_sz int64) — we do
    NOT precompute a full (L, L) mask and slice, because each chunk's
    (q_sz, k_sz) is small enough that a fresh broadcast comparison is
    cheaper than the slice + dispatch overhead.
    """
    q_sz = q_end - q_start
    k_sz = k_end - k_start
    i_idx = torch.arange(q_sz, device=device).unsqueeze(1) + q_start
    j_idx = torch.arange(k_sz, device=device).unsqueeze(0) + k_start
    return (j_idx <= i_idx) & (j_idx >= i_idx - W + 1)


class SlidingWindowSDPA(torch.autograd.Function):
    """SDPA with sliding-window causal attention, internal chunking.

    Forward chunks queries into windows of `window_size`; each chunk computes
    attention only against the band of (up to) 2W-1 relevant keys instead of
    full L. All chunks live inside ONE autograd.Function — the Python chunk
    loop has no per-chunk autograd overhead. An earlier attempt that wrapped
    nF.scaled_dot_product_attention in an external Python loop was a wash
    (backward got slower) because each chunk paid full autograd.Function
    ctx + dispatch overhead. Pulling chunking inside one Function was the
    architectural fix.

    Backward symmetric: Q gradient is written per chunk (Q windows don't
    overlap); K/V gradients accumulate across chunks (K/V windows DO overlap
    between consecutive chunks).

    ctx saves Q, K, V, plus each chunk's P. Memory: ~`L * 2W` instead of
    full P's `L * L`, so for `W = L/4` the per-layer P footprint drops 4×.

    ─────────────────────────────────────────────────────────────────────
    Empirical findings (RTX 3090 ×2, nanchat d20, SSSL window pattern):
    ─────────────────────────────────────────────────────────────────────

    Isolated microbench (B=2, H=10, L=2048, D=128, W=512, bf16):
        full_via_sdpa (1 SDPA + L×L sliding mask): 2.29 ms fwd  / 5.46 ms fwd+bwd
        SlidingWindowSDPA:                         2.10 ms fwd  / 5.36 ms fwd+bwd
                                                   ↑ 1.09× fwd, ~1.02× fwd+bwd

    Early prediction from this microbench: ~0.5% end-to-end speedup.
    THIS WAS WRONG. Real training shows much bigger gains because the
    isolated bench can't capture the system-level effects of cutting P
    matrix memory across 20 layers.

    End-to-end training A/B (20-iter d20 base_train, 2× RTX 3090):

      Config                              tok/sec  MFU    dt    Peak     Total
      ──────────────────────────────────  ──────── ─────  ────  ──────── ──────
      PyTorch SDPA, B=2 (baseline)         22,725  46.2%  46.1s 16,464   n/a
      nanoops Lookup default, B=2          28,800  58.5%  36.4s 19,665   5.46m
      + SlidingWindowSDPA, B=2             30,594  62.2%  34.3s 17,612   5.14m
      + SlidingWindowSDPA + B=4 + exp_seg  32,678  66.4%  32.1s 22,722   4.81m
                                          ──────── ─────                ──────
                                          +43.8% vs baseline       -30% dt

    The B=4 row needs both SlidingWindowSDPA (cuts the P-matrix peak so
    B=4 doesn't immediately OOM) AND `PYTORCH_ALLOC_CONF=expandable_segments:True`
    (recovers 1-2 GiB of allocator fragmentation that otherwise pushes
    the run past the 24 GiB ceiling). Either one alone leaves the run
    OOM at the first training iter. With both, peak settles at
    ~22.7/24 GiB — tight but stable. nanoops/train.sh sets both by default.
    (The current train.sh default ALSO enables NANOOPS_MLP_CHECKPOINT=1,
    which trades a +7% wall-time cost for ~3.7 GiB of MLP activation
    memory savings — so the actual "today" stack at d20+B=4 is
    ~30,500 tok/s / 19 GiB peak / ~31 h ETA. The +44% above is the
    sliding-window stack's peak measurement; MLP checkpoint exists to
    trade some of that wall time back for headroom that unlocks the
    larger --depth=24 config.)

    Loss curves match across all rows to within ~6e-5 bf16 rounding noise.

    Why the end-to-end win is so much bigger than the 1.09× isolated
    microbench would predict:
      1. P matrix shrinks (L, L) → (L, 2W) per sliding layer. 15 sliding
         layers × ~167 MB saved per layer ≈ 2.5 GB peak savings (2 GB
         observed after subtracting chunked ctx overhead).
      2. Less HBM bandwidth pressure (smaller P → fewer attention reads).
      3. Headroom freed up by the P-matrix cut UNLOCKS B=4 entirely — and
         B=4 alone is +6.8% tok/sec on top of the sliding savings because
         GEMMs hit better utilization at the larger batch.
      4. Allocator no longer hugging the 24 GB ceiling — less
         fragmentation, fewer cache-miss alloc paths.

    Warmup is ~60s longer because torch.compile must compile the chunked
    forward+backward path (one autograd.Function with 4 chunks unrolled
    into the trace). One-time cost amortized across the full training run.
    ─────────────────────────────────────────────────────────────────────

    Lesson: micro-benchmark isolated attention timing is a POOR predictor
    of end-to-end attention-kernel impact at this scale. Memory pressure
    and HBM-bandwidth side effects can dominate the small per-call speedup.
    Always validate with a real training run.

    Always on when NANOOPS=1. The patched _sdpa_attention now routes
    *all* training attention (both sliding S layers and full-attention
    L layers, when Tq == Tk) through this Function — unifying the
    autograd graph shape and making peak memory easier to reason about.
    Inference paths (Tq == 1, cached gen with Tq != Tk) still use
    PyTorch's native SDPA.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        window_size: int,
        enable_gqa: bool = False,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        B, H_q, L, D = query.shape
        H_kv = key.shape[-3]
        W = window_size
        # Default chunk size is W // 2 — more chunks (8 vs 4 for nanchat W=512)
        # but each chunk's score matrix is narrower (C + W - 1 cells per row
        # instead of 2W - 1), so less compute wasted on masked positions.
        # Trade-off vs C == W: more Python dispatch overhead, smaller GEMMs
        # but less wasted FLOPs. Net is hardware-dependent; microbench to pick.
        C = chunk_size if chunk_size is not None else max(1, W // 2)
        scale_val = 1.0 / math.sqrt(D)

        # GQA: repeat K/V heads to match Q's head count. Same trick as in
        # ScaledDotProductAttention. Saved ctx keeps the ORIGINAL (un-
        # expanded) K and V so we don't pay the G× memory tax twice; the
        # expansion is recomputed cheaply in backward.
        if enable_gqa and H_q != H_kv:
            assert H_q % H_kv == 0, f"GQA needs H_q ({H_q}) divisible by H_kv ({H_kv})"
            G = H_q // H_kv
            key_e = key.repeat_interleave(G, dim=-3)
            value_e = value.repeat_interleave(G, dim=-3)
        else:
            G = 1
            key_e, value_e = key, value

        output = torch.empty_like(query)
        chunk_info = []
        # Flash-style: save only LSE per chunk, not the full P matrix.
        # Each LSE is (..., q_sz, 1) — 1 / k_sz the size of P. We re-derive
        # P in backward from `(scores - lse).exp()`. The cost: 1 extra
        # Q @ K^T matmul per chunk in backward; the saving: ~k_sz× less
        # ctx memory per chunk (~700-900 MB total at nanchat d24 scale).
        lse_list = []

        for q_start in range(0, L, C):
            q_end = min(q_start + C, L)
            # Sliding-window key range still depends on W (not C): each query
            # sees its last W keys regardless of how we chunk the queries.
            k_start = max(0, q_start - W + 1)
            k_end = q_end

            q_blk = query[..., q_start:q_end, :]
            k_blk = key_e[..., k_start:k_end, :]
            v_blk = value_e[..., k_start:k_end, :]

            scores = q_blk @ k_blk.transpose(-2, -1)
            scores.mul_(scale_val)

            mask = _sliding_chunk_mask(q_start, q_end, k_start, k_end, W, query.device)
            scores.masked_fill_(~mask, float("-inf"))

            # logsumexp is numerically stable AND ignores -inf entries
            # (exp(-inf) = 0 contributes nothing to the sum). Sliding+
            # causal guarantees at least one allowed key per query (the
            # diagonal `j == i` is always True), so LSE is never -inf
            # and `(scores - lse).exp()` is never NaN.
            lse = scores.logsumexp(dim=-1, keepdim=True)  # (..., q_sz, 1)
            # Reuse `scores`'s storage for `probs`: scores has no other
            # refs after this point in the loop. Saves one P-sized alloc.
            probs = scores.sub_(lse).exp_()  # transient
            output[..., q_start:q_end, :] = probs @ v_blk

            chunk_info.append((q_start, q_end, k_start, k_end))
            lse_list.append(lse)

        ctx.save_for_backward(query, key, value, *lse_list)
        ctx.chunk_info = chunk_info
        ctx.scale = scale_val
        ctx.G = G
        # Explicit int() cast — W came from the (non-tensor) window_size arg
        # which could be int / numpy int / 0-D tensor depending on caller.
        # Backward's mask helper uses W in arithmetic on torch.arange tensors;
        # forcing Python int avoids ambiguity (no implicit promotion to a
        # 0-D CUDA tensor inside the arithmetic).
        ctx.window_size = int(W)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        saved = ctx.saved_tensors
        query, key, value = saved[0], saved[1], saved[2]
        lse_list = saved[3:]
        chunk_info = ctx.chunk_info
        scale_val = ctx.scale
        G = ctx.G
        W = ctx.window_size

        # Re-expand K/V to match Q's head count for the chunked matmuls;
        # collapse back at the end. Same pattern as in ScaledDotProductAttention.
        if G > 1:
            key_e = key.repeat_interleave(G, dim=-3)
            value_e = value.repeat_interleave(G, dim=-3)
        else:
            key_e, value_e = key, value

        grad_query = torch.empty_like(query)
        # K and V gradient buffers are zero-init because consecutive chunks
        # have OVERLAPPING key ranges — their dK/dV contributions get
        # accumulated via `+=` below. NOTE: in bf16 this accumulation is
        # NOT bitwise-reproducible across runs: each chunk's contribution
        # is added in chunk order, and bf16 add is non-associative. parity
        # tests on this op observed ~1e-2 max diff in k.grad/v.grad vs the
        # one-big-matmul reference — that's within bf16 rounding noise but
        # don't bit-compare across implementations.
        grad_key_e = torch.zeros_like(key_e)
        grad_value_e = torch.zeros_like(value_e)

        for (q_start, q_end, k_start, k_end), lse in zip(chunk_info, lse_list):
            q_blk = query[..., q_start:q_end, :]
            k_blk = key_e[..., k_start:k_end, :]
            v_blk = value_e[..., k_start:k_end, :]
            g_out_blk = grad_output[..., q_start:q_end, :]

            # Flash-style: re-materialize P from the saved LSE and a fresh
            # Q@K^T matmul. This is the price of not having P in ctx — one
            # extra (q_sz, k_sz) matmul per chunk. P lives only for the
            # duration of this chunk's backward block, then is freed.
            # Note on bf16 drift: this Q@K^T is a separate cuBLAS call
            # from the forward's; cuBLAS bf16 matmul is determinstic given
            # the same inputs, so scores here match forward's scores
            # bitwise. The LSE was computed from forward's scores, so
            # P = (scores_bwd - lse_fwd).exp() = exp(scores_fwd - lse) by
            # transitivity. parity tests verify to fp64 precision.
            scores = q_blk @ k_blk.transpose(-2, -1)
            scores.mul_(scale_val)
            mask = _sliding_chunk_mask(q_start, q_end, k_start, k_end, W, query.device)
            scores.masked_fill_(~mask, float("-inf"))
            # P = exp(scores - LSE) — numerically stable because scores ≤ LSE
            # row-wise by construction; equivalent to softmax(scores) but
            # uses the LSE we already paid for in forward. Reuses scores'
            # storage (no other refs left in this iteration).
            probs = scores.sub_(lse).exp_()

            grad_v_blk = probs.transpose(-2, -1) @ g_out_blk
            grad_probs = g_out_blk @ v_blk.transpose(-2, -1)
            pg = probs * grad_probs
            inner = pg.sum(dim=-1, keepdim=True)
            pg.addcmul_(probs, inner, value=-1)
            pg.mul_(scale_val)
            grad_q_blk = pg @ k_blk
            grad_k_blk = pg.transpose(-2, -1) @ q_blk

            grad_query[..., q_start:q_end, :] = grad_q_blk
            grad_key_e[..., k_start:k_end, :] += grad_k_blk
            grad_value_e[..., k_start:k_end, :] += grad_v_blk

        # GQA: collapse the G replicas back to H_kv via unflatten + sum.
        if G > 1:
            grad_key = grad_key_e.unflatten(-3, (-1, G)).sum(dim=-3)
            grad_value = grad_value_e.unflatten(-3, (-1, G)).sum(dim=-3)
        else:
            grad_key = grad_key_e
            grad_value = grad_value_e

        # window_size, enable_gqa, chunk_size are non-differentiable
        return grad_query, grad_key, grad_value, None, None, None


@_allow_in_graph
def sliding_window_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
    enable_gqa: bool = False,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Sliding-window causal SDPA. window_size = max number of past keys
    each query attends to (including itself), so window_size=L is full
    causal attention. Set enable_gqa=True for Grouped-Query Attention
    (H_q > H_kv); K/V get repeat_interleave'd to match H_q.

    chunk_size controls how many queries each chunk processes. Decoupled
    from window_size (defaults to window_size // 2 — see SlidingWindowSDPA
    docstring for the C vs W trade-off table). Smaller chunk_size → less
    wasted compute on masked positions but more Python dispatch overhead;
    larger chunk_size → bigger GEMMs but more cells get masked out.

    Hooked into nanchat by `nanoops.integration.patch_nanchat()`: replaces
    nanchat.flash_attention._sdpa_attention so sliding training layers
    route here. Always on when NANOOPS=1. See SlidingWindowSDPA docstring
    for the empirical end-to-end findings (+6.2% tok/sec, -10.4% peak memory).
    """
    return SlidingWindowSDPA.apply(
        query, key, value, window_size, enable_gqa, chunk_size
    )
