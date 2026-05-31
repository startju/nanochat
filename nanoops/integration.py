"""Monkey-patch helpers to swap nanoops into nanchat.

Two layers of patching:

1. **F namespace** — replaces the `F` (torch.nn.functional) attribute on each
   target module with a proxy whose specific attributes route through nanoops.
   Everything not overridden falls through to the real `torch.nn.functional`.

2. **torch top-level functions** — overrides individual attributes on the
   `torch` module itself (sigmoid, tanh). nanchat calls these as
   `torch.sigmoid(x)` / `torch.tanh(x)`, not via the F namespace. Targeted
   per-attribute swaps (NOT a full-module proxy) so the rest of `torch`
   stays untouched.

Two entry points:
  - `patch_nanchat()`: permanent swap (used by `scripts/base_train.py` when
    `NANOOPS=1` env var is set).
  - `patched()`: context manager that swaps in and restores on exit (used by
    `tests/test_nanchat_integration.py` to compare PyTorch vs nanoops paths).

Patched ops:
  F-namespace (auto-swapped on the modules in `_TARGET_MODULES`):
    F.linear, F.embedding, F.rms_norm, F.cross_entropy, F.softmax,
    F.scaled_dot_product_attention
  torch top-level attributes:
    torch.sigmoid, torch.tanh
  Module-level functions (replaced directly on the host module):
    nanochat.gpt.apply_rotary_emb
  Class methods (targeted swaps, can't be reached via F-namespace):
    nanochat.gpt.MLP.forward             — relu_square fused
                                             + optional MLP activation ckpt
    nanochat.gpt.GPT.forward             — optional fused attention QKV path
    nanochat.gpt.CausalSelfAttention.forward
                                          — optional L-layer activation ckpt
    nanochat.optim.DistMuonAdamW._compute_adamw / _compute_muon
                                          — CPU optim state offload (dual-GPU)
    nanochat.optim.MuonAdamW._step_adamw / _step_muon
                                          — CPU optim state offload (single-GPU)

NOT patched (intentional):
  - F.relu — nanchat does `F.relu(x).square()` (a chain of two ops); nanoops's
    fused `relu_square` would need to replace the whole chain at a different
    call site, not the `F.relu` slot.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Iterator

import torch
import torch.nn.functional as F_orig
import torch.utils.checkpoint as _ckpt

import nanoops.functional as nF
from nanochat.common import COMPUTE_DTYPE


# Captured at _apply() time so the L-attn checkpoint wrapper can call the
# un-patched original CausalSelfAttention.forward. _restore() resets to None.
_orig_attn_forward = None
_orig_gpt_forward = None


_F_OVERRIDES = {
    "linear": nF.linear,
    "embedding": nF.embedding,
    "rms_norm": nF.rms_norm,
    "cross_entropy": nF.cross_entropy,
    "softmax": nF.softmax,
    "scaled_dot_product_attention": nF.scaled_dot_product_attention,
}

_TORCH_OVERRIDES = {
    # Patched as attributes directly on the `torch` module — nanchat calls
    # `torch.sigmoid(x)` / `torch.tanh(x)` (not via F namespace). Scope is the
    # whole Python process, but only these two names are touched; the rest of
    # `torch` is untouched.
    "sigmoid": nF.sigmoid,
    "tanh": nF.tanh,
}


# Module-level functions in nanchat that have nanoops equivalents but are NOT
# looked up via F namespace (so the F-namespace patch doesn't touch them).
# Format: (module_path, attr_name) -> replacement.
_MODULE_FUNC_OVERRIDES = {
    # nanchat defines apply_rotary_emb as a top-level function in gpt.py and
    # calls it directly via module lookup (`gpt.apply_rotary_emb(...)`).
    # Replacing this attribute swaps in nanoops's autograd Function version.
    ("nanochat.gpt", "apply_rotary_emb"): nF.apply_rotary_emb,
}


def _patched_mlp_forward(self, x):
    """nanchat MLP.forward, but `F.relu(x).square()` -> `nF.relu_square(x)`.

    `F.relu(x).square()` is a TWO-op chain: F.relu returns a fresh tensor,
    then `.square()` is a tensor method on it. Neither a single F-namespace
    patch nor a single torch.X patch can route the chain to nanoops's fused
    `relu_square` — we have to replace the whole forward. nanchat has exactly
    one such site (`gpt.py:137` MLP.forward), so a targeted class-method
    swap is the minimal fix.
    """
    if os.environ.get("NANOOPS_MLP_CHECKPOINT"):
        return _ckpt.checkpoint(_mlp_inner, self, x, use_reentrant=False)
    return _mlp_inner(self, x)


def _mlp_inner(self, x):
    """The actual MLP forward — same as the patched body, separated out so
    activation checkpointing can wrap it cleanly."""
    x = self.c_fc(x)
    x = nF.relu_square(x)
    x = self.c_proj(x)
    return x


def _patched_block_forward(self, x, ve, cos_sin, window_size, kv_cache):
    """Block.forward with the mlp side replaced by `fused_mlp`.

    Original (nanchat.gpt.Block.forward):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))

    Patched: attn side unchanged; the second line collapses
    `norm + c_fc + relu² + c_proj + outer-residual-add` into one fused
    call (3 Triton kernels fwd + 4 Triton kernels bwd, see TRITON_zh.md
    §3). The public fused call keeps `(B,T,C)`; its wrapper flattens to
    `M=B*T` before launching Triton kernels.

    nanochat's `norm()` is `F.rms_norm(x, (x.size(-1),))` — plain
    RMSNorm without per-channel affine (see gpt.py:42), so the fused op
    has no norm-weight parameter.

    Inference paths (kv_cache present) bypass the fusion — the kv_cache
    interaction with the (B,T,C) → 2D reshape needs more thought and
    inference time isn't on the bench path.
    """
    # attn side — unchanged
    x = x + self.attn(_orig_norm(x), ve, cos_sin, window_size, kv_cache)
    # mlp side — fused (training, CUDA only)
    if kv_cache is not None or not x.is_cuda:
        # Inference (kv_cache present) or CPU (e.g. tiny test harness):
        # fall back to the original mlp+norm+residual chain.
        return x + self.mlp(_orig_norm(x))
    # `.contiguous()` is cheap when already contiguous (returns same tensor);
    # if attn produced a non-contiguous view (e.g. via stride tricks) the
    # fused kernel needs contiguous input for its index arithmetic.
    # Weight dtype cast (fp32 master → bf16 activation) is handled inside
    # `fused_mlp` itself, so we pass the raw module weights here.
    return _fused_mlp(
        x.contiguous(),
        self.mlp.c_fc.weight,
        self.mlp.c_proj.weight,
    )


# Captured at _apply() time so `_patched_block_forward` can call the
# original `norm` function (since the F-namespace patch may have routed
# `nanchat.gpt.F` through nanoops).
_orig_norm = None
_fused_mlp = None
_fused_attn_qkv = None
_fused_attn_tail = None


def _rms_norm_no_weight(x: torch.Tensor) -> torch.Tensor:
    return nF.rms_norm(x, (x.size(-1),))


def _attn_qkv_residual(
    self,
    x,
    x0,
    resid_scale,
    x0_scale,
    ve_ids,
    ve_weight,
    cos_sin,
    window_size,
):
    """Attention residual using fused residual/x0 mix + QKV/rotary/QK-norm.

    This replaces the training pair:
        x = resid_lambdas[i] * x + x0_lambdas[i] * x0
        x = x + self.attn(norm(x), ve, ...)

    The fusion must live above `CausalSelfAttention.forward` because nanchat
    applies both the residual/x0 blend and `norm(x)` one level up from the
    Q/K/V linears. `ve_ids`/`ve_weight` are passed separately so the Triton op
    can fuse the value-embedding lookup instead of consuming a precomputed
    `ve` tensor.
    """
    B, T, _C = x.shape
    attn = self.attn
    has_ve = ve_weight is not None
    ve_gate_weight = attn.ve_gate.weight if has_ve else None
    q, k, v, x_mix = _fused_attn_qkv(
        x,
        x0,
        resid_scale,
        x0_scale,
        ve_ids if has_ve else None,
        ve_weight,
        attn.ve_gate_channels if has_ve else 1,
        ve_gate_weight,
        attn.c_q.weight,
        attn.c_k.weight,
        attn.c_v.weight,
        cos_sin[0],
        cos_sin[1],
        attn.n_head,
        attn.n_kv_head,
        attn.head_dim,
        1.2,
        1e-6,
    )
    window = window_size[0]
    sdpa_window = T if (window < 0 or window >= T) else window + 1
    return _fused_attn_tail(q, k, v, attn.c_proj.weight, x_mix, sdpa_window)


def _attn_native_residual(self, x, ve, cos_sin, window_size):
    """Attention residual matching the non-QKV-fused Block.forward path."""
    return x + self.attn(_rms_norm_no_weight(x), ve, cos_sin, window_size, None)


def _mlp_residual(self, x):
    if _fused_mlp is None:
        return x + self.mlp(_rms_norm_no_weight(x))

    return _fused_mlp(
        x.contiguous(),
        self.mlp.c_fc.weight,
        self.mlp.c_proj.weight,
    )


def _patched_gpt_forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
    """GPT.forward with fused training attention QKV.

    Falls back to the original forward for inference/KV-cache and non-CUDA
    paths. Training base_train uses fixed-shape CUDA batches, which is the path
    this fusion is tuned for.
    """
    if kv_cache is not None or not idx.is_cuda:
        return _orig_gpt_forward(self, idx, targets, kv_cache, loss_reduction)

    B, T = idx.size()

    assert T <= self.cos.size(1), (
        f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
    )
    assert idx.device == self.cos.device, (
        f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
    )
    assert self.cos.dtype == COMPUTE_DTYPE, (
        f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
    )
    cos_sin = self.cos[:, :T], self.sin[:, :T]

    x = self.transformer.wte(idx)
    x = x.to(COMPUTE_DTYPE)
    x = _rms_norm_no_weight(x)

    # Smear: same training path as nanochat.gpt.GPT.forward.
    assert T > 1, "Training forward pass should have T > 1"
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

    x0 = x
    n_layer = self.config.n_layer
    backout_layer = n_layer // 2
    x_backout = None
    qkv_mode = os.environ.get("NANOOPS_FUSED_ATTN_QKV_MODE", "all")
    for i, block in enumerate(self.transformer.h):
        resid_scale = self.resid_lambdas[i]
        x0_scale = self.x0_lambdas[i]
        ve_weight = (
            self.value_embeds[str(i)].weight
            if str(i) in self.value_embeds
            else None
        )
        use_fused_qkv = (
            qkv_mode == "all"
            or (qkv_mode == "ve" and ve_weight is not None)
            or (qkv_mode == "no_ve" and ve_weight is None)
        )
        is_full_attn = self.window_sizes[i][0] < 0 or self.window_sizes[i][0] >= T
        if use_fused_qkv:
            if os.environ.get("NANOOPS_L_ATTN_CHECKPOINT") and is_full_attn:
                x = _ckpt.checkpoint(
                    _attn_qkv_residual,
                    block,
                    x,
                    x0,
                    resid_scale,
                    x0_scale,
                    idx if ve_weight is not None else None,
                    ve_weight,
                    cos_sin,
                    self.window_sizes[i],
                    use_reentrant=False,
                )
            else:
                x = _attn_qkv_residual(
                    block,
                    x,
                    x0,
                    resid_scale,
                    x0_scale,
                    idx if ve_weight is not None else None,
                    ve_weight,
                    cos_sin,
                    self.window_sizes[i],
                )
        else:
            ve = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if ve_weight is not None
                else None
            )
            x = resid_scale * x + x0_scale * x0
            x = _attn_native_residual(block, x, ve, cos_sin, self.window_sizes[i])
        x = _mlp_residual(block, x)
        if i == backout_layer:
            x_backout = x

    if x_backout is not None:
        x = x - self.backout_lambda.to(x.dtype) * x_backout
    x = _rms_norm_no_weight(x)

    softcap = 15
    logits = self.lm_head(x)
    logits = logits[..., : self.config.vocab_size]
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)

    if targets is not None:
        return nF.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
    return logits


def _patched_l_attn_forward(self, x, ve, cos_sin, window_size, kv_cache):
    """Activation-checkpoint the full-attention (L) layers. Only the L
    layers are checkpointed — sliding-window S layers already use
    chunked SDPA with LSE-only ctx so they're cheap. For tight-memory
    setups (single-GPU d24), this frees enough headroom to clear the
    activation-driven OOM cliff that CPU offload alone doesn't catch.

    Only installed by `_apply()` when NANOOPS_L_ATTN_CHECKPOINT=1; if
    the env var isn't set the patch isn't applied at all, so this
    wrapper is never reached (no per-call env-var lookup cost).
    Inference paths (kv_cache present) bypass the checkpoint.
    """
    if kv_cache is None and (window_size[0] < 0 or window_size[0] >= x.size(1)):
        return _ckpt.checkpoint(
            _orig_attn_forward,
            self,
            x,
            ve,
            cos_sin,
            window_size,
            kv_cache,
            use_reentrant=False,
        )
    return _orig_attn_forward(self, x, ve, cos_sin, window_size, kv_cache)


_TARGET_MODULES = [
    # nanchat modules — call F.* directly
    "nanochat.gpt",
    "nanochat.flash_attention",
    "nanochat.engine",
    # PyTorch nn module internals — `nn.X.forward()` looks up F.* from
    # its own host module's namespace, not from any of the nanchat modules
    # above. nanchat overrides `Linear.forward` so `nn.Linear` is moot in
    # practice, but `nn.Embedding` is used as-is (wte, value_embeds), and
    # bare `nn.Linear` could slip in via future code. Patching here is
    # cheap insurance — within one NANOOPS=1 process, replacing PyTorch's
    # F.embedding / F.linear with nanoops's is exactly the intent.
    "torch.nn.modules.sparse",  # nn.Embedding -> F.embedding
    "torch.nn.modules.linear",  # nn.Linear -> F.linear
    "torch.nn.modules.normalization",  # nn.RMSNorm / nn.LayerNorm -> F.rms_norm / F.layer_norm
    "torch.nn.modules.loss",  # nn.CrossEntropyLoss -> F.cross_entropy
    "torch.nn.modules.activation",  # nn.Softmax -> F.softmax
]


def _make_patched_F() -> object:
    """Build a proxy: nanoops overrides win; everything else falls through to F."""
    proxy = type("PatchedF", (), {})()
    for attr in dir(F_orig):
        if not attr.startswith("_"):
            setattr(proxy, attr, getattr(F_orig, attr))
    for name, op in _F_OVERRIDES.items():
        setattr(proxy, name, op)
    return proxy


_PATCHED = False  # set to True after _apply() runs; cleared by _restore()


def _apply() -> dict[str, dict]:
    """Apply F-namespace + torch.X + module-func + method patches.
    Returns originals dict for restore.

    Idempotence guard: refuses to re-patch if a previous _apply() hasn't
    been _restore()'d. Without this guard, a second _apply() would
    capture the ALREADY-patched class methods / module funcs as
    "originals", and the restore path would put back patched versions
    instead of the true originals.
    """
    global _PATCHED, _orig_attn_forward, _orig_gpt_forward, _orig_norm
    global _fused_mlp, _fused_attn_qkv, _fused_attn_tail
    if _PATCHED:
        raise RuntimeError(
            "nanoops.integration already patched; call _restore() before _apply() again"
        )
    originals: dict[str, dict] = {"F": {}, "torch": {}, "module_func": {}, "method": {}}
    # F-namespace patch
    proxy = _make_patched_F()
    for modname in _TARGET_MODULES:
        mod = importlib.import_module(modname)
        if hasattr(mod, "F"):
            originals["F"][modname] = mod.F
            mod.F = proxy
    # torch.X attribute patch
    for name, op in _TORCH_OVERRIDES.items():
        originals["torch"][name] = getattr(torch, name)
        setattr(torch, name, op)
    # Module-level function attribute patch (e.g. nanchat.gpt.apply_rotary_emb)
    for (modname, attr_name), replacement in _MODULE_FUNC_OVERRIDES.items():
        mod = importlib.import_module(modname)
        originals["module_func"][(modname, attr_name)] = getattr(mod, attr_name)
        setattr(mod, attr_name, replacement)
    # MLP.forward: route the `F.relu(x).square()` chain to fused relu_square
    # (and optionally activation-checkpoint via NANOOPS_MLP_CHECKPOINT=1).
    gpt_mod = importlib.import_module("nanochat.gpt")
    originals["method"][("nanochat.gpt", "MLP", "forward")] = gpt_mod.MLP.forward
    gpt_mod.MLP.forward = _patched_mlp_forward
    # Block.forward: opt-in via NANOOPS_FUSED_MLP=1. Replaces the
    # mlp half of Block.forward — `x + mlp(norm(x))` — with a single
    # `fused_mlp(x, W_fc, W_proj)` call that collapses
    # norm + c_fc + relu² + c_proj + outer residual into 3 fwd Triton
    # kernels + 4 bwd Triton kernels.
    # See nanoops/TRITON_zh.md §3 for the fusion breakdown. Supersedes
    # the relu_square fusion (which is a subset of what's fused here).
    if os.environ.get("NANOOPS_FUSED_MLP"):
        from .triton_fused_mlp import fused_mlp as _fmb

        assert _orig_norm is None, "_orig_norm already captured — call _restore() first"
        _orig_norm = gpt_mod.norm
        _fused_mlp = _fmb
        originals["method"][("nanochat.gpt", "Block", "forward")] = (
            gpt_mod.Block.forward
        )
        gpt_mod.Block.forward = _patched_block_forward
    # Fused attention QKV: opt-in via NANOOPS_FUSED_ATTN_QKV=1. This is
    # installed at GPT.forward instead of CausalSelfAttention.forward because
    # the fused op owns the outer RMSNorm that nanchat applies one level up in
    # Block.forward, and it needs token ids + VE table to fuse value embedding.
    if os.environ.get("NANOOPS_FUSED_ATTN_QKV"):
        from .triton_fused_attn_output import attn_output_proj_residual as _tail
        from .triton_fused_attn_qkv import norm_qkv_projection_with_residual_mix as _nqp

        assert _orig_gpt_forward is None, (
            "_orig_gpt_forward already captured — call _restore() before _apply()"
        )
        _orig_gpt_forward = gpt_mod.GPT.forward
        _fused_attn_qkv = _nqp
        _fused_attn_tail = _tail
        originals["method"][("nanochat.gpt", "GPT", "forward")] = _orig_gpt_forward
        gpt_mod.GPT.forward = _patched_gpt_forward
    # L-layer activation checkpoint: opt-in via NANOOPS_L_ATTN_CHECKPOINT=1.
    # Installed conditionally so when the env var is OFF, attention forward
    # stays the original (no per-call wrapper cost). Env var read once here
    # at patch time, same pattern as NANOOPS_OFFLOAD_OPTIM below.
    if os.environ.get("NANOOPS_L_ATTN_CHECKPOINT"):
        assert _orig_attn_forward is None, (
            "_orig_attn_forward already captured — call _restore() before _apply()"
        )
        _orig_attn_forward = gpt_mod.CausalSelfAttention.forward
        originals["method"][("nanochat.gpt", "CausalSelfAttention", "forward")] = (
            _orig_attn_forward
        )
        gpt_mod.CausalSelfAttention.forward = _patched_l_attn_forward
    # Optimizer CPU offload (opt-in via NANOOPS_OFFLOAD_OPTIM=1). Moves
    # DistMuonAdamW's per-rank optim state to CPU pinned memory; H2D/D2H
    # per optimizer step. Freed GPU memory: ~2.5 GB Muon state + ~300 MB
    # AdamW state per rank, enough to clear d24+B=1's fragmentation OOM
    # cliff. See nanoops/cpu_offload.py for the patch bodies.
    if os.environ.get("NANOOPS_OFFLOAD_OPTIM"):
        from . import cpu_offload

        optim_mod = importlib.import_module("nanochat.optim")
        # Distributed path (>1 GPU)
        dist_cls = optim_mod.DistMuonAdamW
        originals["method"][("nanochat.optim", "DistMuonAdamW", "_compute_adamw")] = (
            dist_cls._compute_adamw
        )
        originals["method"][("nanochat.optim", "DistMuonAdamW", "_compute_muon")] = (
            dist_cls._compute_muon
        )
        dist_cls._compute_adamw = cpu_offload.patched_compute_adamw
        dist_cls._compute_muon = cpu_offload.patched_compute_muon
        # Single-GPU path
        single_cls = optim_mod.MuonAdamW
        originals["method"][("nanochat.optim", "MuonAdamW", "_step_adamw")] = (
            single_cls._step_adamw
        )
        originals["method"][("nanochat.optim", "MuonAdamW", "_step_muon")] = (
            single_cls._step_muon
        )
        single_cls._step_adamw = cpu_offload.patched_step_adamw
        single_cls._step_muon = cpu_offload.patched_step_muon
    _PATCHED = True  # global declared at top of _apply()
    return originals


def _restore(originals: dict[str, dict]) -> None:
    for modname, original_F in originals["F"].items():
        importlib.import_module(modname).F = original_F
    for name, op in originals["torch"].items():
        setattr(torch, name, op)
    for (modname, attr_name), original in originals["module_func"].items():
        setattr(importlib.import_module(modname), attr_name, original)
    for (modname, cls_name, method_name), original in originals["method"].items():
        cls = getattr(importlib.import_module(modname), cls_name)
        setattr(cls, method_name, original)
    global _PATCHED, _orig_attn_forward, _orig_gpt_forward, _orig_norm
    global _fused_mlp, _fused_attn_qkv, _fused_attn_tail
    _PATCHED = False
    _orig_attn_forward = None
    _orig_gpt_forward = None
    _orig_norm = None
    _fused_mlp = None
    _fused_attn_qkv = None
    _fused_attn_tail = None


def patch_nanchat() -> list[str]:
    """Permanently swap nanoops into nanchat. Returns the list of patched op names."""
    _apply()
    names = (
        [f"F.{n}" for n in _F_OVERRIDES]
        + [f"torch.{n}" for n in _TORCH_OVERRIDES]
        + [f"{mod}.{attr}" for mod, attr in _MODULE_FUNC_OVERRIDES]
        + ["MLP.forward(relu_square fused)"]
    )
    if os.environ.get("NANOOPS_MLP_CHECKPOINT"):
        names.append("MLP.forward(activation checkpoint)")
    if os.environ.get("NANOOPS_OFFLOAD_OPTIM"):
        names.append("MuonAdamW/DistMuonAdamW(CPU optim state offload)")
    if os.environ.get("NANOOPS_L_ATTN_CHECKPOINT"):
        names.append("CausalSelfAttention.forward(L-only activation checkpoint)")
    if os.environ.get("NANOOPS_FUSED_MLP"):
        names.append("Block.forward(fused_mlp — supersedes relu_square fusion)")
    if os.environ.get("NANOOPS_FUSED_ATTN_QKV"):
        names.append("GPT.forward(norm_qkv_projection + attn_output_proj_residual fused)")
    return names


def maybe_patch_nanchat(env_var: str = "NANOOPS") -> bool:
    """If `$NANOOPS` is set (any truthy value), apply the patch and print
    the swap summary. Returns whether the patch was applied.

    This is the entry point training scripts (`scripts/base_train.py`)
    call unconditionally — centralizing the env-var check so call sites
    stay one line.
    """
    if not os.environ.get(env_var):
        return False
    patched_names = patch_nanchat()
    print(f"[nanoops] swapped in: {', '.join(patched_names)}")
    return True


@contextlib.contextmanager
def patched() -> Iterator[None]:
    """Temporarily swap nanoops in; restore PyTorch ops on exit."""
    originals = _apply()
    try:
        yield
    finally:
        _restore(originals)
