"""CPU offload of optimizer state for nanchat's Muon + AdamW optimizers.

Drop-in replacements for the per-step optimizer methods on BOTH
nanchat optimizer classes: state (exp_avg / exp_avg_sq for AdamW,
momentum_buffer / second_momentum_buffer for Muon) lives in pinned
CPU memory; per optimizer step we H2D copy to GPU, run the existing
fused kernel, then D2H back.

Wired in by `nanoops.integration._apply` when `NANOOPS_OFFLOAD_OPTIM=1`;
reversed by `_restore` via the standard `originals` dict.

Two class pairs are handled:
  - `DistMuonAdamW` (distributed, >1 GPU): patches `_compute_adamw` and
    `_compute_muon` — state lives sharded across ranks via ZeRO-1, this
    moves each rank's slice to CPU.
  - `MuonAdamW` (single GPU): patches `_step_adamw` and `_step_muon` —
    full-size state lives on CPU (no ZeRO sharding to begin with).

The single-GPU path is reached when `nanoops/train.sh` is invoked with
NPROC=1 (no torchrun → no RANK env → nanchat's Factory picks MuonAdamW).

Limitation: state loaded from a checkpoint comes back on whichever
device the checkpoint stored. The first optim step after resume re-pays
the H2D + D2H cost from a non-pinned CPU buffer; subsequent steps
re-pin via the lazy alloc path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

# Imported at module top — the file is only imported from
# nanoops.integration._apply() when NANOOPS_OFFLOAD_OPTIM=1, by which
# time nanchat.optim is guaranteed already loaded (no circular import).
from nanochat.optim import adamw_step_fused, muon_step_fused


def _cpu_pinned_zeros(
    shape: torch.Size | tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Optimizer state on the CPU side: zeros tensor pinned for async DMA."""
    return torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)


def migrate_optimizer_state_to_cpu_pinned(optimizer: Any) -> None:
    """Walk `optimizer.state` and migrate every GPU-resident tensor back to
    CPU pinned memory. Call this ONCE immediately after
    `optimizer.load_state_dict(loaded_state)` when a checkpoint resume put
    state on GPU. base_train.py calls this when NANOOPS_OFFLOAD_OPTIM=1.

    Doing it eagerly (before the training loop's first forward) means the
    GPU has ~17 GiB headroom for the transient "old GPU tensor still alive
    while the new CPU pinned tensor is being allocated" phase. The
    patched_*_compute / patched_step_* functions can then assume state is
    CPU-resident on entry, no per-iter device-check overhead.
    """
    for p_state in optimizer.state.values():
        for k, v in list(p_state.items()):
            if isinstance(v, torch.Tensor) and v.device.type != "cpu":
                p_state[k] = v.detach().to("cpu", non_blocking=False).pin_memory()


def patched_compute_adamw(
    self: Any,
    group: dict[str, Any],
    info: dict[str, Any],
    gather_list: list[dict[str, Any]],
    rank: int,
    world_size: int,
) -> None:
    """CPU-offloaded version of DistMuonAdamW._compute_adamw.

    State (exp_avg, exp_avg_sq) lives in pinned CPU memory; per optimizer
    step we H2D copy to GPU buffers, run the fused AdamW kernel, then
    D2H back. Dual-GPU only (single-GPU goes through the simpler
    patched_step_adamw via train.sh's NPROC=1 / no-torchrun branch).
    """
    param_infos = info["param_infos"]
    for p in group["params"]:
        pinfo = param_infos[p]
        pinfo["future"].wait()
        grad_slice = pinfo["grad_slice"]
        state = self.state[p]

        if pinfo["is_small"]:
            p_slice = p
        else:
            rank_size = p.shape[0] // world_size
            p_slice = p[rank * rank_size : (rank + 1) * rank_size]

        if not state:
            state["step"] = 0
            state["exp_avg"] = _cpu_pinned_zeros(p_slice.shape, p_slice.dtype)
            state["exp_avg_sq"] = _cpu_pinned_zeros(p_slice.shape, p_slice.dtype)
        state["step"] += 1

        # H2D — async because state is pinned
        exp_avg_g = state["exp_avg"].to(p.device, non_blocking=True)
        exp_avg_sq_g = state["exp_avg_sq"].to(p.device, non_blocking=True)

        self._adamw_step_t.fill_(state["step"])
        self._adamw_lr_t.fill_(group["lr"])
        self._adamw_beta1_t.fill_(group["betas"][0])
        self._adamw_beta2_t.fill_(group["betas"][1])
        self._adamw_eps_t.fill_(group["eps"])
        self._adamw_wd_t.fill_(group["weight_decay"])

        adamw_step_fused(
            p_slice,
            grad_slice,
            exp_avg_g,
            exp_avg_sq_g,
            self._adamw_step_t,
            self._adamw_lr_t,
            self._adamw_beta1_t,
            self._adamw_beta2_t,
            self._adamw_eps_t,
            self._adamw_wd_t,
        )

        # D2H — async copy back to pinned CPU buffer
        state["exp_avg"].copy_(exp_avg_g, non_blocking=True)
        state["exp_avg_sq"].copy_(exp_avg_sq_g, non_blocking=True)

        if not pinfo["is_small"]:
            future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
            gather_list.append(dict(future=future, params=None))


def patched_compute_muon(
    self: Any,
    group: dict[str, Any],
    info: dict[str, Any],
    gather_list: list[dict[str, Any]],
    rank: int,
) -> None:
    """CPU-offloaded version of DistMuonAdamW._compute_muon.

    Muon state buffers (momentum_buffer, second_momentum_buffer) live on
    CPU pinned memory; only the OWNED slice is copied H2D per step
    (non-owned chunks don't participate on this rank). Dual-GPU only —
    single-GPU goes through patched_step_muon via train.sh's NPROC=1
    no-torchrun branch.
    """
    info["future"].wait()
    params = group["params"]
    chunk_size = info["chunk_size"]
    grad_chunk = info["grad_chunk"]
    p = params[0]
    shape, device, dtype = p.shape, p.device, p.dtype

    start_idx = rank * chunk_size
    num_owned = min(chunk_size, max(0, len(params) - start_idx))

    state = self.state[p]
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = _cpu_pinned_zeros((chunk_size, *shape), dtype)
    if "second_momentum_buffer" not in state:
        state_shape = (
            (chunk_size, shape[-2], 1)
            if shape[-2] >= shape[-1]
            else (chunk_size, 1, shape[-1])
        )
        state["second_momentum_buffer"] = _cpu_pinned_zeros(state_shape, dtype)
    red_dim = -1 if shape[-2] >= shape[-1] else -2

    updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

    if num_owned > 0:
        # H2D the OWNED slice only — saves PCIe bandwidth for non-owned chunks
        mom_g = state["momentum_buffer"][:num_owned].to(device, non_blocking=True)
        mom2_g = state["second_momentum_buffer"][:num_owned].to(
            device, non_blocking=True
        )

        owned_params = [params[start_idx + i] for i in range(num_owned)]
        stacked_owned = torch.stack(owned_params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_fused(
            grad_chunk[:num_owned],
            stacked_owned,
            mom_g,
            mom2_g,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        updated_params[:num_owned].copy_(stacked_owned)

        # D2H — async copy back
        state["momentum_buffer"][:num_owned].copy_(mom_g, non_blocking=True)
        state["second_momentum_buffer"][:num_owned].copy_(mom2_g, non_blocking=True)

    if num_owned < chunk_size:
        updated_params[num_owned:].zero_()

    stacked_params = info["stacked_grads"]
    future = dist.all_gather_into_tensor(
        stacked_params, updated_params, async_op=True
    ).get_future()
    gather_list.append(
        dict(future=future, stacked_params=stacked_params, params=params)
    )


# -----------------------------------------------------------------------------
# Single-GPU (MuonAdamW) versions: full-size state on CPU pinned memory.
# Same H2D/D2H pattern but without the reduce_scatter / all_gather glue.


def patched_step_adamw(self: Any, group: dict[str, Any]) -> None:
    """CPU-offloaded version of MuonAdamW._step_adamw (single-GPU path)."""
    for p in group["params"]:
        if p.grad is None:
            continue
        grad = p.grad
        state = self.state[p]

        if not state:
            state["step"] = 0
            state["exp_avg"] = _cpu_pinned_zeros(p.shape, p.dtype)
            state["exp_avg_sq"] = _cpu_pinned_zeros(p.shape, p.dtype)
        state["step"] += 1

        exp_avg_g = state["exp_avg"].to(p.device, non_blocking=True)
        exp_avg_sq_g = state["exp_avg_sq"].to(p.device, non_blocking=True)

        self._adamw_step_t.fill_(state["step"])
        self._adamw_lr_t.fill_(group["lr"])
        self._adamw_beta1_t.fill_(group["betas"][0])
        self._adamw_beta2_t.fill_(group["betas"][1])
        self._adamw_eps_t.fill_(group["eps"])
        self._adamw_wd_t.fill_(group["weight_decay"])

        adamw_step_fused(
            p,
            grad,
            exp_avg_g,
            exp_avg_sq_g,
            self._adamw_step_t,
            self._adamw_lr_t,
            self._adamw_beta1_t,
            self._adamw_beta2_t,
            self._adamw_eps_t,
            self._adamw_wd_t,
        )

        state["exp_avg"].copy_(exp_avg_g, non_blocking=True)
        state["exp_avg_sq"].copy_(exp_avg_sq_g, non_blocking=True)


def patched_step_muon(self: Any, group: dict[str, Any]) -> None:
    """CPU-offloaded version of MuonAdamW._step_muon (single-GPU path)."""
    params = group["params"]
    if not params:
        return

    p = params[0]
    state = self.state[p]
    num_params = len(params)
    shape, device, dtype = p.shape, p.device, p.dtype

    if "momentum_buffer" not in state:
        state["momentum_buffer"] = _cpu_pinned_zeros((num_params, *shape), dtype)
    if "second_momentum_buffer" not in state:
        state_shape = (
            (num_params, shape[-2], 1)
            if shape[-2] >= shape[-1]
            else (num_params, 1, shape[-1])
        )
        state["second_momentum_buffer"] = _cpu_pinned_zeros(state_shape, dtype)
    red_dim = -1 if shape[-2] >= shape[-1] else -2

    # H2D copy the full state to GPU buffers (single-GPU has no ZeRO
    # sharding, so unlike the dist version we copy momentum_buffer in full).
    mom_g = state["momentum_buffer"].to(device, non_blocking=True)
    mom2_g = state["second_momentum_buffer"].to(device, non_blocking=True)

    stacked_grads = torch.stack([p.grad for p in params])
    stacked_params = torch.stack(params)

    self._muon_momentum_t.fill_(group["momentum"])
    self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
    self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
    self._muon_wd_t.fill_(group["weight_decay"])

    muon_step_fused(
        stacked_grads,
        stacked_params,
        mom_g,
        mom2_g,
        self._muon_momentum_t,
        self._muon_lr_t,
        self._muon_wd_t,
        self._muon_beta2_t,
        group["ns_steps"],
        red_dim,
    )

    # D2H copy state back
    state["momentum_buffer"].copy_(mom_g, non_blocking=True)
    state["second_momentum_buffer"].copy_(mom2_g, non_blocking=True)

    # Copy updated params back to originals (same as original implementation)
    torch._foreach_copy_(params, list(stacked_params.unbind(0)))
