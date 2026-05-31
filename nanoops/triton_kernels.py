"""Triton kernels for nanoops (Tier 3 — opt-in CUDA kernel rewrites).

Each kernel here mirrors the math of the corresponding Python op in
`functional.py` but fuses multiple passes into a single GPU kernel.
Wins come from:
  - fewer kernel-launch overheads (~30 us each)
  - fewer round-trips to HBM (a chain like `norm → linear → relu²` is
    3× HBM traffic; fused is 1× input read + 1× output write)
  - smaller working set (no intermediate buffers between ops)

The training integration activates selected kernels via env var per op
(e.g. `NANOOPS_FUSED_MLP_BLOCK=1` for the production-wired
`fused_mlp_block` Triton op — see `nanoops/integration.py` for the flag
list). That integration layer falls back to eager Python when a kernel
is unavailable or disabled; this module itself is only a direct re-export shim.

The actual code lives in three feature-split sibling modules:
  - `triton_fused_add_norm.py` — FusedAddNorm + shared TileConfig helper
  - `triton_fused_mlp_block.py` — FusedMLPBlock (reuses Step 0 kernel
    from triton_fused_add_norm)
  - `triton_fused_attn_qkv.py` / `triton_fused_attn_sdpa.py` /
    `triton_fused_attn_output.py` — attention-side kernels

This module is a thin re-export shim so existing
`from nanoops.triton_kernels import …` callers keep working unchanged.

Parity tests live in tests/test_triton_*.py.
"""

from __future__ import annotations

# ── FusedAddNorm + shared utilities ─────────────────────────────────
from .triton_fused_add_norm import (
    TileConfig,
    _fused_add_norm_bwd_impl,
    _fused_add_norm_bwd_op,
    _fused_add_norm_fwd_impl,
    _fused_add_norm_fwd_op,
    _pick_tile_config,
    fused_add_norm,
)

# ── FusedMLPBlock ───────────────────────────────────────────────────
from .triton_fused_mlp_block import (
    _fused_mlp_block_bwd_impl,
    _fused_mlp_block_bwd_op,
    _fused_mlp_block_fwd_impl,
    _fused_mlp_block_fwd_op,
    fused_mlp_block,
)

# ── Attention-side kernels ──────────────────────────────────────────
from .triton_fused_attn_output import (
    output_proj_residual,
)
from .triton_fused_attn_qkv import (
    _norm_qkv_projection_bwd_impl,
    _norm_qkv_projection_bwd_op,
    _norm_qkv_projection_residual_mix_fwd_impl,
    _norm_qkv_projection_residual_mix_fwd_op,
    norm_qkv_projection_with_residual_mix,
)
from .triton_fused_attn_sdpa import (
    _flash_sdpa_bwd_impl,
    _flash_sdpa_bwd_op,
    _flash_sdpa_fwd_impl,
    _flash_sdpa_fwd_op,
    flash_sdpa,
)

# Private @triton.jit kernels — re-exported for direct benchmark scripts
# (/tmp/sweep_*.py, /tmp/bench_*.py) that import them by name. Guarded
# because they only exist when triton is installed; on a no-triton env
# the names just aren't bound — same behavior as the pre-split file.
try:
    from .triton_fused_add_norm import (
        _fused_add_norm_bwd_inline_kernel,
        _fused_add_norm_bwd_kernel,
        _fused_add_norm_fwd_kernel,
        _fused_add_norm_inner_kernel,
    )
    from .triton_fused_mlp_block import (
        _cast_matmul_kernel,
        _mlp_dW_fc_bwd_kernel,
        _mlp_dW_proj_bwd_kernel,
        _mlp_dx_bwd_kernel,
        _mlp_dz_bwd_kernel,
        _relu_sq_linear_residual_fwd_kernel,
    )
except ImportError:
    pass
