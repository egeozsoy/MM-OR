"""
Backend selection for the ops MM-OR used to get from compiled third-party packages.

`torch_scatter` is used if present (so the original 2023 stack keeps bit-identical
behaviour), otherwise the native ATen implementation in native_ops.py takes over. The
native path was validated against torch_scatter: exact for reduce="max", ~1e-7 for
reduce="mean" (see MODERNIZATION.md).
"""

try:
    from torch_scatter import segment_csr  # noqa: F401
    SEGMENT_CSR_BACKEND = "torch_scatter"
except ImportError:  # pragma: no cover - depends on the environment
    from .native_ops import segment_csr  # noqa: F401
    SEGMENT_CSR_BACKEND = "native"
