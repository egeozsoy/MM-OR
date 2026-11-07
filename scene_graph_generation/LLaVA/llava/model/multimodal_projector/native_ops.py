"""
Native-ATen replacements for the compiled third-party ops MM-OR depends on.

`torch_scatter` needs a compiled wheel per torch/CUDA version, and its entire surface in
MM-OR is `segment_csr` at two call sites in pointtransformerv3.py, with reduce="max" and
reduce="mean".

These were validated elementwise against torch_scatter on the original stack (see
MODERNIZATION.md) and are written to stay ONNX-exportable: repeat_interleave, index_add_
and scatter_reduce_ all have ONNX representations.
"""

import torch


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """
    Drop-in replacement for torch_scatter.segment_csr for the shapes MM-OR uses.

    src    : [N, C] values, already sorted so that each segment is contiguous
    indptr : [S+1]  CSR offsets; segment s covers src[indptr[s]:indptr[s+1]]
    reduce : "sum" | "mean" | "max" | "min"
    returns: [S, C]

    Scope note: handles 1-D indptr reducing over dim 0, which is all MM-OR needs.
    torch_scatter's broadcasting of higher-rank indptr is deliberately NOT reproduced --
    a wrong-shape input raises instead of silently differing from the original.
    """
    if src.dim() != 2:
        raise ValueError(f"segment_csr replacement expects 2-D src, got {tuple(src.shape)}")
    if indptr.dim() != 1:
        raise ValueError(f"expects 1-D indptr, got {tuple(indptr.shape)}")

    num_segments = indptr.numel() - 1
    lengths = indptr[1:] - indptr[:-1]

    # Fail loudly and informatively on a malformed indptr. torch.repeat_interleave would
    # otherwise raise a bare "repeats can not be negative", which says nothing about which
    # input was wrong -- and that error, seen on a remote host, is very hard to chase.
    if bool((lengths < 0).any()):
        bad = int((lengths < 0).sum())
        first = int(torch.nonzero(lengths < 0)[0])
        raise ValueError(
            f"segment_csr: indptr must be non-decreasing, got {bad} negative segment "
            f"length(s); first at index {first} "
            f"(indptr[{first}]={int(indptr[first])} > indptr[{first+1}]={int(indptr[first+1])}). "
            f"indptr dtype={indptr.dtype}, numel={indptr.numel()}, "
            f"src shape={tuple(src.shape)}, device={src.device}."
        )
    if int(indptr[-1]) > src.shape[0]:
        raise ValueError(
            f"segment_csr: indptr[-1]={int(indptr[-1])} exceeds src rows={src.shape[0]}"
        )

    # Segment id per row, e.g. lengths [2,3] -> [0,0,1,1,1].
    seg = torch.repeat_interleave(
        torch.arange(num_segments, device=src.device), lengths
    )

    if reduce == "sum":
        out = src.new_zeros((num_segments, src.shape[1]))
        return out.index_add_(0, seg, src)

    if reduce == "mean":
        out = src.new_zeros((num_segments, src.shape[1]))
        out.index_add_(0, seg, src)
        # clamp(min=1) only guards empty segments, which cannot occur here (idx_ptr comes
        # from cumsum of unique(return_counts=True), so every count >= 1). Kept so the
        # function is total rather than UB if reused elsewhere.
        return out / lengths.clamp(min=1).unsqueeze(1).to(out.dtype)

    if reduce in ("max", "min"):
        amode = "amax" if reduce == "max" else "amin"
        out = src.new_zeros((num_segments, src.shape[1]))
        # include_self=False so empty segments stay 0, matching torch_scatter, and
        # non-empty segments are not polluted by the zero init.
        return out.scatter_reduce_(
            0, seg.unsqueeze(1).expand(-1, src.shape[1]), src, reduce=amode, include_self=False
        )

    raise ValueError(f"unsupported reduce={reduce!r}")
