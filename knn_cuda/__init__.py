"""Compatibility wrapper for the installed knn_cuda extension.

The upstream package recompiles its extension on every import through
torch.utils.cpp_extension.load, which unnecessarily requires Ninja even when
the prebuilt shared object is already present. Load that binary directly and
fall back to a PyTorch implementation when it is unavailable.
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


__version__ = "0.2-local"


def _load_prebuilt_extension():
    local_dir = Path(__file__).resolve().parent
    for entry in sys.path:
        if not entry:
            continue
        try:
            candidate = (
                Path(entry).resolve()
                / "knn_cuda"
                / "csrc"
                / "_ext"
                / "knn"
                / "knn.so"
            )
        except (OSError, RuntimeError):
            continue
        if not candidate.is_file() or local_dir in candidate.parents:
            continue
        try:
            spec = importlib.util.spec_from_file_location("knn", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except (ImportError, OSError):
            continue
    return None


_knn = _load_prebuilt_extension()
USING_PREBUILT = _knn is not None


def _transpose(tensor, enabled=False):
    if enabled:
        return tensor.transpose(0, 1).contiguous()
    return tensor


def knn(ref, query, k):
    """Return distances and zero-based neighbor indices for [D, N] tensors."""
    if _knn is not None:
        distances, indices = _knn.knn(ref, query, k)
        return distances, indices - 1

    # The extension returns [K, M] for query [D, M]. Match that layout.
    pairwise = torch.cdist(
        query.transpose(0, 1).unsqueeze(0).float(),
        ref.transpose(0, 1).unsqueeze(0).float(),
    ).squeeze(0)
    distances, indices = torch.topk(
        pairwise, k=k, dim=-1, largest=False, sorted=True
    )
    return distances.transpose(0, 1).contiguous(), indices.transpose(0, 1).contiguous()


class KNN(nn.Module):
    """Drop-in replacement for knn_cuda.KNN."""

    def __init__(self, k, transpose_mode=False):
        super().__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):
        if ref.ndim != 3 or query.ndim != 3:
            raise ValueError("ref and query must be rank-3 batched tensors")
        if ref.shape[0] != query.shape[0]:
            raise ValueError("ref and query must have the same batch size")

        batch_distances = []
        batch_indices = []
        with torch.no_grad():
            for batch_idx in range(ref.shape[0]):
                ref_item = _transpose(ref[batch_idx], self._t)
                query_item = _transpose(query[batch_idx], self._t)
                distances, indices = knn(ref_item.float(), query_item.float(), self.k)
                batch_distances.append(_transpose(distances, self._t))
                batch_indices.append(_transpose(indices, self._t))

        return torch.stack(batch_distances), torch.stack(batch_indices)
