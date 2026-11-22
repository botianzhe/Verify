"""Discrete cosine transforms compatible with the trained watermark network.

To keep runners self-contained, we provide a small implementation of 1D DCT-II
and its inverse (IDCT-III) operating over the last dimension.

This is sufficient for the repo's usage patterns (feature transforms over a
fixed channel dimension) and avoids adding a hard dependency.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def dct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """DCT-II over the last dimension."""
    if x.numel() == 0:
        return x

    x_shape = x.shape
    n = x_shape[-1]
    x = x.contiguous().view(-1, n)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    vc = torch.fft.fft(v, dim=1)

    k = -torch.arange(n, device=x.device, dtype=x.dtype) * (math.pi / (2.0 * n))
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v = vc[:, :n]
    out = (v.real * w_r - v.imag * w_i) * 2.0

    if norm == "ortho":
        out[:, 0] /= math.sqrt(n) * 2.0
        if n > 1:
            out[:, 1:] /= math.sqrt(n / 2.0) * 2.0

    return out.view(*x_shape)


def idct(x: torch.Tensor, norm: Optional[str] = None) -> torch.Tensor:
    """Inverse of DCT-II (i.e., IDCT-III) over the last dimension."""
    if x.numel() == 0:
        return x

    x_shape = x.shape
    n = x_shape[-1]
    x = x.contiguous().view(-1, n)

    if norm == "ortho":
        x = x.clone()
        x[:, 0] *= math.sqrt(n) * 2.0
        if n > 1:
            x[:, 1:] *= math.sqrt(n / 2.0) * 2.0

    x = x / 2.0

    k = torch.arange(n, device=x.device, dtype=x.dtype) * (math.pi / (2.0 * n))
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v_t_r = x
    v_t_i = torch.zeros_like(v_t_r)

    v_r = v_t_r * w_r - v_t_i * w_i
    v_i = v_t_r * w_i + v_t_i * w_r

    v = torch.complex(v_r, v_i)

    # Build the full spectrum for the IFFT.
    v_full = torch.zeros((v.shape[0], 2 * n), dtype=v.dtype, device=v.device)
    v_full[:, :n] = v
    v_full[:, n + 1 :] = torch.conj(v[:, 1:]).flip([1])

    v_time = torch.fft.ifft(v_full, dim=1).real

    out = torch.zeros_like(x)
    out[:, ::2] = v_time[:, : (n + 1) // 2]
    out[:, 1::2] = v_time[:, (n + 1) // 2 :].flip([1])

    return out.view(*x_shape)
