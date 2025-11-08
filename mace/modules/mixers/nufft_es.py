# nufft_es.py
# --------------------------------------------------------------------------------------
# ADDED: Differentiable NUFFT-ES (forward & adjoint) operator for LibraKAN spectral branch
# This implementation provides a robust, dependency-light Type-2 NUFFT (uniform -> NU freqs)
# with an exact adjoint. It uses a vectorized O(NK) fallback to guarantee correctness
# and PyTorch autograd compatibility on any platform.
# --------------------------------------------------------------------------------------

from __future__ import annotations
import math
import torch
from torch import nn
from typing import Optional, Tuple

def _check_complex(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        return x.to(torch.complex64) if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else x.to(torch.complex128)
    return x

class NUFFTES(nn.Module):
    """
    Type-2 NUFFT (uniform samples -> non-uniform frequency samples) with exact adjoint.

    y_k = sum_{n=0}^{N-1} x_n * exp(-i * omega_k * n)         (forward)
    adjoint maps grad_y back to x by:
    (A^H g)_n = sum_{k=0}^{K-1} g_k * exp(+i * omega_k * n)

    Args:
        max_len: optional cap for N to avoid unintended huge allocations
    """
    def __init__(self, max_len: Optional[int] = None):
        super().__init__()
        self.max_len = max_len

    @torch.no_grad()
    def _validate(self, x: torch.Tensor, omega: torch.Tensor) -> Tuple[int, int]:
        if x.dim() < 1:
            raise ValueError("x must be at least 1D [N] or [..., N].")
        if omega.dim() == 1:
            pass
        elif omega.dim() == 2:
            # allow batched omega with shared last dim
            if omega.shape[0] != x.shape[0]:
                raise ValueError("If omega is 2D, its leading dim must match x batch dim.")
        else:
            raise ValueError("omega must be 1D [K] or 2D [B,K].")
        N = x.shape[-1]
        if self.max_len is not None and N > self.max_len:
            raise ValueError(f"N={N} exceeds configured max_len={self.max_len}")
        return N, omega.shape[-1]

    def forward(self, x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Forward NUFFT-ES (Type-2, exact O(NK) evaluation).

        Shapes:
            x: [..., N] (real or complex)
            omega: [K] or [B, K], in radians (expected in [-pi, pi], not strictly required)
        Returns:
            y: [..., K] complex tensor
        """
        x = _check_complex(x)
        N, K = self._validate(x, omega)
        batch_shape = x.shape[:-1]
        device = x.device
        dtype = x.dtype
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        # Build n index vector [N]
        n = torch.arange(N, device=device, dtype=real_dtype)
        # Prepare omega to broadcast against batch
        if omega.dim() == 1:
            om = omega[None, :].to(n.dtype)  # [1,K]
            # Compute exp(-i * omega * n)
            # shape: [N,K] -> broadcast to batch
            phase = torch.exp(-1j * (n[:, None] * om[0][None, :]))  # [N,K]
            # y = sum_n x_n * phase
            x_flat = x.reshape(-1, N)  # [B*, N]
            y_list = []
            for xb in x_flat:  # loop over batch to limit mem
                yb = torch.matmul(xb.to(dtype), phase)  # [K]
                y_list.append(yb)
            y = torch.stack(y_list, dim=0).reshape(*batch_shape, K)
            return y
        else:
            if tuple(batch_shape) != (omega.shape[0],):
                # broadcast omega to batch or restrict x to leading batch
                if omega.shape[0] == 1:
                    om = omega.expand(batch_shape[0], -1)  # [B,K]
                else:
                    raise ValueError("When omega is [B,K], x must have leading batch B.")
            else:
                om = omega
            # compute per-batch phases
            x_flat = x  # [B,N]
            if x_flat.dim()==1:
                x_flat = x_flat[None, :]
            phases = torch.exp(-1j * (n[None, :, None] * om[:, None, :].to(n.dtype)))  # [B,N,K]
            y = torch.einsum("bn,bnk->bk", x_flat.to(dtype), phases)  # [B,K]
            return y.reshape(*batch_shape, K)

    def adjoint(self, gy: torch.Tensor, omega: torch.Tensor, N: int) -> torch.Tensor:
        """
        Adjoint of forward operator. Maps gradient on y back to x-length N.

        Args:
            gy: [..., K] complex gradient
            omega: [K] or [B,K] radians
            N: output signal length
        Returns:
            gx: [..., N] complex tensor
        """
        gy = _check_complex(gy)
        device = gy.device
        dtype = gy.dtype
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        n = torch.arange(N, device=device, dtype=real_dtype)  # [N]
        K = gy.shape[-1]
        batch_shape = gy.shape[:-1]

        if omega.dim() == 1:
            om = omega[None, :].to(n.dtype)  # [1,K]
            phase = torch.exp(+1j * (n[:, None] * om[0][None, :]))  # [N,K]
            gy_flat = gy.reshape(-1, K)  # [B*,K]
            gx_list = []
            for g in gy_flat:
                gxb = torch.matmul(phase, g.to(dtype))  # [N]
                gx_list.append(gxb)
            gx = torch.stack(gx_list, dim=0).reshape(*batch_shape, N)
            return gx
        else:
            if tuple(batch_shape) != (omega.shape[0],):
                if omega.shape[0] == 1:
                    om = omega.expand(batch_shape[0], -1)
                else:
                    raise ValueError("When omega is [B,K], gy must have leading batch B.")
            else:
                om = omega
            phases = torch.exp(+1j * (n[None, :, None] * om[:, None, :].to(n.dtype)))  # [B,N,K]
            gx = torch.einsum("bnk,bk->bn", phases, gy.to(dtype))  # [B,N]
            return gx.reshape(*batch_shape, N)
