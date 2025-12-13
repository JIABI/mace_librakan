from __future__ import annotations

import math
import torch
from torch import nn

from .shrinkage import GeneralizedShrinkage  # adjust import based on your directory


class PhysicsSpectralKernel(nn.Module):
    """
    High-frequency physics-guided radial kernel for edge distances.

    Given a radial distance r in (0, cutoff], this module computes a set of
    Fourier-like spectral responses exp(i * omega_k * r / cutoff), followed by
    generalized shrinkage S_{lambda,p} to enforce sparsity in the high-frequency
    components. This behaves as a lightweight NUFFT-style spectral encoder,
    suitable for edge-level radial modeling in message-passing networks.

    Args:
        num_freq: number of spectral frequencies.
        cutoff: radial cutoff, used for normalization.
        omega_max: maximum angular frequency for initialization.
        shrinkage_lambda: shrinkage intensity (λ).
        shrinkage_p: exponent p ∈ (0,1], controlling shrinkage smoothness.
        trainable_lambda: whether λ is learnable.
        learnable_omega: whether the frequency grid is learnable.
        internal_dtype: internal compute dtype (usually float32).
    """

    def __init__(
            self,
            num_freq: int,
            cutoff: float,
            omega_max: float = math.pi,
            *,
            shrinkage_lambda: float = 1e-2,
            shrinkage_p: float = 0.7,
            trainable_lambda: bool = False,
            learnable_omega: bool = True,
            internal_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if num_freq <= 0:
            raise ValueError("num_freq must be positive.")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive.")

        self.num_freq = int(num_freq)
        self.cutoff = float(cutoff)
        self.omega_max = float(omega_max)
        self.internal_dtype = internal_dtype

        # Initialize a linearly spaced set of frequencies in [0, omega_max]
        omega = torch.linspace(
            0.0,
            self.omega_max,
            steps=self.num_freq,
            dtype=self.internal_dtype,
        )  # shape: [num_freq]

        if learnable_omega:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)

        # Generalized shrinkage to sparsify high-frequency activations
        self.shrink = GeneralizedShrinkage(
            lam=shrinkage_lambda,
            p=shrinkage_p,
            trainable_lambda=trainable_lambda,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral features for a batch of radial distances.

        Args:
            r: [..., 1] or [...] distance tensor.

        Returns:
            Real-valued spectral features [..., num_freq].
        """
        if r.dim() == 0:
            r = r.view(1, 1)
        if r.size(-1) == 1:
            r = r[..., 0]

        r = r.to(self.internal_dtype)
        t = r / max(self.cutoff, 1e-6)  # normalized distance in (0,1]

        omega = self.omega[None, :].to(self.internal_dtype)  # [1, num_freq]

        # Complex Fourier-like basis: exp(i ω_k t)
        phase = torch.exp(1j * (t.unsqueeze(-1) * omega))  # [..., num_freq] complex

        # Apply shrinkage in the spectral domain
        phase_shrunk = self.shrink(phase)  # complex-valued

        # Return real part (imag part optional)
        features = torch.real(phase_shrunk)

        return features