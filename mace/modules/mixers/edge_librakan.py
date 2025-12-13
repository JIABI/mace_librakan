# edge_librakan.py
# --------------------------------------------------------------------------------------
# Edge-scale LibraKAN with geometry-conditioned Ω and non-uniform sampling for NUFFT–ES.
# Works on SCALAR edge features only: [..., E, C_in] -> [..., E, C_out].
# Ω = Ω(r_ij, Z_i, Z_j, ρ_i, angle, CNA), samples = g(r_ij, Z_i, Z_j, ρ_i, angle, CNA).
# Includes generalized shrinkage S_{λ,p} and optional pre-LayerNorm for stability.
# --------------------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Dict
import torch
from torch import nn

from .nufft_es import NUFFTES                   # differentiable NUFFT–ES (forward & adjoint)
from .shrinkage import GeneralizedShrinkage     # S_{lambda,p} shrinkage

# ---------------------------
# Utilities
# ---------------------------
def _act(name: str):
    return dict(relu=nn.ReLU, gelu=nn.GELU, silu=nn.SiLU).get(name, nn.GELU)

def _maybe_layernorm(x: torch.Tensor, ln: Optional[nn.LayerNorm]) -> torch.Tensor:
    return ln(x) if ln is not None else x

# ---------------------------
# Edge LibraKAN
# ---------------------------
class EdgeLibraKAN(nn.Module):
    """
    Edge-scale LibraKAN mixer (scalar-only).

    Inputs
    ------
    x_edge: [B, E, C_in]     Scalar edge features (post message passing).
    geom:   dict with keys:
        r_ij:   [B, E, 1]        Inter-atomic distance.
        z_i:    [B, E, Zd]       Source atom embedding.
        z_j:    [B, E, Zd]       Target atom embedding.
        rho_i:  [B, E, 1]        Local density (smoothed coordination) of i.
        angle:  [B, E, Ad] | None   Optional angle/three-body embedding.
        cna:    [B, E] (int) | [B, E, Dc] (float) | None   CNA descriptor.

    Output
    ------
    y: [B, E, C_out]         Residual edge update (scalar channel).

    Notes
    -----
    - Only scalar branch is touched; equivariant tensor products remain unchanged.
    - Ω and non-uniform samples are conditioned on geometry/topology (incl. CNA).
    - NUFFT–ES forward+adjoint forms the spectral feature; S_{λ,p} enforces sparsity.
    - During training, self._l1_penalty is a scalar tensor (same device) for loss accumulation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        # frequency dictionary & sampling
        freq_dim: int = 192,            # |Ω|
        sample_dim: int = 96,           # # non-uniform samples per edge
        cond_dim: int = 96,             # hidden size for conditioning MLPs
        spectral_scale: float = 1.0,    # global scale on Ω
        es_beta: float = 6.0,           # ES window parameter for NUFFT–ES
        # shrinkage
        p_shrink: float = 0.5,          # p in (0,1]; p=1 => soft-threshold
        lambda_init: float = 0.02,
        lambda_trainable: bool = True,
        l1_alpha: float = 3e-4,
        # local branch & fusion
        base_activation: str = "gelu",
        dropout: float = 0.0,
        local_hidden: Optional[int] = None,  # None => in_channels
        use_alpha_fuse: bool = True,
        alpha_min: float = 0.0,         # clamp alpha in [alpha_min, 1]
        alpha_tau: float = 1.0,         # temperature for alpha logistic
        # CNA support
        use_cna: bool = True,
        cna_num_classes: int = 16,      # if discrete CNA labels (e.g., HA codes bucketed)
        cna_embed_dim: int = 16,
        cna_proj_dim: int = 16,         # if continuous CNA vector, project to this
        # stability
        pre_layernorm: bool = True,     # LN on input edge scalars inside this module
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freq_dim = freq_dim
        self.sample_dim = sample_dim
        self.cond_dim = cond_dim
        self.spectral_scale = spectral_scale
        self.es_beta = es_beta
        self.l1_alpha = l1_alpha

        self.use_alpha_fuse = use_alpha_fuse
        self.alpha_min = alpha_min
        self.alpha_tau = alpha_tau

        self.use_cna = use_cna
        self.cna_num_classes = cna_num_classes
        self.cna_embed_dim = cna_embed_dim
        self.cna_proj_dim = cna_proj_dim

        Act = _act(base_activation)

        # --- Optional LayerNorm on input scalar edges ---
        self.pre_ln = nn.LayerNorm(in_channels) if pre_layernorm else None

        # --- Local (spatial) branch: lightweight MLP over edge scalars ---
        H = local_hidden or in_channels
        local_layers = [nn.Linear(in_channels, H, bias=False), Act()]
        if dropout > 0:
            local_layers.append(nn.Dropout(dropout))
        self.local = nn.Sequential(*local_layers)

        # --- Conditioning reducer (lazy rebuild to match runtime cond_raw dim) ---
        # Placeholder; rebuilt at first forward pass when cond_raw dim is known.
        self.cond_reduce = nn.Sequential(nn.Linear(2, cond_dim), nn.SiLU())
        self._cond_built = False

        # --- Ω(u,v) and per-frequency gate ---
        self.omega_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), Act(),
            nn.Linear(cond_dim, 2 * freq_dim),   # (u,v) per frequency
        )
        self.omega_gate = nn.Linear(cond_dim, freq_dim)  # logits -> sigmoid in forward

        # --- Non-uniform sample locations (sx, sy) ---
        self.sample_proj = nn.Linear(cond_dim, 2 * sample_dim)

        # --- Map local features -> spectral coefficients (real; imag=0) ---
        self.coeff_map = nn.Linear(H, freq_dim, bias=False)

        # --- NUFFT–ES operator ---
        self.nufft = NUFFTES(beta=es_beta)

        # --- Generalized shrinkage S_{λ,p} ---
        lam = torch.full([1, 1, freq_dim], float(lambda_init))
        self.shrink = GeneralizedShrinkage(p=p_shrink, lam=lam, trainable=lambda_trainable)

        # --- Output head ---
        out_layers = []
        if dropout > 0:
            out_layers.append(nn.Dropout(dropout))
        out_layers.append(nn.Linear(freq_dim, out_channels, bias=False))
        self.out_head = nn.Sequential(*out_layers)

        # --- Optional local-spectral fusion weight alpha ---
        if use_alpha_fuse:
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        # CNA submodules are created lazily based on dtype/shape
        self.cna_emb: Optional[nn.Embedding] = None
        self.cna_lin: Optional[nn.Sequential] = None

        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
        self._l1_penalty = self._zero

    # ---------------------------
    # Helpers
    # ---------------------------
    def _build_cond_reduce(self, cond_raw_dim: int):
        self.cond_reduce = nn.Sequential(nn.Linear(cond_raw_dim, self.cond_dim), nn.SiLU()).to(next(self.parameters()).device)
        self._cond_built = True

    def _cna_features(self, cna: torch.Tensor) -> torch.Tensor:
        """
        Turn CNA input into a feature vector at edge level.
        Accepts either integer labels [B,E] or continuous vectors [B,E,Dc].
        """
        if cna is None:
            return None
        if cna.dtype in (torch.int32, torch.int64):
            if self.cna_emb is None:
                self.cna_emb = nn.Embedding(self.cna_num_classes, self.cna_embed_dim).to(cna.device)
            return self.cna_emb(cna)  # [B,E,cna_embed_dim]
        else:
            if self.cna_lin is None:
                in_dim = cna.size(-1)
                self.cna_lin = nn.Sequential(nn.Linear(in_dim, self.cna_proj_dim), nn.SiLU()).to(cna.device)
            return self.cna_lin(cna)  # [B,E,cna_proj_dim]

    def _make_condition(self, geom: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [geom["r_ij"], geom["z_i"], geom["z_j"], geom["rho_i"]]
        if geom.get("angle", None) is not None:
            parts.append(geom["angle"])
        if self.use_cna and ("cna" in geom) and (geom["cna"] is not None):
            parts.append(self._cna_features(geom["cna"]))
        cond_raw = torch.cat(parts, dim=-1)  # [B,E,cond_raw]
        if (not self._cond_built) or (cond_raw.size(-1) != self.cond_reduce[0].in_features):
            self._build_cond_reduce(cond_raw.size(-1))
        return self.cond_reduce(cond_raw)     # [B,E,cond_dim]

    @staticmethod
    def _shrink_complex(z: torch.Tensor, shrink: GeneralizedShrinkage) -> torch.Tensor:
        """
        Apply shrinkage to complex tensor by shrinking real and imag parts separately.
        Assumes `shrink(x)` supports real tensors.
        """
        if torch.is_complex(z):
            zr = shrink(z.real)
            zi = shrink(z.imag)
            return torch.complex(zr, zi)
        return shrink(z)

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, x_edge: torch.Tensor, geom: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x_edge: [B,E,C_in]
        geom:   see class docstring
        """
        # 1) Optional pre-LayerNorm for stability
        x = _maybe_layernorm(x_edge, self.pre_ln)

        # 2) Local branch
        local = self.local(x)                                    # [B,E,H]

        # 3) Build geometry/topology-conditioned vector
        cond = self._make_condition(geom)                        # [B,E,cond_dim]

        # 4) Frequency dictionary Ω(u,v) and gate
        omega_uv = self.omega_mlp(cond).view(
            x.shape[0], x.shape[1], self.freq_dim, 2
        )                                                        # [B,E,F,2]
        omega_uv = self.spectral_scale * omega_uv
        gate = torch.sigmoid(self.omega_gate(cond))              # [B,E,F]

        # 5) Non-uniform samples (sx, sy)
        samp = self.sample_proj(cond).view(
            x.shape[0], x.shape[1], self.sample_dim, 2
        )                                                        # [B,E,S,2]

        # 6) Local -> spectral coefficients (complex with imag=0)
        coeff_real = self.coeff_map(local)                       # [B,E,F]
        coeff = torch.complex(coeff_real, torch.zeros_like(coeff_real))

        # 7) NUFFT–ES forward & adjoint: samples -> stabilized spectral feature
        y_samp = self.nufft.forward(coeff, omega_uv, samp)       # [B,E,S] complex
        z_spec = self.nufft.adjoint(y_samp, omega_uv, self.freq_dim)  # [B,E,F] complex

        # 8) Gate + generalized shrinkage S_{λ,p}
        z_gated = z_spec * gate                                  # broadcast on last dim
        z_shrunk = self._shrink_complex(z_gated, self.shrink)    # complex-safe
        z_real = z_shrunk.real                                   # [B,E,F]

        # 9) Sparse regularization term (exposed to trainer)
        if self.training and (self.l1_alpha > 0):
            self._l1_penalty = self.l1_alpha * z_real.abs().mean()
        else:
            self._l1_penalty = self._zero.to(x.device)

        # 10) Optional local-spectral fusion
        if self.use_alpha_fuse:
            tau = self.alpha_tau if self.alpha_tau > 0 else 1.0
            alpha = torch.sigmoid(self.alpha_logit / tau)
            if self.alpha_min > 0:
                alpha = torch.clamp(alpha, min=self.alpha_min)
            local_F = self.coeff_map(local)                      # [B,E,F]
            spec_feat = (1.0 - alpha) * local_F + alpha * z_real # [B,E,F]
        else:
            spec_feat = z_real

        # 11) Project back to out_channels
        out = self.out_head(spec_feat)                           # [B,E,C_out]
        return out
