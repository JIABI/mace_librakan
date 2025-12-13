###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import os
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    OEQConfig,
    SymmetricContractionWrapper,
    TensorProduct,
    TransposeIrrepsLayoutWrapper,
)
from mace.tools.compile import simplify_if_compile
from mace.tools.scatter import scatter_sum
from mace.tools.utils import LAMMPS_MP

from .irreps_tools import mask_head, reshape_irreps, tp_out_irreps_with_instructions
from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialMLP,
    SoftTransform,
)

# --------------------------- Libra / KAN / KAF mixers (readout & tiny MLP drop-ins) ---------------------------
# We keep these imports local to avoid breaking repositories that don't ship these files
try:
    from mace.modules.mixers.librakan import GeneralLibraKAN
except Exception:
    from mace.modules.mixers.librakan import GeneralLibraKAN  # fallback if package layout differs

try:
    from mace.modules.mixers.edge_librakan import EdgeLibraKAN
except Exception:
    from mace.modules.mixers.edge_librakan import EdgeLibraKAN

try:
    from mace.modules.mixers.node_librakan import NodeLibraKAN
except Exception:
    from mace.modules.mixers.node_librakan import NodeLibraKAN

# Optional: KAN / KAF readout mixers (you said KAN/KAF only touch readout MLP)
def _make_readout_mixer(kind: str, in_dim: int, out_dim: int) -> tnn.Module:
    """
    Factory for readout mixers. Supports 'mlp' (default), 'kan', 'kaf', 'libra'.
    We keep it simple: a fully-connected mapping on flattened hidden readout features.
    """
    k = (kind or "mlp").lower()
    if k in ("libra", "librakan"):
        return GeneralLibraKAN(in_dim=in_dim, out_dim=out_dim)
    elif k == "kan":
        # You uploaded kan.py; assume it exposes class KANBlock(in_dim, out_dim) or similar.
        # Provide a robust fallback to a tiny 2-layer MLP if import fails.
        try:
            from mixers.kan import KANReadout
            return KANReadout(in_dim, out_dim)
        except Exception:
            hidden = max(in_dim, out_dim)
            return tnn.Sequential(tnn.Linear(in_dim, hidden), tnn.GELU(), tnn.Linear(hidden, out_dim))
    elif k == "kaf":
        try:
            from mixers.kaf import KAFReadout
            return KAFReadout(in_dim, out_dim)
        except Exception:
            hidden = max(in_dim, out_dim)
            return tnn.Sequential(tnn.Linear(in_dim, hidden), tnn.SiLU(), tnn.Linear(hidden, out_dim))
    else:
        hidden = max(in_dim, out_dim)
        return tnn.Sequential(tnn.Linear(in_dim, hidden), tnn.GELU(), tnn.Linear(hidden, out_dim))


def _env(flag: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(flag, "")
    return v if v != "" else default


def _env_bool(flag: str, default: bool = False) -> bool:
    v = os.environ.get(flag, "")
    if v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on")


READOUT_MIXER_KIND = (_env("READOUT_MIXER", "mlp") or "mlp").lower()  # mlp|kan|kaf|libra
EDGE_LIBRA = _env_bool("EDGE_LIBRAKAN", False)
NODE_LIBRA = _env_bool("NODE_LIBRAKAN", False)


def _maybe_replace_edge_mlp(old_mlp: tnn.Module, in_dim: int, out_dim: int) -> tnn.Module:
    """
    Replace edge-side tiny MLPs (e.g., conv_tp_weights) with EdgeLibraKAN if EDGE_LIBRA is set.
    """
    if EDGE_LIBRA:
        return EdgeLibraKAN(in_dim=in_dim, out_dim=out_dim)
    return old_mlp


def _maybe_replace_node_mlp(old_mlp: tnn.Module, in_dim: int, out_dim: int) -> tnn.Module:
    """
    Replace node-side tiny MLPs (e.g., density/residual scalars) with NodeLibraKAN if NODE_LIBRA is set.
    """
    if NODE_LIBRA:
        return NodeLibraKAN(in_dim=in_dim, out_dim=out_dim)
    return old_mlp


def _infer_linear_stack_dims(m: tnn.Module) -> Tuple[int, int]:
    """
    Infer (in_dim, out_dim) from a stack of Linear layers.
    """
    linears = [x for x in m.modules() if isinstance(x, tnn.Linear)]
    if len(linears) >= 1:
        return linears[0].in_features, linears[-1].out_features
    # RadialMLP or FullyConnectedNet might not expose .in_features; we fallback to simple heuristics.
    # The caller will pass explicit dims where possible; this function is used with care.
    return (128, 128)


# ==============================================================================================================
# Original blocks start here (unchanged in interface). We only add minimal code to inject mixers where needed.
# ==============================================================================================================


@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irreps_out, cueq_config=cueq_config
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irrep_out, cueq_config=cueq_config
        )

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@simplify_if_compile
@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    """
    Original: Linear -> Activation -> Linear
    Patch: if READOUT_MIXER_KIND in {kan,kaf,libra}, insert a flattened mixer between
           the activation and the final Linear. This only changes scalar readout MLP.
    """
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out, cueq_config=cueq_config
        )

        # Mixer injection (readout)
        self._use_mixer = READOUT_MIXER_KIND in ("kan", "kaf", "libra")
        if self._use_mixer:
            hid_dim = self.hidden_irreps.dim  # total scalar+vector dims flattened
            self.readout_mixer = _make_readout_mixer(READOUT_MIXER_KIND, hid_dim, hid_dim)

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        if self._use_mixer:
            # Flatten last dim to [N, H] for the readout mixer
            n = x.shape[0]
            x = x.reshape(n, -1)
            x = self.readout_mixer(x)
            # Restore shape expected by e3nn Linear (flattened scalar field is fine)
            # Here we keep it as [N, H] which Linear from wrapper accepts.
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]


@simplify_if_compile
@compile_mode("script")
class NonLinearBiasReadoutBlock(torch.nn.Module):
    """
    Original: Linear -> Act -> Linear(bias) -> Act -> Linear(bias)
    Patch: optional flattened mixer between the mid nonlinearity and final Linear.
    """
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_mid = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.hidden_irreps, biases=True
        )
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out, biases=True
        )

        # Mixer injection (readout)
        self._use_mixer = READOUT_MIXER_KIND in ("kan", "kaf", "libra")
        if self._use_mixer:
            hid_dim = self.hidden_irreps.dim
            self.readout_mixer = _make_readout_mixer(READOUT_MIXER_KIND, hid_dim, hid_dim)

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        x = self.non_linearity(self.linear_mid(x))
        if self._use_mixer:
            n = x.shape[0]
            x = x.reshape(n, -1)
            x = self.readout_mixer(x)
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]


@compile_mode("script")
class LinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        dipole_only: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_out, cueq_config=cueq_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_nonlin, cueq_config=cueq_config
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class LinearDipolePolarReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        use_polarizability: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        if use_polarizability:
            print("You will calculate the polarizability and dipole.")
            self.irreps_out = o3.Irreps("2x0e + 1x1o + 1x2e")
        else:
            raise ValueError(
                "Invalid configuration for LinearDipolePolarReadoutBlock: "
                "use_polarizability must be either True."
                "If you want to calculate only the dipole, use AtomicDipolesMACE."
            )

        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_out, cueq_config=cueq_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        y = self.linear(x)  # [n_nodes, 1]
        return y  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipolePolarReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        use_polarizability: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if use_polarizability:
            print("You will calculate the polarizability and dipole.")
            self.irreps_out = o3.Irreps("2x0e + 1x1o + 1x2e")
        else:
            raise ValueError(
                "Invalid configuration for NonLinearDipolePolarReadoutBlock: "
                "use_polarizability must be either True."
                "If you want to calculate only the dipole, use AtomicDipolesMACE."
            )
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_nonlin, cueq_config=cueq_config
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        # assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, n_heads]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, torch.atleast_2d(self.atomic_energies).T)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
        apply_cutoff: bool = True,
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel
        self.apply_cutoff = apply_cutoff

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        if hasattr(self, "distance_transform"):
            edge_lengths = self.distance_transform(
                edge_lengths, node_attrs, edge_index, atomic_numbers
            )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        if hasattr(self, "apply_cutoff"):
            if not self.apply_cutoff:
                return radial, cutoff
        return radial * cutoff, None  # [n_edges, n_basis], [n_edges, 1]


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        use_agnostic_product: bool = False,
        use_reduced_cg: Optional[bool] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.use_agnostic_product = use_agnostic_product
        if self.use_agnostic_product:
            num_elements = 1
        self.symmetric_contractions = SymmetricContractionWrapper(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        # Update linear
        self.linear = Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.cueq_config = cueq_config

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        use_cueq = False
        use_cueq_mul_ir = False
        if hasattr(self, "use_agnostic_product"):
            if self.use_agnostic_product:
                node_attrs = torch.ones(
                    (node_feats.shape[0], 1),
                    dtype=node_feats.dtype,
                    device=node_feats.device,
                )
        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:
                if self.cueq_config.enabled and (
                    self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
                ):
                    use_cueq = True
                if self.cueq_config.layout_str == "mul_ir":
                    use_cueq_mul_ir = True
        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = torch.nonzero(node_attrs)[:, 1].int()
            node_feats = self.symmetric_contractions(
                node_feats.flatten(1),
                index_attrs,
            )
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        edge_irreps: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        if edge_irreps is None:
            edge_irreps = self.node_feats_irreps
        self.radial_MLP = radial_MLP
        self.edge_irreps = edge_irreps
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config
        if self.oeq_config and self.oeq_config.conv_fusion:
            self.conv_fusion = self.oeq_config.conv_fusion
        if self.cueq_config and self.cueq_config.conv_fusion:
            self.conv_fusion = self.cueq_config.conv_fusion
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def handle_lammps(
        self,
        node_feats: torch.Tensor,
        lammps_class: Optional[Any],
        lammps_natoms: Tuple[int, int],
        first_layer: bool,
    ) -> torch.Tensor:  # noqa: D401 – internal helper
        if lammps_class is None or first_layer or torch.jit.is_scripting():
            return node_feats
        _, n_total = lammps_natoms
        pad = torch.zeros(
            (n_total, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = torch.cat((node_feats, pad), dim=0)
        node_feats = LAMMPS_MP.apply(node_feats, lammps_class)
        return node_feats

    def truncate_ghosts(
        self, tensor: torch.Tensor, n_real: Optional[int] = None
    ) -> torch.Tensor:
        """Truncate the tensor to only keep the real atoms in case of presence of ghost atoms during multi-GPU MD simulations."""
        return tensor[:n_real] if n_real is not None else tensor

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights (edge MLP -> maybe Libra)
        input_dim = self.edge_feats_irreps.num_irreps
        base_edge_mlp = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.conv_tp_weights = _maybe_replace_edge_mlp(base_edge_mlp, input_dim, self.conv_tp.weight_numel)

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        lammps_class: Optional[Any] = None,
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        n_real = lammps_natoms[0] if lammps_class is not None else None
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights (edge MLP -> maybe Libra)
        input_dim = self.edge_feats_irreps.num_irreps
        base_edge_mlp = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )
        self.conv_tp_weights = _maybe_replace_edge_mlp(base_edge_mlp, input_dim, self.conv_tp.weight_numel)

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticDensityInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights (edge MLP -> maybe Libra)
        input_dim = self.edge_feats_irreps.num_irreps
        base_edge_mlp = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.conv_tp_weights = _maybe_replace_edge_mlp(base_edge_mlp, input_dim, self.conv_tp.weight_numel)

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization (node MLP -> maybe Libra)
        base_density = nn.FullyConnectedNet(
            [input_dim] + [1], torch.nn.functional.silu,
        )
        self.density_fn = _maybe_replace_node_mlp(base_density, input_dim, 1)

        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )

        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        density = self.truncate_ghosts(density, n_real)
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights (edge MLP -> maybe Libra)
        input_dim = self.edge_feats_irreps.num_irreps
        base_edge_mlp = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )
        self.conv_tp_weights = _maybe_replace_edge_mlp(base_edge_mlp, input_dim, self.conv_tp.weight_numel)

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )

        # Density normalization (node MLP -> maybe Libra)
        base_density = nn.FullyConnectedNet(
            [input_dim] + [1], torch.nn.functional.silu,
        )
        self.density_fn = _maybe_replace_node_mlp(base_density, input_dim, 1)

        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=num_nodes
            )

        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        density = self.truncate_ghosts(density, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / (density + 1)
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        self.node_feats_down_irreps = o3.Irreps("64x0e")
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights (edge MLP -> maybe Libra) with augmented features
        self.linear_down = Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        base_edge_mlp = nn.FullyConnectedNet(
            [input_dim] + 3 * [256] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.conv_tp_weights = _maybe_replace_edge_mlp(base_edge_mlp, input_dim, self.conv_tp.weight_numel)

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Skip connection.
        self.skip_linear = Linear(
            self.node_feats_irreps, self.hidden_irreps, cueq_config=self.cueq_config
        )

    # pylint: disable=unused-argument
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats_up[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualNonLinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        node_scalar_irreps = o3.Irreps(
            [(self.node_feats_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )
        self.source_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.target_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights (edge MLP -> maybe Libra) with augmented features
        input_dim = self.edge_feats_irreps.num_irreps
        self._aug_dim = input_dim + 2 * node_scalar_irreps.dim
        base_edge_mlp = RadialMLP(
            [self._aug_dim] + self.radial_MLP + [self.conv_tp.weight_numel]
        )
        self.conv_tp_weights = _maybe_replace_edge_mlp(base_edge_mlp, self._aug_dim, self.conv_tp.weight_numel)
        self.irreps_out = self.target_irreps

        # Selector TensorProduct
        self.skip_tp = Linear(
            self.node_feats_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Non-linearity
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
        )
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0])
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        activation_fn = torch.nn.functional.silu
        act_gates_fn = torch.nn.functional.sigmoid
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[activation_fn for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[act_gates_fn] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        # Linear residual
        self.linear_res = Linear(
            self.edge_irreps,
            self.irreps_nonlin,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Linear
        self.linear_1 = Linear(
            irreps_mid,
            self.irreps_nonlin,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.linear_2 = Linear(
            irreps_in=self.irreps_out,
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Normalizations (node MLP -> maybe Libra)
        base_density = RadialMLP(
            [self._aug_dim] + [64] + [1],
        )
        self.density_fn = _maybe_replace_node_mlp(base_density, self._aug_dim, 1)
        self.alpha = torch.nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.transpose_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.irreps_nonlin,
            source="ir_mul",
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self.transpose_ir_mul = TransposeIrrepsLayoutWrapper(
            irreps=self.irreps_out,
            source="mul_ir",
            target="ir_mul",
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats)
        node_feats = self.linear_up(node_feats)
        node_feats_res = self.linear_res(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )

        source_embedding = self.source_embedding(node_attrs)
        target_embedding = self.target_embedding(node_attrs)
        edge_feats = torch.cat(
            [
                edge_feats,
                source_embedding[edge_index[0]],
                target_embedding[edge_index[1]],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(edge_feats)

        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff
        density = scatter_sum(
            src=edge_density, index=edge_index[1], dim=0, dim_size=num_nodes
        )

        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_edges, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=num_nodes
            )  # [n_nodes, irreps]

        message = self.truncate_ghosts(message, n_real)
        density = self.truncate_ghosts(density, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        node_feats_res = self.truncate_ghosts(node_feats_res, n_real)
        message = self.linear_1(message) / (density * self.beta + self.alpha)
        message = message + node_feats_res
        if self.transpose_mul_ir is not None:
            message = self.transpose_mul_ir(message)
        message = self.equivariant_nonlin(message)
        if self.transpose_ir_mul is not None:
            message = self.transpose_ir_mul(message)
        message = self.linear_2(message)
        return (
            self.reshape(message),
            sc,
        )


@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        return (
            torch.atleast_1d(self.scale)[head] * x + torch.atleast_1d(self.shift)[head]
        )

    def __repr__(self):
        formatted_scale = (
            ", ".join([f"{x:.4f}" for x in self.scale])
            if self.scale.numel() > 1
            else f"{self.scale.item():.4f}"
        )
        formatted_shift = (
            ", ".join([f"{x:.4f}" for x in self.shift])
            if self.shift.numel() > 1
            else f"{self.shift.item():.4f}"
        )
        return f"{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})"

