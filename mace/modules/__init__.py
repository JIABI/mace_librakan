# mace/modules/__init__.py
from typing import Callable, Dict, Optional, Type

import torch

# ---- Core blocks / ops ----
from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearBiasReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
    ScaleShiftBlock,
)

from .loss import (
    DipolePolarLoss,
    DipoleSingleLoss,
    UniversalLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesL1L2Loss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
)

from .models import (
    MACE,
    AtomicDielectricMACE,
    AtomicDipolesMACE,
    EnergyDipolesMACE,
    ScaleShiftMACE,
)

from .radial import BesselBasis, GaussianBasis, PolynomialCutoff, ZBLBasis
from .symmetric_contraction import SymmetricContraction
from .utils import (
    compute_avg_num_neighbors,
    compute_dielectric_gradients,
    compute_fixed_charge_dipole,
    compute_fixed_charge_dipole_polar,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_rms_dipoles,
    compute_statistics,
)

# ---- Optional: mixers.make_mixer (KAN/KAF/Libra) ----
# If your repo ships mace/modules/mixers/__init__.py exposing make_mixer,
# we surface it here for convenience; otherwise it's simply absent.
try:
    from .mixers import make_mixer  # noqa: F401
except Exception:
    make_mixer = None  # type: ignore

# ---- Optional: MIL pooling ----
try:
    from .mil_pooling import ConjunctivePooling  # noqa: F401
except Exception:
    ConjunctivePooling = None  # type: ignore

# ---------- Registries ----------

# Interaction block registry
interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
    "RealAgnosticResidualNonLinearInteractionBlock": RealAgnosticResidualNonLinearInteractionBlock,
}

# Readout registry
# 为了兼容：即使选择了 Libra/KAF/KAN，这里也统一指向 NonLinearReadoutBlock，
# 实际的 mixer 由 tools/model_script_utils._resolve_readout_and_mixers() 通过
# readout_kwargs['readout_mixer_kind'] 在 blocks.NonLinearReadoutBlock 内部完成替换。
readout_classes: Dict[str, Type[LinearReadoutBlock]] = {
    "LinearReadoutBlock": LinearReadoutBlock,
    "LinearDipoleReadoutBlock": LinearDipoleReadoutBlock,
    "NonLinearDipoleReadoutBlock": NonLinearDipoleReadoutBlock,
    "NonLinearReadoutBlock": NonLinearReadoutBlock,
    "NonLinearBiasReadoutBlock": NonLinearBiasReadoutBlock,
    # Thin aliases to avoid KeyError in legacy configs
    "LibraReadoutBlock": NonLinearReadoutBlock,
    "KAFReadoutBlock": NonLinearReadoutBlock,
    "KANReadoutBlock": NonLinearReadoutBlock,
}

# Scaling/normalization strategies
scaling_classes: Dict[str, Callable] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
    "rms_dipoles_scaling": compute_rms_dipoles,
}

# Gate dictionary
gate_dict: Dict[str, Optional[Callable]] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

__all__ = [
    # Core blocks
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "ZBLBasis",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "LinearDipoleReadoutBlock",
    "LinearDipolePolarReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "NonLinearDipolePolarReadoutBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "GaussianBasis",
    # Models
    "MACE",
    "ScaleShiftMACE",
    "AtomicDipolesMACE",
    "AtomicDielectricMACE",
    "EnergyDipolesMACE",
    # Losses
    "WeightedEnergyForcesLoss",
    "WeightedForcesLoss",
    "WeightedEnergyForcesVirialsLoss",
    "WeightedEnergyForcesStressLoss",
    "DipoleSingleLoss",
    "WeightedEnergyForcesDipoleLoss",
    "WeightedHuberEnergyForcesStressLoss",
    "UniversalLoss",
    "WeightedEnergyForcesL1L2Loss",
    # Utils
    "SymmetricContraction",
    "interaction_classes",
    "compute_mean_std_atomic_inter_energy",
    "compute_avg_num_neighbors",
    "compute_statistics",
    "compute_fixed_charge_dipole",
    "compute_fixed_charge_dipole_polar",
    "compute_dielectric_gradients",
    # Registries & helpers
    "readout_classes",
    "scaling_classes",
    "gate_dict",
]

# Optional exports if present
if make_mixer is not None:
    __all__.append("make_mixer")
if ConjunctivePooling is not None:
    __all__.append("ConjunctivePooling")
