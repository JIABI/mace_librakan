# mace/tools/model_script_utils.py
import ast
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from e3nn import o3

from mace import modules
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.utils import AtomicNumberTable
from functools import partial

# =========================
# Minimal Mixer Extension
# =========================

from functools import partial


# ---------- helpers: bool / hidden / layers ----------
def _site_factory_call(ctor, in_features: int, out_features: int, **fixed):
    """
    Bridge MACE's site-MLP factory signature to your mixer ctor.

    MACE calls factories as:  factory(in_features=<int>, out_features=<int>)
    Your mixer expects:        ctor(in_dim=<int>, out_dim=<int>, **kwargs)

    This adapter converts the argument names and forwards any fixed kwargs.
    """
    return ctor(in_dim=int(in_features), out_dim=int(out_features), **fixed)

def _str2bool(x) -> bool:
    """Robust string->bool conversion for CLI/env flags."""
    return str(x).strip().lower() in ("1", "true", "yes", "on", "y", "t")


def _parse_optional_hidden(x):
    """
    Returns: None | int | list[int]
    Accepts: None, 1024, [1024,512], (1024,512), "1024,512", " [ 1024 , 512 ] ".
    For lists/tuples, all items must be convertible to int.
    """
    if x in (None, "", "None"):
        return None
    if isinstance(x, int):
        return int(x)
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if v in (None, "", "None"):
                continue
            out.append(int(v))
        return out if out else None

    s = str(x).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [int(p) for p in parts]
    return int(s)


def _parse_optional_layers(x):
    """
    Returns: None | list[ int | 'H' | 'H2' ]
    Accepts: None, "", [H], [H,128], (H, H2), "H,128", " [ H , 128 ] ".
    Only 'H'/'H2' placeholders and integers are allowed.
    """
    if x in (None, "", "None"):
        return None

    if isinstance(x, (list, tuple)):
        tokens = list(x)
    else:
        s = str(x).strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            s = s[1:-1].strip()
        tokens = [t.strip() for t in s.split(",")] if "," in s else [s]

    out = []
    for t in tokens:
        if t in ("", "None"):
            continue
        if t in ("H", "H2"):
            out.append(t)
        else:
            try:
                out.append(int(t))
            except Exception as e:
                raise ValueError(f"Invalid layer token for local_layers: {t!r}") from e
    return out if out else None


from functools import partial


# ---------- NEW: top-level core for scalar Libra readout (picklable) ----------

def _scalar_libra_readout_core(

        in_features: int,

        out_features: int,

        *,

        hidden_irreps,

        hidden,

        p,

        lam,

        trainable_lambda,

        activation,

        use_layernorm,

        dropout,

        local_kind,

        local_layers,

        F,

        learn_omega,

        es_fmax,

        spectral_scale,

        omega_max,

        alpha_min,

        alpha_tau,

        active_threshold,

):
    """

    Top-level factory for ReadoutScalarLibraKAN.

    IMPORTANT: This function is at module scope (not a local inner def),

    so a partial(_scalar_libra_readout_core, ...) is picklable.

    """

    from mace.modules.mixers.librakan import ReadoutScalarLibraKAN

    # We ignore in_features/out_features here, because ReadoutScalarLibraKAN

    # uses hidden_irreps.dim internally to set H.

    return ReadoutScalarLibraKAN(

        hidden_irreps=hidden_irreps,

        hidden=hidden,

        p=p,

        lam=lam,

        trainable_lambda=trainable_lambda,

        activation=activation,

        local_kind=local_kind,

        local_layers=local_layers,

        dropout=dropout,

        use_layernorm=use_layernorm,

        F=F,

        spectral_scale=spectral_scale,

        es_fmax=es_fmax,

        omega_max=omega_max,

        learn_omega=learn_omega,

        alpha_min=alpha_min,

        alpha_tau=alpha_tau,

        active_threshold=active_threshold,

    )


def _build_libra_readout_factory(args):
    """

    Picklable factory for scalar-only LibraKAN readout mixer.

    NOTE:

    We must align H with the readout MLP irreps (MLP_irreps), not the

    message-passing hidden irreps. Otherwise, x.shape[-1] != H and

    reshape(-1, H) in ReadoutScalarLibraKAN will fail.

    """

    from e3nn import o3

    import math

    # ---- use MLP_irreps for readout hidden space ----

    mlp_irreps_str = getattr(args, "MLP_irreps", None)

    if mlp_irreps_str in (None, "", "None"):

        hidden_irreps_str = getattr(args, "hidden_irreps", None)

        if hidden_irreps_str in (None, "", "None"):
            raise ValueError(

                "Libra readout requires args.MLP_irreps or args.hidden_irreps "

                "to be set; please pass --MLP_irreps on the command line."

            )

        hidden_irreps = o3.Irreps(hidden_irreps_str)

    else:

        hidden_irreps = o3.Irreps(mlp_irreps_str)

    # ---- shrinkage / activation / local branch ----

    lam = float(getattr(args, "libra_lambda_init", 1e-2))

    p = float(getattr(args, "libra_p", 1.0))

    trainable = _str2bool(getattr(args, "libra_lambda_trainable", "true"))

    act = str(getattr(args, "libra_base_activation", "gelu"))

    use_ln = _str2bool(getattr(args, "libra_readout_use_layernorm", "false"))

    dropout = float(getattr(args, "libra_readout_dropout", 0.0))

    local_kind = str(getattr(args, "libra_readout_local_kind", "mlp"))

    local_layers = _parse_optional_layers(

        getattr(args, "libra_readout_local_layers", "[H]")

    )

    # ---- spectral range ----

    es_fmax_arg = getattr(args, "libra_readout_es_fmax", None)

    if es_fmax_arg in (None, "", "None"):

        spec_scale = float(getattr(args, "libra_readout_spectral_scale", 1.0))

        omega_max = spec_scale * math.pi

        es_fmax = None

    else:

        es_fmax = float(es_fmax_arg)

        omega_max = es_fmax

        spec_scale = float(getattr(args, "libra_readout_spectral_scale", 1.0))

    # ---- dictionary size / learnable freqs ----

    F_arg = getattr(args, "libra_readout_F", None)

    F = None if F_arg in (None, "", "None") else int(F_arg)

    learn_omega = _str2bool(getattr(args, "libra_readout_learn_omega", "true"))

    # ---- fusion controls ----

    alpha_min = float(getattr(args, "libra_readout_alpha_min", 0.0))

    alpha_tau = float(getattr(args, "libra_readout_alpha_tau", 1.0))

    # ---- internal hidden width for LibraKAN over scalars ----

    raw_hidden = getattr(args, "readout_hidden", None)

    hidden = _parse_optional_hidden(raw_hidden)

    if isinstance(hidden, (list, tuple)):
        hidden = int(hidden[0]) if len(hidden) > 0 else None

    active_threshold = float(

        getattr(args, "libra_readout_active_threshold", 1e-3)

    )

    # IMPORTANT: do NOT define an inner def make() here.

    # We instead return a partial to a top-level function (_scalar_libra_readout_core),

    # which is picklable.

    make = partial(

        _scalar_libra_readout_core,

        hidden_irreps=hidden_irreps,

        hidden=hidden,

        p=p,

        lam=lam,

        trainable_lambda=trainable,

        activation=act,

        local_kind=local_kind,

        local_layers=local_layers,

        dropout=dropout,

        use_layernorm=use_ln,

        F=F,

        spectral_scale=spec_scale,

        es_fmax=es_fmax,

        omega_max=omega_max,

        learn_omega=learn_omega,

        alpha_min=alpha_min,

        alpha_tau=alpha_tau,

        active_threshold=active_threshold,
    )
    make.__mixer_kind__ = "libra_readout_scalar"
    return make

def _build_libra_node_factory1(args):
    """Picklable factory for GeneralLibraKAN used to replace NODE site MLPs."""
    from mace.modules.mixers.node_librakan import NodeLibraKAN
    import math

    lam = float(getattr(args, "libra_lambda_init", 1e-2))
    p = float(getattr(args, "libra_p", 1.0))
    trainable = _str2bool(getattr(args, "libra_lambda_trainable", "true"))
    act = str(getattr(args, "libra_base_activation", "gelu"))
    use_ln = _str2bool(getattr(args, "libra_node_use_layernorm", "false"))
    dropout = float(getattr(args, "libra_node_dropout", 0.0))
    local_kind = str(getattr(args, "libra_node_local_kind", "act"))
    local_layers = _parse_optional_layers(getattr(args, "libra_node_local_layers", ""))

    es_fmax_arg = getattr(args, "libra_node_es_fmax", None)
    if es_fmax_arg in (None, "", "None"):
        spec_scale = float(getattr(args, "libra_node_spectral_scale", 0.8))
        omega_max = spec_scale * math.pi
        es_fmax = None
    else:
        es_fmax = float(es_fmax_arg)
        omega_max = es_fmax

    F_arg = getattr(args, "libra_node_F", None)
    F = None if F_arg in (None, "", "None") else int(F_arg)
    learn_omega = _str2bool(getattr(args, "libra_node_learn_omega", "true"))
    alpha_min = float(getattr(args, "libra_node_alpha_min", 0.1))
    alpha_tau = float(getattr(args, "libra_node_alpha_tau", 1.0))
    use_cna = _str2bool(getattr(args, "libra_edge_use_cna", "true"))

    make = partial(
        _site_factory_call,
        NodeLibraKAN,
        hidden=None,
        p=p,
        lam=lam,
        trainable_lambda=trainable,
        activation=act,
        use_layernorm=use_ln,
        dropout=dropout,
        local_kind=local_kind,
        local_layers=local_layers,
        F=F,
        learn_omega=learn_omega,
        es_fmax=es_fmax,
        spectral_scale=float(getattr(args, "libra_node_spectral_scale", 0.8)),
        alpha_min=alpha_min,
        alpha_tau=alpha_tau,
        omega_max=omega_max,
        use_cna=use_cna,
    )
    make.__mixer_kind__ = "libra"
    return make


def _build_libra_node_factory(args):
    """Picklable factory for node LibraKAN site MLPs (scalar-only optional)."""
    import math
    from mace.modules.mixers.node_librakan import NodeLibraKAN
    from mace.modules.mixers.node_scalar_librakan import NodeScalarLibraKAN

    lam = float(getattr(args, "libra_lambda_init", 1e-1))
    p = float(getattr(args, "libra_p", 1.0))
    trainable = _str2bool(getattr(args, "libra_lambda_trainable", "false"))
    act = str(getattr(args, "libra_base_activation", "gelu"))
    use_ln = _str2bool(getattr(args, "libra_node_use_layernorm", "false"))
    dropout = float(getattr(args, "libra_node_dropout", 0.0))
    local_kind = str(getattr(args, "libra_node_local_kind", "act"))
    local_layers = _parse_optional_layers(getattr(args, "libra_node_local_layers", ""))

    # spectral range
    es_fmax_arg = getattr(args, "libra_node_es_fmax", None)
    if es_fmax_arg in (None, "", "None"):
        spec_scale = float(getattr(args, "libra_node_spectral_scale", 0.8))
        omega_max = spec_scale * math.pi
        es_fmax = None
    else:
        es_fmax = float(es_fmax_arg)
        omega_max = es_fmax

    F_arg = getattr(args, "libra_node_F", None)
    F = None if F_arg in (None, "", "None") else int(F_arg)
    learn_omega = _str2bool(getattr(args, "libra_node_learn_omega", "true"))
    alpha_min = float(getattr(args, "libra_node_alpha_min", 0.10))
    alpha_tau = float(getattr(args, "libra_node_alpha_tau", 1.00))

    scalar_only = _str2bool(getattr(args, "libra_node_scalar_only", "false"))
    hidden_irreps = getattr(args, "hidden_irreps", None)

    if scalar_only:
        # Use the scalar-only adapter; bridge (in_features, out_features) → ctor signature
        make = partial(
            _site_factory_call,
            NodeScalarLibraKAN,
            hidden_irreps=hidden_irreps,
            p=p, lam=lam, trainable_lambda=trainable, activation=act,
            F=F, spectral_scale=spec_scale, es_fmax=es_fmax,
            alpha_min=alpha_min, alpha_tau=alpha_tau, learn_omega=learn_omega,
            use_layernorm=use_ln, dropout=dropout,
            local_kind=local_kind, local_layers=local_layers,
            residual="add", beta=0.02,
        )
        make.__mixer_kind__ = "libra_node_scalar"
        return make

    # default: full GeneralLibraKAN on the whole node vector (square)
    make = partial(
        _site_factory_call,
        NodeLibraKAN,
        hidden=None,
        p=p, lam=lam, trainable_lambda=trainable, activation=act,
        F=F, spectral_scale=spec_scale, es_fmax=es_fmax,
        alpha_min=alpha_min, alpha_tau=alpha_tau, learn_omega=learn_omega,
        use_layernorm=use_ln, dropout=dropout,
        local_kind=local_kind, local_layers=local_layers,
    )
    make.__mixer_kind__ = "libra_node"
    return make

def _build_libra_edge_factory_librakan_version(args):
    """Picklable factory for GeneralLibraKAN used to replace EDGE site MLPs."""
    from mace.modules.mixers.librakan import GeneralLibraKAN
    import math

    lam = float(getattr(args, "libra_lambda_init", 1e-2))
    p = float(getattr(args, "libra_p", 1.0))
    trainable = _str2bool(getattr(args, "libra_lambda_trainable", "true"))
    act = str(getattr(args, "libra_base_activation", "gelu"))
    use_ln = _str2bool(getattr(args, "libra_edge_use_layernorm", "false"))
    dropout = float(getattr(args, "libra_edge_dropout", 0.0))
    local_kind = str(getattr(args, "libra_edge_local_kind", "act"))
    local_layers = _parse_optional_layers(getattr(args, "libra_edge_local_layers", ""))

    es_fmax_arg = getattr(args, "libra_edge_es_fmax", None)
    if es_fmax_arg in (None, "", "None"):
        spec_scale = float(getattr(args, "libra_edge_spectral_scale", 0.6))
        omega_max = spec_scale * math.pi
        es_fmax = None
    else:
        es_fmax = float(es_fmax_arg)
        omega_max = es_fmax

    F_arg = getattr(args, "libra_edge_F", None)
    F = None if F_arg in (None, "", "None") else int(F_arg)
    learn_omega = _str2bool(getattr(args, "libra_edge_learn_omega", "true"))
    alpha_min = float(getattr(args, "libra_edge_alpha_min", 0.05))
    alpha_tau = float(getattr(args, "libra_edge_alpha_tau", 1.2))
    use_cna = _str2bool(getattr(args, "libra_edge_use_cna", "true"))

    make = partial(
        _site_factory_call,
        GeneralLibraKAN,
        hidden=None,
        p=p,
        lam=lam,
        trainable_lambda=trainable,
        activation=act,
        use_layernorm=use_ln,
        dropout=dropout,
        local_kind=local_kind,
        local_layers=local_layers,
        F=F,
        learn_omega=learn_omega,
        es_fmax=es_fmax,
        spectral_scale=float(getattr(args, "libra_edge_spectral_scale", 0.6)),
        alpha_min=alpha_min,
        alpha_tau=alpha_tau,
        omega_max=omega_max,
        use_cna = use_cna
    )
    make.__mixer_kind__ = "libra"
    return make


def _build_libra_edge_factory(args):
    """Picklable factory for edge PhysicsSpectralRadialMixer (reusing --edge_librakan flag)."""
    from mace.modules.mixers.edge_physics_spectral import EdgePhysicsSpectralMixer

    # Reuse global radial settings as physical prior
    num_basis = int(getattr(args, "num_radial_basis", 8))
    cutoff = float(getattr(args, "r_max", 5.0))
    envelope_exponent = int(getattr(args, "num_cutoff_basis", 5))

    # Light-weight residual configuration (can be overridden via CLI if desired)
    residual_hidden = (
        int(getattr(args, "libra_edge_hidden", 32))
        if hasattr(args, "libra_edge_hidden")
        else 32
    )
    dropout = float(getattr(args, "libra_edge_dropout", 0.0))
    base_activation = str(getattr(args, "libra_base_activation", "gelu"))

    make = partial(
        _site_factory_call,
        EdgePhysicsSpectralMixer,
        num_basis=num_basis,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent,
        residual_hidden=residual_hidden,
        dropout=dropout,
        base_activation=base_activation,
    )
    # Tag for logging / debugging
    make.__mixer_kind__ = "libra"
    return make

def _build_kaf_readout_factory(args):
    """
    Picklable factory for KAF readout mixer.
    Use your adapter or concrete class; do NOT return a local def/lambda.
    """
    # NOTE: You are importing from capitalized file names; keep exactly as your codebase:
    from mace.modules.mixers.KAF_MIXER import KAFMixer as _KAF
    make = partial(_KAF)   # only (in_dim, out_dim) will be passed by the caller
    make.__mixer_kind__ = "kaf"
    return make


def _build_kan_readout_factory(args):
    """
    Picklable factory for KAN readout mixer.
    """
    from mace.modules.mixers.KAN_MIXER import KANMixer as _KAN
    make = partial(_KAN)
    make.__mixer_kind__ = "kan"
    return make


def _resolve_readout_and_mixers(args):
    if getattr(args, "librakan_readout", False):
        ro_kind = "libra"
    elif getattr(args, "kaf_readout", False):
        ro_kind = "kaf"
    elif getattr(args, "kan_readout", False):
        ro_kind = "kan"
    else:
        ro_kind = "mlp"

    ro_cls = modules.NonLinearReadoutBlock
    ro_kwargs: Dict[str, Any] = {}

    if ro_kind == "libra":
        ro_kwargs = {
            "readout_mixer_kind": "libra",
            "readout_mixer_factory": _build_libra_readout_factory(args),
        }
    elif ro_kind == "kaf":
        ro_kwargs = {
            "readout_mixer_kind": "kaf",
            "readout_mixer_factory": _build_kaf_readout_factory(args),
        }
    elif ro_kind == "kan":
        ro_kwargs = {
            "readout_mixer_kind": "kan",
            "readout_mixer_factory": _build_kan_readout_factory(args),
        }

    rh = getattr(args, "readout_hidden", None)
    if rh not in (None, "", "None"):
        ro_kwargs["readout_hidden"] = rh

    # Interaction site factories (FLAT dict — this is what MACE expects)
    inter_factories: Dict[str, Any] = {}
    if getattr(args, "node_librakan", False):
        inter_factories["node_mlp_factory"] = _build_libra_node_factory(args)
    if getattr(args, "edge_librakan", False):
        inter_factories["edge_mlp_factory"] = _build_libra_edge_factory(args)
    print(
        f"[resolver] readout_kind={ro_kind}, "
        f"node_libra={'node_mlp_factory' in inter_factories}, "
        f"edge_libra={'edge_mlp_factory' in inter_factories}",
        flush=True,
    )

    return ro_cls, ro_kwargs, inter_factories


# =========================
# Official configure_model (保留官方结构)
# =========================

def configure_model(
    args,
    train_loader,
    atomic_energies,
    model_foundation=None,
    heads=None,
    z_table=None,
    head_configs=None,
):
    logging.info(f"HEADS used: {heads}")
    # Selecting outputs
    compute_virials = args.loss == "virials"
    compute_stress = args.loss in ("stress", "huber", "universal")

    if compute_virials:
        args.compute_virials = True
        args.error_table = "PerAtomRMSEstressvirials"
    elif compute_stress:
        args.compute_stress = True
        args.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": args.compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": compute_stress,
        "dipoles": args.compute_dipole,
        "polarizabilities": args.compute_polarizability,
    }
    logging.info(
        "During training the following quantities will be reported: "
        + ", ".join([rep for rep, v in output_args.items() if v])
    )
    logging.info("===========MODEL DETAILS===========")

    # Scaling
    if args.scaling == "no_scaling":
        args.std = 1.0
        if head_configs is not None:
            for head_config in head_configs:
                head_config.std = 1.0
        logging.info("No scaling selected")

    if (
        head_configs is not None
        and args.std is not None
        and not isinstance(args.std, list)
    ):
        atomic_inter_scale = []
        for head_config in head_configs:
            if hasattr(head_config, "std") and head_config.std is not None:
                atomic_inter_scale.append(head_config.std)
            elif args.std is not None:
                atomic_inter_scale.append(
                    args.std if isinstance(args.std, float) else 1.0
                )
        args.std = atomic_inter_scale
    elif (args.mean is None or args.std is None) and (
        args.model not in ("AtomicDipolesMACE", "AtomicDielectricMACE")
    ):
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )

    # Optional embedding specs
    if args.embedding_specs is not None:
        args.embedding_specs = ast.literal_eval(args.embedding_specs)
        logging.info("Using embedding specifications from command line arguments")
        logging.info(f"Embedding specifications: {args.embedding_specs}")

    # ===== Foundation path =====
    if model_foundation is not None and args.model in [
        "MACE",
        "ScaleShiftMACE",
        "MACELES",
    ]:
        logging.info("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)
        model_config_foundation["atomic_energies"] = atomic_energies

        if args.foundation_model_elements:
            foundation_z_table = AtomicNumberTable(
                [int(z) for z in model_foundation.atomic_numbers]
            )
            model_config_foundation["atomic_numbers"] = foundation_z_table.zs
            model_config_foundation["num_elements"] = len(foundation_z_table)
            z_table = foundation_z_table
            logging.info(
                f"Using all elements from foundation model: {foundation_z_table.zs}"
            )
        else:
            model_config_foundation["atomic_numbers"] = z_table.zs
            model_config_foundation["num_elements"] = len(z_table)
            logging.info(f"Using filtered elements: {z_table.zs}")

        args.max_L = model_config_foundation["hidden_irreps"].lmax

        if (
            args.model == "ScaleShiftMACE"
            or model_foundation.__class__.__name__ == "ScaleShiftMACE"
        ):
            model_config_foundation["atomic_inter_shift"] = (
                _determine_atomic_inter_shift(args.mean, heads)
            )
        else:
            model_config_foundation["atomic_inter_shift"] = [0.0] * len(heads)
        model_config_foundation["atomic_inter_scale"] = [1.0] * len(heads)
        args.avg_num_neighbors = model_config_foundation["avg_num_neighbors"]
        args.model = (
            "FoundationMACELES" if args.model == "MACELES" else "FoundationMACE"
        )
        model_config_foundation["heads"] = heads
        model_config = model_config_foundation

        logging.info("Model configuration extracted from foundation model")
        logging.info(f"Using {args.loss} loss function for fine-tuning")
        logging.info(
            f"Message passing with hidden irreps {model_config_foundation['hidden_irreps']})"
        )
        logging.info(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: "
            f"{model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) "
            f"and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.info(
            f"Radial cutoff: {model_config_foundation['r_max']} A "
            f"(total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )
    else:
        # ===== Fresh build path =====
        logging.info("Building model")
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
        )
        logging.info(
            f"{args.num_interactions} layers, each with correlation order: {args.correlation} "
            f"(body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
        )
        logging.info(
            f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
        )
        logging.info(
            f"Radial cutoff: {args.r_max} A (total receptive field for each atom: {args.r_max * args.num_interactions} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {args.distance_transform}"
        )

        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        logging.info(f"Hidden irreps: {args.hidden_irreps}")

        cueq_config = None
        if args.only_cueq:
            logging.info("Using only the backend of the model")
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
                conv_fusion=(args.device == "cuda"),
            )

        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            edge_irreps=o3.Irreps(args.edge_irreps) if args.edge_irreps else None,
            atomic_energies=atomic_energies,
            apply_cutoff=args.apply_cutoff,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
            use_reduced_cg=args.use_reduced_cg,
            use_so3=args.use_so3,
            cueq_config=cueq_config,
            # --- MIL flags ---
            use_mil_pooling=args.use_mil_pooling,
            mil_d_attn=args.mil_d_attn,
            mil_dropout=args.mil_dropout,
        )
        model_config_foundation = None

    model = _build_model(args, model_config, model_config_foundation, heads)

    if model_foundation is not None:
        model = load_foundations_elements(
            model,
            model_foundation,
            z_table,
            load_readout=args.foundation_filter_elements,
            max_L=args.max_L,
        )

    return model, output_args


def _determine_atomic_inter_shift(mean, heads):
    if isinstance(mean, np.ndarray):
        if mean.size == 1:
            return mean.item()
        if mean.size == len(heads):
            return mean.tolist()
        logging.info("Mean not in correct format, using default value of 0.0")
        return [0.0] * len(heads)
    if isinstance(mean, list) and len(mean) == len(heads):
        return mean
    if isinstance(mean, float):
        return [mean] * len(heads)
    logging.info("Mean not in correct format, using default value of 0.0")
    return [0.0] * len(heads)


def _build_model(
    args, model_config, model_config_foundation, heads
):  # pylint: disable=too-many-return-statements
    def _purge_mil_keys(cfg: dict):
        for k in ("use_mil_pooling", "mil_d_attn", "mil_dropout"):
            if k in cfg:
                cfg.pop(k, None)

    # 解析读出层 + 交互层 Mixer
    ro_cls, ro_kwargs, inter_factories = _resolve_readout_and_mixers(args)

    if args.model == "MACE":
        if args.interaction_first not in [
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
        ]:
            args.interaction_first = "RealAgnosticInteractionBlock"
        _purge_mil_keys(model_config)
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
            # === MIXERS 注入 ===
            readout_cls=ro_cls,
            readout_kwargs=ro_kwargs,
            interaction_mlp_factories= inter_factories,
            # --- MIL flags ---
            use_mil_pooling=args.use_mil_pooling,
            mil_d_attn=args.mil_d_attn,
            mil_dropout=args.mil_dropout,
        )

    if args.model == "ScaleShiftMACE":
        _purge_mil_keys(model_config)
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
            # === MIXERS 注入 ===
            readout_cls=ro_cls,
            readout_kwargs=ro_kwargs,
            interaction_mlp_factories= inter_factories,
        )

    if args.model == "FoundationMACE":
        _purge_mil_keys(model_config)
        return modules.ScaleShiftMACE(
            **model_config_foundation,
            # === MIXERS 注入（若支持会生效）===
            readout_cls=ro_cls,
            readout_kwargs=ro_kwargs,
            interaction_mlp_factories= inter_factories,
            # --- MIL flags ---
            use_mil_pooling=args.use_mil_pooling,
            mil_d_attn=args.mil_d_attn,
            mil_dropout=args.mil_dropout,
        )

    if args.model == "FoundationMACELES":
        from mace.modules.extensions import MACELES
        _purge_mil_keys(model_config)
        return MACELES(
            les_arguments=args.les_arguments,
            **model_config_foundation,
            # === MIXERS 注入 ===
            readout_cls=ro_cls,
            readout_kwargs=ro_kwargs,
            interaction_mlp_factories= inter_factories,
            # --- MIL flags ---
            use_mil_pooling=args.use_mil_pooling,
            mil_d_attn=args.mil_d_attn,
            mil_dropout=args.mil_dropout,
        )

    if args.model == "ScaleShiftBOTNet":
        raise RuntimeError("ScaleShiftBOTNet is deprecated, use MACE instead")

    if args.model == "BOTNet":
        raise RuntimeError("BOTNet is deprecated, use MACE instead")

    if args.model == "AtomicDipolesMACE":
        assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            args.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        return modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )

    if args.model == "AtomicDielectricMACE":
        args.error_table = "DipolePolarRMSE"
        assert (
            args.loss == "dipole_polar"
        ), "Use dipole_polar loss with AtomicDielectricMACE model"
        assert args.error_table in (
            "DipoleRMSE",
            "DipolePolarRMSE",
        ), "Use error_table DipoleRMSE with AtomicDielectricMACE model"
        return modules.AtomicDielectricMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            use_polarizability=True,
        )

    if args.model == "EnergyDipolesMACE":
        assert (
            args.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            args.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        return modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )

    if args.model == "MACELES":
        from mace.modules.extensions import MACELES
        _purge_mil_keys(model_config)
        return MACELES(
            les_arguments=args.les_arguments,
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            embedding_specs=args.embedding_specs,
            use_embedding_readout=args.use_embedding_readout,
            use_last_readout_only=args.use_last_readout_only,
            use_agnostic_product=args.use_agnostic_product,
            # === MIXERS 注入 ===
            readout_cls=ro_cls,
            readout_kwargs=ro_kwargs,
            interaction_mlp_factories= inter_factories,
            # --- MIL flags ---
            use_mil_pooling=args.use_mil_pooling,
            mil_d_attn=args.mil_d_attn,
            mil_dropout=args.mil_dropout,
        )

    raise RuntimeError(f"Unknown model: '{args.model}'")

