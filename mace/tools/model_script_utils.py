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

def _str2bool(x: str) -> bool:
    return str(x).lower() in ("1", "true", "yes", "on")
def _site_factory_call(ctor, in_features, out_features, **fixed):
    """Bridge MACE's (in_features, out_features) to mixer ctor(in_dim, out_dim)."""
    return ctor(in_dim=int(in_features), out_dim=int(out_features), **fixed)
def _parse_optional_hidden(x):
    """
    Return: None | int | List[int]
    Accepts None, int, list/tuple of ints, string forms like "1024", "1024,512", "[1024,512]".
    """
    if x in (None, "", "None"):
        return None
    # already parsed types
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    # string cases
    if isinstance(x, str):
        s = x.strip()
        # bracketed Python literal list/tuple
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                import ast as _ast
                seq = _ast.literal_eval(s)
                if isinstance(seq, (list, tuple)):
                    return [int(v) for v in seq]
            except Exception:
                pass
        # comma-separated "1024,512"
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return [int(p) for p in parts]
        # single int in string
        return int(s)
    # fallback: try int, else raise
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Cannot parse readout_hidden={x!r}") from e


def _build_libra_readout_factory(args):
    """Picklable factory for GeneralLibraKAN readout."""
    from mace.modules.mixers.librakan import GeneralLibraKAN
    import math

    lam = float(getattr(args, "libra_lambda_init", 1e-2))
    p = float(getattr(args, "libra_p", 1.0))
    trainable = _str2bool(getattr(args, "libra_lambda_trainable", "true"))
    act = str(getattr(args, "libra_base_activation", "gelu"))

    # omega_max: prefer es_fmax if provided; else spectral_scale * pi
    _es_fmax = getattr(args, "libra_es_fmax", None)
    if _es_fmax in (None, "", "None"):
        spec_scale = float(getattr(args, "libra_spectral_scale", 1.0))
        omega_max = spec_scale * math.pi
    else:
        omega_max = float(_es_fmax)

    # robustly parse hidden (None | int | List[int])
    raw_hidden = getattr(args, "readout_hidden", None)
    hidden = _parse_optional_hidden(raw_hidden)
    # If GeneralLibraKAN expects a single int, collapse list/tuple to first
    if isinstance(hidden, (list, tuple)):
        hidden = int(hidden[0]) if len(hidden) > 0 else None

    # Return a top-level picklable callable. in_dim/out_dim will be provided at call-site.
    make = partial(
        GeneralLibraKAN,
        hidden=hidden,
        p=float(p),
        lam=float(lam),
        trainable_lambda=bool(trainable),
        activation=act,
        omega_max=float(omega_max),
    )
    make.__mixer_kind__ = "libra"  # tag for kind inference in blocks.py
    return make


def _build_libra_site_factory(args):
    """Picklable factory for node/edge GeneralLibraKAN (site MLP replacement)."""
    from mace.modules.mixers.librakan import GeneralLibraKAN
    import math

    lam = float(getattr(args, "libra_lambda_init", 1e-2))
    p = float(getattr(args, "libra_p", 1.0))
    trainable = _str2bool(getattr(args, "libra_lambda_trainable", "true"))
    act = str(getattr(args, "libra_base_activation", "gelu"))

    _es_fmax = getattr(args, "libra_es_fmax", None)
    if _es_fmax in (None, "", "None"):
        spec_scale = float(getattr(args, "libra_spectral_scale", 1.0))
        omega_max = spec_scale * math.pi
    else:
        omega_max = float(_es_fmax)

    make = partial(
        _site_factory_call,
        GeneralLibraKAN,
        hidden=None,  # site MLP: keep single layer unless you add multi-layer support
        p=float(p),
        lam=float(lam),
        trainable_lambda=bool(trainable),
        activation=act,
        omega_max=float(omega_max)*0.6,
        F = min(32, int(getattr(args, "libra_F", 96.0))),
    )
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
        inter_factories["node_mlp_factory"] = _build_libra_site_factory(args)
    if getattr(args, "edge_librakan", False):
        inter_factories["edge_mlp_factory"] = _build_libra_site_factory(args)
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

