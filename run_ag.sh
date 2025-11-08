#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MACE + (KAN | KAF | LibraKAN) — single switch script
# Usage:
#   bash train_ag_mixer.sh -m kan
#   bash train_ag_mixer.sh -m kaf
#   bash train_ag_mixer.sh -m libra
# or:
#   MIXER=libra NODE_LIBRAKAN=true EDGE_LIBRAKAN=true bash train_ag_mixer.sh
# ==============================================================================

# ----------------------------- CLI parsing ------------------------------------
MIXER="${MIXER:-libra}"          # default: libra
NODE_LIBRAKAN="${NODE_LIBRAKAN:-true}"
EDGE_LIBRAKAN="${EDGE_LIBRAKAN:-false}"

while getopts ":m:n:e:" opt; do
  case $opt in
    m) MIXER="$OPTARG" ;;                    # kan|kaf|libra
    n) NODE_LIBRAKAN="$OPTARG" ;;            # true|false
    e) EDGE_LIBRAKAN="$OPTARG" ;;            # true|false
    *) echo "Usage: $0 [-m kan|kaf|libra] [-n true|false] [-e true|false]"; exit 1 ;;
  esac
done

# ----------------------------- Dataset ----------------------------------------
AG_TRAIN="/home/ubuntu/PycharmProjects/mace/ag/Ag-train.xyz"
AG_VALID="/home/ubuntu/PycharmProjects/mace/ag/Ag-valid.xyz"
AG_TEST="/home/ubuntu/PycharmProjects/mace/ag/Ag-test.xyz"
NAME="MACE_Ag_${MIXER}"

# ----------------------------- Core train -------------------------------------
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float64}"
SEED="${SEED:-1}"
LR="${LR:-0.008}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-8}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
SCHED_PATIENCE="${SCHED_PATIENCE:-5}"
LR_FACTOR="${LR_FACTOR:-0.5}"
EMA_DECAY="${EMA_DECAY:-0.99999}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-4}"
HIDDEN_IRREPS="${HIDDEN_IRREPS:-32x0e}"
MLP_IRREPS="${MLP_IRREPS:-16x0e}"
NUM_CHANNELS="${NUM_CHANNELS:-128}"
NUM_RADIAL_BASIS="${NUM_RADIAL_BASIS:-8}"
MAX_ELL="${MAX_ELL:-3}"
MAX_L="${MAX_L:-1}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
R_MAX="${R_MAX:-6.0}"
CORRELATION="${CORRELATION:-2}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/ubuntu/Downloads/mace-kan/checkpoints}"

LOSS="${LOSS:-universal}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-10.0}"
COMPUTE_STRESS="${COMPUTE_STRESS:-true}"
STRESS_WEIGHT="${STRESS_WEIGHT:-10}"
ERROR_TABLE="${ERROR_TABLE:-PerAtomMAE}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1}"

USE_SWA="${USE_SWA:-true}"
START_SWA="${START_SWA:-30}"
SWA_ENERGY_WEIGHT="${SWA_ENERGY_WEIGHT:-1.0}"
SWA_FORCES_WEIGHT="${SWA_FORCES_WEIGHT:-50.0}"

CLIP_GRAD="${CLIP_GRAD:-100}"

# ----------------------------- KAF params -------------------------------------
KAF_F="${KAF_F:-256}"
KAF_DROPOUT="${KAF_DROPOUT:-0.0}"
KAF_USE_LAYERNORM="${KAF_USE_LAYERNORM:-false}"
KAF_BASE_ACTIVATION="${KAF_BASE_ACTIVATION:-gelu}"
KAF_ACT_EXPECTATION="${KAF_ACT_EXPECTATION:-1.64}"
KAF_HIDDEN="${KAF_HIDDEN:-[1024,512]}"   # 可空，例："[1024,512]"

# ----------------------------- LibraKAN params --------------------------------
LIBRA_F="${LIBRA_F:-128}"
LIBRA_SPECTRAL_SCALE="${LIBRA_SPECTRAL_SCALE:-0.7}"
LIBRA_ES_BETA="${LIBRA_ES_BETA:-6.0}"
LIBRA_ES_FMAX="${LIBRA_ES_FMAX:-}"          # 空=自动；如需限制设 0.45
LIBRA_LAMBDA_INIT="${LIBRA_LAMBDA_INIT:-0.01}"
LIBRA_LAMBDA_TRAINABLE="${LIBRA_LAMBDA_TRAINABLE:-true}"
LIBRA_L1_ALPHA="${LIBRA_L1_ALPHA:-0}"
LIBRA_DROPOUT="${LIBRA_DROPOUT:-0.0}"
LIBRA_BASE_ACTIVATION="${LIBRA_BASE_ACTIVATION:-gelu}"
LIBRA_USE_LAYERNORM="${LIBRA_USE_LAYERNORM:-false}"
READOUT_HIDDEN="${READOUT_HIDDEN:-[1024,512]}"  # 可空

# -------------------- Compute mean per-atom E0 from XYZ comments --------------
mean_e0=$(
python3 - "$AG_TRAIN" <<'PY'
import re, sys, numpy as np
fn = sys.argv[1]
energies, natoms = [], []
with open(fn) as f:
    while True:
        line = f.readline()
        if not line: break
        try:
            n = int(line.strip())
        except ValueError:
            break
        natoms.append(n)
        c = f.readline()
        m = re.search(r'energy\s*=\s*([-\d\.Ee+]+)', c)
        if not m:
            raise RuntimeError(f"No energy in comment: {c!r}")
        energies.append(float(m.group(1)))
        for _ in range(n):
            f.readline()
energies = np.array(energies, float)
natoms = np.array(natoms, float)
print((energies/natoms).mean())
PY
)
echo "[`date '+%Y-%m-%d %H:%M:%S'`] Mean E0/Ag = ${mean_e0} eV"

# ----------------------------- Mixer flags ------------------------------------
MIXER_FLAGS=()
case "${MIXER}" in
  mlp)
    echo "[Mixer] Using baseline MLP readout"
    ;;
  kan)
    echo "[Mixer] Using KAN readout"
    MIXER_FLAGS+=(--kan_readout)
    ;;
  kaf)
    echo "[Mixer] Using KAF readout"
    MIXER_FLAGS+=(--kaf_readout --kaf_F "${KAF_F}" --kaf_dropout "${KAF_DROPOUT}")
    MIXER_FLAGS+=(--kaf_use_layernorm "${KAF_USE_LAYERNORM}" --kaf_base_activation "${KAF_BASE_ACTIVATION}")
    MIXER_FLAGS+=(--kaf_activation_expectation "${KAF_ACT_EXPECTATION}")
    if [[ -n "${KAF_HIDDEN}" ]]; then MIXER_FLAGS+=(--kaf_hidden "${KAF_HIDDEN}"); fi
    ;;
  libra)
    echo "[Mixer] Using LibraKAN readout"
    MIXER_FLAGS+=(--librakan_readout --libra_F "${LIBRA_F}" --libra_spectral_scale "${LIBRA_SPECTRAL_SCALE}")
    MIXER_FLAGS+=(--libra_es_beta "${LIBRA_ES_BETA}" --libra_lambda_init "${LIBRA_LAMBDA_INIT}")
    MIXER_FLAGS+=(--libra_lambda_trainable "${LIBRA_LAMBDA_TRAINABLE}" --libra_l1_alpha "${LIBRA_L1_ALPHA}")
    MIXER_FLAGS+=(--libra_dropout "${LIBRA_DROPOUT}" --libra_base_activation "${LIBRA_BASE_ACTIVATION}")
    MIXER_FLAGS+=(--libra_use_layernorm "${LIBRA_USE_LAYERNORM}")
    if [[ -n "${LIBRA_ES_FMAX}" ]]; then MIXER_FLAGS+=(--libra_es_fmax "${LIBRA_ES_FMAX}"); fi
    if [[ -n "${READOUT_HIDDEN}" ]]; then MIXER_FLAGS+=(--readout_hidden "${READOUT_HIDDEN}"); fi
    ;;
  *)
    echo "Invalid MIXER='${MIXER}'. Use: mlp | kan | kaf | libra"; exit 1 ;;
esac

if [[ "${NODE_LIBRAKAN}" == "true" ]]; then
  echo "[Mixer] NODE LibraKAN enabled"
  MIXER_FLAGS+=(--node_librakan)
fi
if [[ "${EDGE_LIBRAKAN}" == "true" ]]; then
  echo "[Mixer] EDGE LibraKAN enabled"
  MIXER_FLAGS+=(--edge_librakan)
fi

# ----------------------------- Train ------------------------------------------
start_time=$(date +%s)
echo "[`date '+%Y-%m-%d %H:%M:%S'`] Training ${NAME} …"

mace_run_train \
  --name "${NAME}" \
  --train_file "${AG_TRAIN}" \
  --valid_file "${AG_VALID}" \
  --test_file "${AG_TEST}" \
  --E0s "{47: ${mean_e0}}" \
  --atomic_numbers "[47]" \
  --use_mil_pooling false \
  --mil_d_attn 16 \
  --mil_dropout 0.1 \
  --energy_key dft_energy \
  --forces_key dft_forces \
  --loss "${LOSS}" \
  --num_workers 4 \
  --energy_weight "${ENERGY_WEIGHT}" \
  --forces_weight "${FORCES_WEIGHT}" \
  --compute_stress "${COMPUTE_STRESS}" \
  --stress_weight "${STRESS_WEIGHT}" \
  --stress_key stress \
  --eval_interval "${EVAL_INTERVAL}" \
  --error_table "${ERROR_TABLE}" \
  --model MACE \
  --interaction_first RealAgnosticResidualInteractionBlock \
  --interaction RealAgnosticResidualInteractionBlock \
  --num_interactions "${NUM_INTERACTIONS}" \
  --max_ell "${MAX_ELL}" \
  --hidden_irreps "${HIDDEN_IRREPS}" \
  --num_channels "${NUM_CHANNELS}" \
  --num_radial_basis "${NUM_RADIAL_BASIS}" \
  --MLP_irreps "${MLP_IRREPS}" \
  --scaling rms_forces_scaling \
  --correlation "${CORRELATION}" \
  --r_max "${R_MAX}" \
  --save_cpu \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --max_num_epochs "${MAX_EPOCHS}" \
  --scheduler_patience "${SCHED_PATIENCE}" \
  --lr_factor "${LR_FACTOR}" \
  $( [[ "${USE_SWA}" == "true" ]] && printf -- "--swa --start_swa %s --swa_energy_weight %s --swa_forces_weight %s" "${START_SWA}" "${SWA_ENERGY_WEIGHT}" "${SWA_FORCES_WEIGHT}" ) \
  --ema \
  --ema_decay "${EMA_DECAY}" \
  --amsgrad \
  --default_dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --clip_grad "${CLIP_GRAD}" \
  --pair_repulsion \
  --distance_transform Agnesi \
  --max_L "${MAX_L}" \
  --batch_size "${BATCH_SIZE}" \
  --valid_batch_size "${VALID_BATCH_SIZE}" \
  --keep_checkpoints \
  --save_all_checkpoints \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --patience 40 \
  "${MIXER_FLAGS[@]}"

end_time=$(date +%s)
echo "[`date '+%Y-%m-%d %H:%M:%S'`] Finished ${NAME}"
elapsed=$(( end_time - start_time ))
printf "Total time: %02d:%02d:%02d\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))