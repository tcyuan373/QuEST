#!/bin/bash
# Mini-scale experiment: 50M Llama + slimpajama (fixed 216M-token train.bin),
# 512 optimizer steps (~134M tokens). Parameterized for the QMC-SR ablation:
#   ARM=qmcfull  (default) QMC-SR fwd weights + Muon NS-then-round   [job 655217]
#   ARM=qmcfwd            QMC-SR fwd weights + plain Muon
#   ARM=iidfwd            iid-SR fwd weights + plain Muon
#   ARM=rtn               deterministic RTN on the SAME uniform grid + plain Muon
#                         (STEQuantizer = parent grid of QMCSRSTEQuantizer, so
#                         rtn vs iidfwd/qmcfwd isolates deterministic-vs-stochastic
#                         with the grid held fixed)
#   ARM=detbase           deterministic FP4STE weights + plain Muon (FP4-grid ref)
set -e

ARM=${ARM:-qmcfull}
case ${ARM} in
  qmcfull) W_QUANT="QMCSRSTEQuantizer"; W_QUANT_KWARGS='{}';            MUON_SR_MODE="update" ;;
  qmcfwd)  W_QUANT="QMCSRSTEQuantizer"; W_QUANT_KWARGS='{}';            MUON_SR_MODE="none" ;;
  iidfwd)  W_QUANT="QMCSRSTEQuantizer"; W_QUANT_KWARGS='{"qmc": false}'; MUON_SR_MODE="none" ;;
  rtn)     W_QUANT="STEQuantizer";      W_QUANT_KWARGS='{}';            MUON_SR_MODE="none" ;;
  detbase) W_QUANT="FP4STEQuantizer";   W_QUANT_KWARGS='{}';            MUON_SR_MODE="none" ;;
  qmcfp4)  W_QUANT="QMCSRFP4Quantizer"; W_QUANT_KWARGS='{}';            MUON_SR_MODE="none" ;;
  iidfp4)  W_QUANT="QMCSRFP4Quantizer"; W_QUANT_KWARGS='{"qmc": false}'; MUON_SR_MODE="none" ;;
  qmcfp4full) W_QUANT="QMCSRFP4Quantizer"; W_QUANT_KWARGS='{}';         MUON_SR_MODE="update" ;;
  *) echo "unknown ARM=${ARM}"; exit 1 ;;
esac
echo "=== ARM=${ARM}: W_QUANT=${W_QUANT} kwargs=${W_QUANT_KWARGS} muon-sr=${MUON_SR_MODE} ==="

export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512
export DATASET=${DATASET:-slimpajama}   # slimpajama | minipile | redpajama | ...
# Seed study: SEED varies init/training RNG AND the data order (data-seed
# offsets with it), so cross-seed spread measures total run-to-run variance.
# NOTE: the QMC antithetic streams are seeded by (step, pair, instance), NOT by
# torch's seed — across SEEDs the qmc arms share their SR noise sequence; iid
# arms draw from the torch RNG and vary fully.
export SEED=${SEED:-0}

# Model size: 50M (default) or 100M, matching train.sh recipes
MODEL_SIZE=${MODEL_SIZE:-50M}
case ${MODEL_SIZE} in
  50M)
    export N_LAYER=7;  export N_EMBD=768;  export N_HEAD=6; export LR=0.0012 ;;
  100M)
    export N_LAYER=8;  export N_EMBD=1024; export N_HEAD=8; export LR=0.0006 ;;
  *) echo "unknown MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac
# lr sweep support: the per-size LR defaults above came from the AdamW recipes
# in train.sh and are not Muon-tuned
export LR=${LR_OVERRIDE:-$LR}

# Token budget: TOKENS / (BATCH*ACC*SEQ) = iterations
#   mini tier (default): 134217728  -> 512 iters
#   1B tier:             1073741824 -> 4096 iters (needs full rebuilt dataset)
export TOKENS=${TOKENS:-134217728}
export DATASETS_DIR=${DATASETS_DIR:-./datasets/}
NAME_TAG=${NAME_TAG:-mini}

# Preemption/resume policy:
#   strict (default): every start (incl. requeue) is a fresh run — restart count
#                     in the dir name, no checkpoints, auto-resume off.
#   own:              fresh init at job birth (job-id-unique dir, nothing to
#                     load), checkpoints every 100 iters, and a preemption
#                     requeue of the SAME job resumes from its OWN checkpoint.
#                     Foreign/stale weights can never be loaded either way.
RESUME_POLICY=${RESUME_POLICY:-strict}
if [ "${RESUME_POLICY}" = "own" ]; then
  AUTO_RESUME=True
  CKPT_INTERVAL=100
  RUN_SUFFIX="job${SLURM_JOB_ID:-local}"
else
  AUTO_RESUME=False
  CKPT_INTERVAL=100000
  RUN_SUFFIX="job${SLURM_JOB_ID:-local}_r${SLURM_RESTART_COUNT:-0}"
fi
export A_QUANT="STEQuantizer"      # activations deterministic in every arm

export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
# hyperparam-sweep knobs (defaults reproduce all prior runs)
WARMUP_PCT=${WARMUP_PCT:-10}                 # warmup as %% of iterations
export MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
export WD=${WD:-0.1}
export SR_BITS=${SR_BITS:-4}
export WARMUP_STEPS=$((ITERATIONS * WARMUP_PCT / 100))

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export MASTER_PORT=$(
    shuf -i 22521-65535 -n 1 | \
    while read p; do
        if ! ss -tuln | grep -q ":$p "; then echo $p; break; fi
    done
)

torchrun --master_port=${MASTER_PORT} --nproc_per_node=${NUM_GPUS} ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --datasets-dir ${DATASETS_DIR} \
    --eval-batches ${EVAL_BATCHES:-32} \
    --seed ${SEED} \
    --data-seed $((1337 + SEED)) \
    --model llama \
    --latest-ckpt-interval ${CKPT_INTERVAL} \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --opt muon \
    --muon-sr-mode ${MUON_SR_MODE} \
    --muon-sr-bits ${SR_BITS} \
    --muon-momentum ${MUON_MOMENTUM} \
    --weight-decay ${WD} \
    --experiment-name "qmc_${NAME_TAG}_${ARM}_${MODEL_SIZE}_${RUN_SUFFIX}" \
    --auto-resume ${AUTO_RESUME}
# Fresh-pretraining guarantee (both policies): the experiment dir embeds the
# cluster-unique SLURM job id, so the first start of any job finds no directory
# and always initializes fresh weights. Under RESUME_POLICY=strict the restart
# count is appended too and checkpointing is disabled, so even requeues restart
# from scratch. Under RESUME_POLICY=own a requeue of the same job resumes from
# the checkpoint that same job wrote -- never anything else.
