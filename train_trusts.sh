#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set common environment variables
export VOCAB_SIZE=32000 # 50304
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512
export DATASET="slimpajama" # "slimpajama"

# # 30M
# export N_LAYER=6
# export N_EMBD=640
# export N_HEAD=5
# export LR=0.0012
# export TOKENS=3000000000 # 3B
# export MODEL_SIZE_PREFIX="30M"

# 50M
export N_LAYER=7
export N_EMBD=768
export N_HEAD=6
export LR=0.0012
export TOKENS=5000000000 # 5B
export MODEL_SIZE_PREFIX="50M"

# Quantization configuration
export W_QUANT="HadamardTrustQuantizer"
export A_QUANT="HadamardTrustQuantizer"
export BITS=1

# Calculate the number of iterations based on tokens and batch settings
export ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))
export WARMUP_STEPS=$((ITERATIONS / 10))

CLIP_SCALE_VALUES=(1.05 1.05 1.15 1.35 1.45 1.50)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Loop through trust values
for TRUST in "${CLIP_SCALE_VALUES[@]}"; do
    echo "Running with TRUST=${TRUST}"

    # Update quantization kwargs with current trust value
    export W_QUANT_KWARGS="{\"bits\": ${BITS}, \"trust\": ${TRUST}}"
    export A_QUANT_KWARGS="{\"bits\": ${BITS}, \"trust\": ${TRUST}}"

    WANDB_PREFIX="UNTIED-${MODEL_SIZE_PREFIX}-${W_QUANT}@${BITS}:${A_QUANT}@${BITS}-${DATASET}-TRUST-${TRUST}"

    torchrun --nproc_per_node=${NUM_GPUS} ./src/main.py \
        --distributed-backend nccl \
        --dataset ${DATASET} \
        --model llama \
        --compile \
        --latest-ckpt-interval 10000 \
        --acc-steps ${ACC_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --wandb \
        --wandb-project "llm-baselines" \
        --wandb-run-prefix "${WANDB_PREFIX}" \
        --n-layer ${N_LAYER} \
        --n-embd ${N_EMBD} \
        --n-head ${N_HEAD} \
        --warmup-steps ${WARMUP_STEPS} \
        --iterations ${ITERATIONS} \
        --lr ${LR} \
        --w-quant ${W_QUANT} \
        --w-quant-kwargs "${W_QUANT_KWARGS}" \
        --a-quant ${A_QUANT} \
        --a-quant-kwargs "${A_QUANT_KWARGS}"
done
