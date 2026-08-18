#!/bin/bash
# Composability campaign (PAPER_PLAN.md week 2, critic's catch): gq4 + mq4
# quantized SIMULTANEOUSLY — additive or superadditive degradation?
# {strat, iid} x seeds {0,1}, 4 jobs ~ 17 GPU-h. Same 50M/fp16/c4slice tier
# as the seed campaign, so single-site anchors (sc_gq4*/sc_mq4*, fp16 3.574)
# are directly comparable. Sites draw from structurally different stream
# families (gquant linear step*100003+idx vs mq XOR-mixed sr_param_seed), so
# cross-site dither is uncorrelated by construction.
set -e
SUB=qmc_tier1b.sub
C50="ARM=fp16,DATASET=c4slice,TOKENS=1073741824,NAME_TAG=compcamp,RESUME_POLICY=own"

for S in 0 1; do
  for M in strat iid; do
    sbatch --job-name=comp_gq4mq4${M}_s${S} \
      --export=ALL,${C50},GQUANT_MODE=${M},GQUANT_BITS=4,MQUANT_MODE=${M},MQUANT_BITS=4,SEED=${S} ${SUB}
  done
done
echo "submitted: 4 composability jobs (gq4+mq4 {strat,iid} x s{0,1})"
