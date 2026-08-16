#!/bin/bash
# Unified seed campaign (PAPER_PLAN.md week 1). ~22 jobs, ~105 GPU-h.
# Launch order: 100M first (longest per-run, most preemption-exposed).
# All runs carry mechanism instrumentation (>Mech lines) for free.
set -e
SUB=qmc_tier1b.sub
C50="ARM=fp16,DATASET=c4slice,TOKENS=1073741824,NAME_TAG=seedcamp,RESUME_POLICY=own"
C100="ARM=fp16,DATASET=slimpajama,MODEL_SIZE=100M,TOKENS=1073741824,NAME_TAG=seedcamp,RESUME_POLICY=own"

# --- 100M: missing strat cells (seed 0) + error bars on {iid, strat} ---
sbatch --job-name=sc_100M_mq4strat_s0 --export=ALL,${C100},MQUANT_MODE=strat,MQUANT_BITS=4,SEED=0 ${SUB}
sbatch --job-name=sc_100M_gq4strat_s0 --export=ALL,${C100},GQUANT_MODE=strat,GQUANT_BITS=4,SEED=0 ${SUB}
sbatch --job-name=sc_100M_gq4iid_s0   --export=ALL,${C100},GQUANT_MODE=iid,GQUANT_BITS=4,SEED=0 ${SUB}
for S in 1 2; do
  sbatch --job-name=sc_100M_mq4strat_s${S} --export=ALL,${C100},MQUANT_MODE=strat,MQUANT_BITS=4,SEED=${S} ${SUB}
  sbatch --job-name=sc_100M_mq4iid_s${S}   --export=ALL,${C100},MQUANT_MODE=iid,MQUANT_BITS=4,SEED=${S} ${SUB}
done

# --- 50M: +2 seeds on headline cells ---
# gq4 {iid,qmc,strat} (det's +0.48 collapse needs no statistical reinforcement,
# but one det run at s1 provides its mech/stall trace for the mechanism figure)
sbatch --job-name=sc_gq4det_s1 --export=ALL,${C50},GQUANT_MODE=det,GQUANT_BITS=4,SEED=1 ${SUB}
for S in 1 2; do
  for M in iid qmc strat; do
    sbatch --job-name=sc_gq4${M}_s${S} --export=ALL,${C50},GQUANT_MODE=${M},GQUANT_BITS=4,SEED=${S} ${SUB}
  done
  for M in det iid qmc strat; do
    sbatch --job-name=sc_mq4${M}_s${S} --export=ALL,${C50},MQUANT_MODE=${M},MQUANT_BITS=4,SEED=${S} ${SUB}
  done
done
echo "submitted: 7x100M + 15x50M = 22 jobs"
