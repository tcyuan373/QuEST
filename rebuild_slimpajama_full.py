"""Rebuild the FULL SlimPajama-6B train/val bins into ./datasets_full/slimpajama6B/.

The original datasets/slimpajama6B/train.bin was left 95.6% zero-padded by an
interrupted tokenization job (only a 216M-token real prefix; see memory notes
2026-07-30). The truncated real prefix currently serves the mini runs; this
script regenerates the complete ~5.8B-token dataset into a SEPARATE directory
so running jobs are never disturbed. Exits nonzero unless the result passes
verification, so SLURM afterok dependencies can gate on it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from data.slimpajama import get_slimpajama_data

DATASETS_DIR = "./datasets_full/"

get_slimpajama_data(DATASETS_DIR, num_proc=40)

# ---- verification: fail loudly rather than leave another silent bad bin ----
path = os.path.join(DATASETS_DIR, "slimpajama6B", "train.bin")
data = np.memmap(path, dtype=np.uint16, mode="r")
n = len(data)
rng = np.random.default_rng(0)
starts = rng.integers(0, n - 513, 2000)
zero_frac = float(np.mean([(np.asarray(data[s : s + 512]) == 0).mean() for s in starts]))
print(f"train.bin tokens: {n:,}; sampled zero-token fraction: {zero_frac:.4%}")

assert n > 5_000_000_000, f"train.bin too small: {n:,} tokens"
assert zero_frac < 0.01, f"train.bin looks zero-padded again: {zero_frac:.2%} zeros"
print("VERIFICATION PASSED")
