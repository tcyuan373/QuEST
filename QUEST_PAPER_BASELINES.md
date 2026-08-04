# QuEST paper baselines (from `for_plots/runs_for_scaling.json`, 104 runs)

Original IST-DASLab runs behind the paper's scaling curves. **Their setup — NOT directly
comparable to our QMC-SR runs**: dataset **C4** with the **Llama-2 32k tokenizer**
(different vocab ⇒ loss values live on a different scale than our gpt2/50304 runs),
optimizer **AdamW**, ~100 tokens/param budgets (5B @50M, 10B @100M vs our 1.07B),
their C4 validation split. Use as anchors for their method family, not as rows in our tables.

## Headline table — final C4 val loss (full token budget)

| Config (W/A) | 30M | 50M | 100M | 200M | 430M | 800M |
|---|---|---|---|---|---|---|
| FP16 baseline (NoQuantizer) | 3.207 | 3.068 | 2.886 | 2.729 | 2.571 | 2.461 |
| **QuEST W4A4** (HadamardTrust) | 3.272 | **3.135** | **2.948** | 2.782 | 2.614 | 2.495 |
| W4A4 Trust (no Hadamard) | 3.304 | 3.166 | 2.978 | 2.809 | 2.638 | 2.517 |
| **W4A4 FP4** (HalfHadamardFP4Trust)¹ | 3.296 | 3.156 | 2.966 | 2.798 | — | — |
| W4A4 4:8 sparse (HadamardFourEightTrust) | 3.379² | 3.237 | 3.052 | 2.891 | 2.713 | 2.589² |
| W8A8 (HalfHadamardTrust) | 3.209 | 3.069 | 2.888 | 2.743³ | — | — |
| W3A3 (HadamardTrust) | 3.372 | 3.226 | 3.037 | 2.861 | 2.678 | — |
| W2A2 (HadamardTrust) | 3.574 | 3.441 | 3.236 | 3.046 | — | — |
| W1A1 (HadamardClip) | 3.945 | 3.791 | 3.601 | 3.423 | 3.224 | 3.079 |
| W4A16 weight-only (HalfHadamardTrust) | 3.247 | 3.105 | 2.916 | 2.756 | — | — |

¹ FP4 rows are logged with `w_bits=16` in the json (the FP4 classes take no `bits` arg) but are 4-bit E2M1.
² HalfHadamard variant. ³ plain Trust (no Half-Hadamard 8-bit run at 200M).

## Notable paper-internal orderings (their setting)
- **INT4-grid Hadamard-Trust beats FP4-Trust at every size** (e.g. 50M: 3.135 vs 3.156;
  100M: 2.948 vs 2.966) — opposite of our early detbase-vs-rtn reads, but their FP4 also
  carries the Hadamard+trust machinery, so grids are compared like-for-like there.
- Full Hadamard > Half-Hadamard > no Hadamard, consistently.
- Their W4A4 gap to FP16 at full budget: ~0.06–0.07 loss (≈6–7% ppl) at every scale.
- Reduced-budget HalfHadamardTrust rows (0.25×/0.5× tokens) exist in the json for
  token-scaling curves.

## Provenance / comparability checklist vs our QMC-SR experiments
| Axis | Paper | Ours |
|---|---|---|
| dataset / val | C4 (Llama-2 tok, 32k vocab) | SlimPajama/MiniPile/RPJ/C4-slice (gpt2, 50304) |
| optimizer | AdamW (lr 1.2e-3 @50M, 6e-4 @100M) | Muon |
| tokens | 100/param (5B @50M, 10B @100M) | 1.07B (tier) |
| forward quantizer | Hadamard + trust STE | plain STE grid ± (QMC-)SR |

The only apples-to-apples statement available without reruns: **relative** gaps within
each family. A direct comparison would require running their quantizers under our
protocol (arms already exist in `QUANTIZER_CLASSES`) or ours under theirs.
