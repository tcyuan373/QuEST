# QuEST paper baselines (from `for_plots/runs_for_scaling.json`, 104 runs)

Original IST-DASLab runs behind the paper's scaling curves.

**CORRECTION-OF-CORRECTION (2026-08-06): their C4 losses are on the LLAMA-2 token
scale, NOT gpt2.** The `tokenizer: "gpt2"` / `vocab_size: 50304` in every run's
`args` is a dead config field — `args.tokenizer` is consumed nowhere in `src/`;
`dataset: "c4"` dispatches to `get_c4_data` (`src/data/c4.py`), which has hardcoded
`AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")` (vocab 32000) since the
original upstream commit (e02f63a). The 2026-08-04 note above-said "same scale" was
wrong. Llama-2 packs fewer chars/token than gpt2, so its per-token CE is
systematically LOWER; their absolute losses are NOT comparable to any of our
gpt2-tokenized runs without a bits-per-byte conversion.

**Also corrected: their effective batch is 512, not 128.** The 1.25B anchor ran
`world_size: 4` × batch 64 × acc 2 = eff-batch 512, `iterations: 4768` — i.e. their
protocol nearly matches our original b512 tier setup (we take 4096 steps @1.07B).

At 50M the architecture and schedule match ours exactly (llama 7L/768d/6H, seq 512,
lr 1.2e-3, wd 0.1, cosine, warmup 10%, grad_clip 1, eval_batches 32). They run
`--compile`, we don't. Quantizer default `bits=4` matches their explicit kwargs.

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
| dataset / val | C4 (Llama-2 tok, 32k vocab — see correction above) | SlimPajama/MiniPile/RPJ/C4-slice (gpt2, 50304); c4llama (Llama-2, 32k) |
| optimizer | AdamW (lr 1.2e-3 @50M, 6e-4 @100M) | Muon |
| tokens | 100/param (5B @50M, 10B @100M) | 1.07B (tier) |
| forward quantizer | Hadamard + trust STE | plain STE grid ± (QMC-)SR |

Their token-matched anchor (Llama-2 token scale): 50M W4A4 HalfHadamardTrust @1.25B
tokens = **3.292** (C4, AdamW, effective batch 512, 4768 opt steps).

**Apples-to-apples bridge (2026-08-05, jobs 731916/17, c4slice @1.07B, Muon,
effective batch 512, 4096 opt steps): `questbase` 3.629, `queststyle` 3.645.**
queststyle−questbase = +0.016 on C4 (+0.015 on slimpajama): SR weights inside
their pipeline are a wash, replicated across datasets.

**EXACT REPLICATION (2026-08-08, job 854842): `c4llama` dataset (same 8 C4 shards,
Llama-2 tokenizer via the ungated hf-internal-testing mirror, their tokenization
semantics) + their protocol (AdamW, eff-batch 512, 4768 steps, 1.25B tokens) →
final val loss 3.294 vs their anchor 3.292.** The entire apparent gap was the
tokenizer scale; harness, data slice, and quantizer pipeline reproduce the paper
essentially exactly. Bonus (854843): questbase + Muon @1.07B on c4llama = 3.289 —
Muon marginally beats their AdamW anchor with 15% fewer tokens.

**Protocol-gap decomposition (2026-08-06, jobs 806891-93, questbase/c4slice/50M):
b128-muon 3.622 (16384 steps @1.07B) | b128-adamw "paper replica" 3.626 (19073
steps @1.25B) | b512-adamw 3.660 | b512-muon 3.629 (ref).** All four within 0.04:
batch size, optimizer, step count, and +17% tokens each explain ~nothing. The
resolution of the apparent 0.33 gap to their 3.292 anchor is the TOKENIZER-scale
correction above — their per-token losses (Llama-2 tokens) are systematically lower
than gpt2-token losses on the same text; a bits-per-byte conversion is required
before comparing. (Ironically their true protocol — eff-batch 512, ~4.8k steps —
was nearly identical to our tier setup all along.) The earlier slimpajama
questbase 3.309 ≈ 3.292 "match" remains coincidence: different dataset AND
different token scale.
