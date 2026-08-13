# QMC-SR + Muon: Program Results

Running results log for the variance-reduced stochastic rounding (SR) program:
antithetic ("QMC") SR quantizers exploiting the averaging windows Muon already
has (grad-accumulation micro-steps, momentum, consecutive-step correlation).

## Standard protocol (unless noted)

- 50M Llama (7L/768d/6H), Muon (lr 1.2e-3, momentum 0.95), eff-batch 512
  (64 x acc 8), seq 512, 4096 opt steps = 1.07B tokens, cosine schedule.
- Dataset `c4slice`: 8/1024 shards of allenai/c4 en, **gpt2 tokenizer** (50304).
  Numbers on `c4llama` (same text, Llama-2 tokenizer) are a DIFFERENT loss
  scale — never compare across tokenizers (see QUEST_PAPER_BASELINES.md).
- Final val_loss at iter 4096, eval 32 batches. Arm-noise sigma ~0.005 inside
  the QuEST pipeline (3-seed); RTN-style arms can be far noisier (see seed study).

## Anchors

| run | val_loss | note |
|---|---|---|
| fp16 (NoQuantizer, c4slice) | 3.574 | job 827911; reference for all fp16-arm ladders |
| questbase W4A4 (c4slice) | 3.625 ± 0.003 | 3 seeds {3.629, 3.623, 3.623}; quantization gap 0.055 ≈ paper's claimed ~0.06 |
| QuEST paper replica (c4llama, AdamW eff512, 4768 steps, 1.25B tok) | 3.294 | vs their 3.292 — replication to 0.002 (job 854842) |
| Muon on their scale (c4llama, 4096 steps, 1.07B tok) | 3.289 | matches their AdamW anchor with ~15% fewer tokens (job 854843) |

## G-quantization: low-precision gradient accumulation (2026-08-09)

`G <- Q(G + g_micro)` per micro-step, NO error feedback (src/optim/gquant.py),
fp16 forward arm, vs fp32-accum ref 3.574. Jobs 862496-99, 870039-44.

| accum bits | det | iid SR | antithetic SR |
|---|---|---|---|
| 8 | 3.582 | 3.588 | 3.583 |
| 6 | 3.627 | 3.652 | **3.624** |
| 4 | 4.053 | 3.831 | **3.790** |

- **Antithetic beats iid at every width, gap monotone in bits**: −0.005 @8b,
  −0.028 @6b, −0.041 @4b. First consistent positive antithetic result of the
  program, exactly in the predicted no-error-feedback accumulation regime.
- **Det collapses at 4 bits** (+0.48 vs ref): classic swamping — small
  micro-grads deterministically rounded away; SR preserves them in expectation.
  Det is fine at 6+ bits.
- Accumulation cost (qmc arm): 8b ~free (+0.01), 6b +0.05, 4b +0.22.
- Single seed per cell.

## Momentum-buffer quantization (2026-08-13)

`buf <- Q(0.95*buf + G)` every step inside Muon (src/muon.py, `mq_*` knobs) —
the same accumulate-without-error-feedback structure, applied to the largest
persistent optimizer state. Fp16 arm vs ref 3.574, jobs 928750-58:

| buffer bits | det | iid SR | antithetic SR |
|---|---|---|---|
| 8 | 3.575 | 3.573 | 3.575 |
| 6 | 3.580 | 3.586 | 3.582 |
| 4 | 3.667 | 3.700 | **3.653** |

- **Antithetic-vs-iid gap replicates the G-quant pattern and is monotone in
  bits**: ~0 @8b, −0.004 @6b, **−0.047 @4b** — the predicted low-bit-leverage
  mechanism (fractional-part correlation survives only when grid indices are
  small) held exactly: antithetic ≈ iid at 8b, separation at 4b.
- **Antithetic beats det at 4 bits too** (3.653 vs 3.667) — best mode overall.
- **Det does NOT collapse** here (3.667 vs G-quant's 4.053 @4b): the fresh
  gradient enters at full precision before each single rounding, so swamping
  is far milder than in the micro-step accumulator.
- **4-bit momentum state is cheap**: +0.079 (qmc) over fp32 — cheaper than
  4-bit G-accumulation (+0.22). 8b/6b are free-to-negligible. Single seed.

### Cross-pipeline transfer (4-bit mq under other forward pipelines)

| forward arm | no mq (seed refs) | + mq4 det | + mq4 qmc |
|---|---|---|---|
| fp16 | 3.574 | 3.667 | 3.653 |
| questbase (QuEST W4A4) | 3.625 ± 0.003 | 3.704 | 3.692 |
| detbase (FP4 det) | 4.211 ± 0.055 (3 seeds) | 4.124 | 4.103 |

Antithetic > det in every pipeline (−0.014 / −0.012 / −0.021); the ~+0.07
4-bit cost transfers to questbase. Under detbase the mq runs land BELOW the
no-mq seed mean (~1.7-2σ) — momentum SR appears to act as a stabilizing
dither on the fragile det-weights arm rather than a cost (single seed each).

## Forward-pipeline three-way: us vs deterministic baselines vs QuEST (c4slice)

Identical tier protocol, W4A4, finals @4096:

| arm | val_loss |
|---|---|
| fp16 reference | 3.574 |
| questbase (QuEST Hadamard+trust) | 3.629 (3-seed mean 3.625 ± 0.003) |
| qmcfull (us: QMC-SR weights + NS-then-round) | 4.080 |
| qmcfwd (QMC-SR weights only) | 4.122 |
| rtn (deterministic, same uniform grid as ours) | 4.142 |
| iidfwd (iid-SR weights) | 4.180 |
| detbase (FP4 deterministic) | 4.274; seeds {4.274, 4.184, 4.175} mean 4.211 ± 0.055 |

Our plain-grid stack beats both deterministic baselines; QuEST's activation
pipeline dominates all uniform-grid-activation arms by ~0.45 — activation-side
design is their edge, which is why the program pivoted to optimizer-state
quantization. Same ordering held on slimpajama. detbase seed verdict
(2026-08-13, jobs 928774/75): moderately noisy (σ 0.055) but NOT
rtn-catastrophic (rtn seeds spanned 3.81-4.93) — the seed-fragility claim
stays specific to the uniform-grid RTN arm. Best-stack attempt (job 928773):
qmcfull + 6-bit NS-round on c4 = **4.335**, WORSE than 4-bit qmcfull (4.080)
and than no NS-round (qmcfwd 4.122) — opposite sign to the 100M slimpajama
result where 6-bit was the fix (3.708); NS-round bit-width sensitivity is
strongly scale/dataset-dependent and plain-grid arm σ~0.1 blurs single-seed
reads. The NS-round lever does not transfer as "6 bits always".

## SR-weights penalty inside the QuEST pipeline (W-bit ladder)

queststyle (= questbase + SR weights) minus questbase, c4slice protocol:

| bits (WxAx) | questbase | queststyle | SR penalty | seeds |
|---|---|---|---|---|
| 4 | 3.625 ± 0.003 | 3.639 ± 0.005 | +0.014 | 3 |
| 3 | 3.700 ± 0.005 | 3.729 ± 0.004 | +0.029 | 3 |
| 2 | 3.876 | 3.936 | +0.060 | 1 |

Penalty is monotone in bit width and ≫ sigma: SR-in-forward-weights HURTS
inside a strong deterministic pipeline, and more so at low bits (the low-bit
hypothesis is dead). Clean, consistent, publishable-as-negative.

## NS-then-round (round the Newton-Schulz output)

- 50M slimpajama tier: biggest single positive effect — qmcfull 3.690 vs
  qmcfwd 3.832 (**−0.142**).
- 100M: 4-bit grid too coarse (+0.05 to +0.19); **sr_bits=6 → 3.708, best 100M
  number of the program** (−0.215 vs 4-bit). Effect survives scale at 6 bits.
- On top of the QuEST pipeline: nothing (queststyle+NS6 3.644 ≈ queststyle 3.645).
- Missing control: det-6-bit NS-round (is the stochastic part necessary?).

## PTQ ladder (one-shot, fp16 ckpt 3.574 → val-config fp16 3.526)

ptq/ptq_ladder.py, GPTQ-style H, QAT-consistent rowwise centered grid:

| bits | RTN | LDLQ | LDLQ-iid | LDLQ-qmc |
|---|---|---|---|---|
| 4 | 3.629 | **3.552** | 3.568 | 3.568 |
| 3 | 3.923 | **3.617** | 3.674 | 3.672 |
| 2 | 5.787 | **4.050** | 4.368 | 4.366 |

Deterministic LDLQ wins at every width; antithetic == iid exactly — error
feedback already sequentially compensates, leaving antithetic nothing to cancel.

## Seed stability (50M slimpajama, plain-grid arms)

qmcfull {3.690, 3.765, 3.892}, iidfwd {4.004, 3.828, 3.935},
rtn {3.812, 4.210, 4.934}: deterministic RTN under quantized Muon is
seed-UNSTABLE (2/3 seeds degrade badly), SR arms stay within ~0.2 — SR
variance-stabilizes quantized Muon training. The QuEST Hadamard+trust pipeline
shows NO such fragility (sigma ~0.003-0.005).

## Grand synthesis

SR's value lives in **temporal accumulation during training** — gradient
accumulation (antithetic wins), seed stability, momentum (pending) — NOT in
one-shot rounding (LDLQ wins), not inside strong forward pipelines (monotone
penalty), and its forward-pass cost grows at low bits. Program ranking:
G-quantization > master-weight/momentum storage > update compression.
