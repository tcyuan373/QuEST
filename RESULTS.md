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

## Momentum-buffer quantization (launched 2026-08-13, pending)

`buf <- Q(0.95*buf + G)` every step inside Muon (src/muon.py, `mq_*` knobs) —
the same accumulate-without-error-feedback structure, applied to the largest
persistent optimizer state. Ladder {det,iid,qmc} x {8,6,4} bits, fp16 arm,
jobs 928750-58 (gated on smoke 928749). Mechanism note: consecutive pre-round
values differ by momentum-scaling of the grid indices, which scrambles
fractional parts at high bits — antithetic leverage is expected mainly at low
bits, so `antithetic ≈ iid at 8b, separation at 4b` would be coherent.

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
| detbase (FP4 deterministic) | 4.274 (1 seed; seed pair pending, jobs 928774/75) |

Our plain-grid stack beats both deterministic baselines; QuEST's activation
pipeline dominates all uniform-grid-activation arms by ~0.45 — activation-side
design is their edge, which is why the program pivoted to optimizer-state
quantization. Same ordering held on slimpajama. Pending additions (2026-08-13):
qmcfull + 6-bit NS-round on c4 (job 928773, best-stack row); momentum-quant
4-bit det/qmc composed under detbase and questbase pipelines (jobs 928769-72,
does the mq verdict transfer beyond the fp16 arm?).

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
