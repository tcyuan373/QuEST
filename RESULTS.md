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

| accum bits | det | iid SR | antithetic SR | lattice SR | vdc SR |
|---|---|---|---|---|---|
| 8 | 3.582 | 3.588 | 3.583 | 3.583 | — |
| 6 | 3.627 | 3.652 | **3.624** | 3.630 | — |
| 4 | 4.053 | 3.831 | **3.790** | 3.794 | 3.802 |

(lattice/vdc columns from jobs 982076-78, 982089; see "Loyal randomized-QMC" below.)

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

| buffer bits | det | iid SR | antithetic SR | lattice SR |
|---|---|---|---|---|
| 8 | 3.575 | 3.573 | 3.575 | 3.574 |
| 6 | 3.580 | 3.586 | 3.582 | 3.580 |
| 4 | 3.667 | 3.700 | **3.653** | 3.662 |

(lattice column from jobs 982079/80/90.)

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

| forward arm | no mq (seed refs) | + mq4 det | + mq4 iid | + mq4 qmc | + mq4 lattice |
|---|---|---|---|---|---|
| fp16 | 3.574 | 3.667 | 3.700 | **3.653** | 3.662 |
| questbase (QuEST W4A4) | 3.625 ± 0.003 | 3.704 | 3.744 | **3.692** | 3.705 |
| detbase (FP4 det) | 4.211 ± 0.055 (3 seeds) | 4.124 | 4.180 | **4.103** | 4.143 |

Antithetic > det in every pipeline (−0.014 / −0.012 / −0.021); the ~+0.07
4-bit cost transfers to questbase. Under detbase the mq runs land BELOW the
no-mq seed mean (~1.7-2σ) — momentum SR appears to act as a stabilizing
dither on the fragile det-weights arm rather than a cost (single seed each).
iid/lattice columns added 2026-08-14 (jobs 982081/82, 982091/92): the mode
ordering **qmc < lattice < det < iid is identical in all three pipelines** —
the strongest structural result of the momentum-quant study, since the three
forward arms span a 0.6-loss range.

### Scale transfer to 100M (slimpajama, jobs 298-302)

| arm | fp16 | mq4 det | mq4 iid | mq4 qmc | mq4 lattice |
|---|---|---|---|---|---|
| val_loss | 3.196 | 3.303 | 3.321 | **3.275** | 3.292 |

Same ordering at 2x model scale and on a different dataset, with the same
spread (qmc−iid −0.046 @100M vs −0.047 @50M) — the antithetic momentum gain
is scale- and data-stable, unlike the NS-round lever.

## Loyal randomized-QMC: does higher-order beat the 2-point pair? (2026-08-14)

Antithetic SR is the crudest possible QMC: a 2-point balanced pair over the
averaging window. Commit 421dd60 added properly loyal randomized-QMC
constructions — **shifted rank-1 lattice** (`mode=lattice`, whole window is one
randomly-shifted lattice rather than 2 antipodal draws) and **van der Corput**
(`mode=vdc`, radical-inverse window points, gquant only) — to test whether more
QMC structure buys more variance reduction. Jobs 982074-92, fp16 arm.

Head-to-head vs antithetic (positive = lattice/vdc is WORSE):

| site | bits | antithetic | lattice | Δ | iid | lattice vs iid |
|---|---|---|---|---|---|---|
| gquant | 8 | 3.583 | 3.583 | 0.000 | 3.588 | −0.005 |
| gquant | 6 | 3.624 | 3.630 | +0.006 | 3.652 | −0.022 |
| gquant | 4 | 3.790 | 3.794 | +0.004 | 3.831 | −0.037 |
| mquant | 8 | 3.575 | 3.574 | −0.001 | 3.573 | +0.001 |
| mquant | 6 | 3.582 | 3.580 | −0.002 | 3.586 | −0.006 |
| mquant | 4 | 3.653 | 3.662 | +0.009 | 3.700 | −0.038 |
| mq4 @ questbase | 4 | 3.692 | 3.705 | +0.013 | 3.744 | −0.039 |
| mq4 @ detbase | 4 | 4.103 | 4.143 | +0.040 | 4.180 | −0.037 |
| mq4 @ 100M | 4 | 3.275 | 3.292 | +0.017 | 3.321 | −0.029 |

van der Corput (gquant 4b only): **3.802** — worse than lattice (3.794) and
antithetic (3.790), better than iid (3.831).

- **Higher-order QMC does not beat the 2-point pair** — antithetic ties or
  wins in all 9 cells, sign-consistent, and the 4-bit deficit grows with
  pipeline fragility (+0.004 to +0.040). But see the nuisance caveat below:
  the per-cell gaps are not individually resolved.
- **Lattice/vdc recover ~80-95% of the iid→antithetic gain** (e.g. mq4:
  iid→lattice −0.038 of the −0.047 total). Three structurally different
  randomizations all land in the same place, which is the real evidence that
  the effect is genuine window-level variance reduction and not a lucky
  antithetic artifact. This is the robust finding of the study.
- **Ordering-nuisance caveat (2026-08-14).** lattice and vdc share the SAME
  base shift and point set (identical seed in gquant.py `_u`) and differ ONLY
  in which point goes to which micro-step; that pure reassignment moved gq4
  by 0.008 (3.794 vs 3.802) — 2x the antithetic-vs-lattice gap (0.004). The
  fine ordering among SR variants at 4 bits is inside assignment/seed noise;
  only iid-vs-everything-else (~0.03-0.05) is resolved. Controls queued:
  `latperm` (same lattice points, random per-coordinate assignment — isolates
  the nuisance directly) and `strat` (Latin-hypercube jitter, the only mode
  with a per-window Var<=iid guarantee for arbitrary integrands).
- **Mechanism claim RETRACTED (2026-08-14).** The earlier "the integrand is
  near-linear in the fractional part, so the antipodal pair is optimal" story
  is wrong: f(u) = floor(t+u) − t is a step function (V(f)=1), and
  Koksma-Hlawka with shifted-lattice star discrepancy 1/m predicts lattice
  should be at LEAST as good as the pair. The measured reversal therefore
  contradicts the natural theory instead of confirming a mechanism — one more
  reason to treat the 0.004-0.017 antithetic edges as unresolved. (The unit
  test's drift-free correlated window shows lattice 4x BETTER than antithetic
  in window MSE, per theory; whatever reverses this in training is not
  explained.)
- Cost: no measurable throughput difference (iter_dt 3.62-3.78s across all
  modes). Single seed per cell; the 4-bit gaps (0.004-0.040) sit at or below
  fp16-arm σ except at detbase/100M.

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
accumulation (antithetic wins), seed stability, momentum (antithetic wins,
ordering invariant across 3 forward pipelines and 2 model scales) — NOT in
one-shot rounding (LDLQ wins), not inside strong forward pipelines (monotone
penalty), and its forward-pass cost grows at low bits. Program ranking:
G-quantization > master-weight/momentum storage > update compression.
Within the family of **blind time-axis point sets** the construction looks
saturated: antithetic/lattice/vdc all land within ~0.01 of each other (their
fine ordering unresolved at single-seed, single-assignment noise) while all
recover most of the iid gap. Pending the latperm/strat controls, remaining
headroom is more likely in WHERE the dither acts than in fancier blind
sequences — the coordinate axis (m ~ 1e6 vs the time axis's m = 8, and NS
mixes within rows) and "sighted" dither (frac(t) is known before u is drawn;
every current mode ignores it) are the unexplored axes.
