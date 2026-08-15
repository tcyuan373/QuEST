# Paper Plan: Window-Stratified Stochastic Rounding for Low-Precision Training State

Consolidated from a 7-agent mock review panel (theory / empirical / systems reviewers,
area chair, 2 related-work scans with web access, completeness critic), 2026-08-15.

## Verdict

- Submission written today: **borderline-reject at ICLR (4-5)**. Three one-line kills:
  no error-feedback baseline (while our own PTQ table shows EF/LDLQ dominating),
  single-seed headline matrices, Muon-only with simulated quantizers.
- Best case if this plan lands: **solid poster-level ICLR accept, spotlight ceiling**
  (scale tops out at 100M/1B tokens; the math is classical; the contribution is the
  mechanism + rigor package).

## Deadlines (verified against iclr.cc CFP)

- **ICLR 2027: abstract Sep 18, full paper Sep 25, 2026 AoE.** Decisions Dec 16.
- Framing-deciding runs (EF outcome, AdamW transfer, seed campaign) must LAND by
  **~Sep 12** — the abstract freezes the framing. That is 4 working weeks, not 6.
- MLSys 2027 "Oct 30" is UNVERIFIED (no CFP posted; tracker prediction). ICML 2027
  ~late Jan provisional. Re-verify both in September. Do not plan around MLSys.

## Novelty situation (the strategic intel)

1. **The construction is anticipated on the spatial axis**: Suresh et al., *Correlated
   Quantization for Distributed Mean Estimation* (ICML 2022) built exactly the
   antithetic pair AND permutation+jitter (LHS) dither across clients, with variance
   proofs; Panferov et al. (UAI 2025) carried it into distributed optimization. Our
   monotone-LHS guarantee **is** McKay-Beckman-Conover 1979 + Owen 1997.
   → Present the theory as a REDUCTION (the on-grid accumulator freezes the rounding
   thresholds, so the classical LHS theorem applies exactly to the sequentially
   re-quantized window) — that reduction is the genuine, if small, theoretical delta.
   Cite Suresh et al. in abstract-adjacent related work, never claim the construction.
2. **The application is partly occupied**: 4-bit-Muon-GRASP (ICLR 2026), MuonQ
   (COLM 2026), 8-bit Muon states (arXiv:2509.23106). "4-bit Muon momentum works" is
   NOT a contribution. **MuonQ explicitly reports SR hurts Muon-state quantization —
   with iid SR.** Our matrix reproduces that (iid 3.700 > det 3.667) and shows
   correlated SR flips it (strat/qmc 3.65x < det). Frame as *resolving their negative
   result*: the problem was iid dither, not SR.
3. **What no prior work does**: correlate the dither along the TIME axis of a
   sequentially re-quantized accumulator / persistently quantized momentum EMA (all
   prior correlated-quantization work aggregates exactly at a server), plus the
   LLM-pretraining evidence, det-collapse/swamping phenomenology, and RTN
   seed-instability findings. This is a MECHANISM paper, not a systems paper.

## Headline claim (area chair's accept-probability-maximizing version)

> When low-precision training state is accumulated without error feedback — 4-bit
> gradient accumulators and momentum buffers — iid stochastic rounding is the wrong
> default: stratifying the dither across the accumulation window recovers ~half the
> 4-bit degradation at zero memory and zero throughput cost, with a provable variance
> guarantee at the fixed-grid site.

Story arc: (1) det collapses under 4-bit accumulation (+0.48 swamping figure) → SR is
necessary; (2) iid SR wastes variance → window stratification, forced-form lemma, LHS
guarantee; (3) robustness matrix (5 constructions × 3 pipelines × 2 scales); (4) EF
comparison as the memory-Pareto argument; (5) scoping negatives (EF/LDLQ one-shot,
forward-SR penalty) as *features of the theory*. Intra-cluster ordering (qmc/lattice/
strat, 0.004-0.013) stated as unresolved at the noise floor BEFORE a reviewer says it.

Title candidates: "Stratify the Dither: Variance-Reduced Stochastic Rounding for
Low-Precision Training State" / "IID Rounding Is the Wrong Default" / "Low-Precision
Accumulation Without Error Feedback".

## TO-DO — sequenced (dedup'd budget ≈ 250-300 GPU-h vs 300-500 available; keep ≥20% as preemption margin)

### Week 0 (Aug 15-19) — gates everything; NO headline GPU runs before these land
- [ ] **Merge the resume fix** (`fix/muon-sr-step-resume`, already built+verified) —
      without it the paired-seed campaign is corrupted by the first preemption.
      [USER DECISION — behavior change on resume path]
- [ ] **Fix mq iid global-RNG asymmetry** (muon.py `torch.rand_like` → dedicated
      Generator): the iid baseline is the only arm on a different RNG discipline —
      a confound in the paper's ONE resolved claim. Fold verification into the seed
      campaign. (0.5 day)
- [ ] **Instrumentation**: wire the existing-but-unlogged saturation counters plus
      swamping/stall-rate (= Topollai-Choromanska "staleness probability" — adopt
      their metric) and window-error-vs-fp32-shadow variance into logs. Mechanism
      data then rides FREE on every later run. (1-2 days)
- [ ] **Start the theorem package** (GPU-free, ~4-6 person-days total):
      L1 unbiasedness+uniqueness (E[floor(t+u)]=t ∀t iff u~U[0,1));
      L2 frozen-threshold lemma (on-grid accumulator ⇒ window error = additive sum of
      centered Bernoullis with dither-independent thresholds) — the crux;
      P1 Var ≤ iid for negatively-associated dither with uniform marginals (LHS via
      Joag-Dev–Proschan NA; antithetic as 2-point case; cite MBC 1979/Owen 1997);
      P2 closed-form window-error variance per mode (gives PREDICTIONS for the
      mechanism figure); N1 lattice has NO Var≤iid guarantee (2-line counterexample —
      converts the awkward lattice losses into theory-consistent facts).
- [ ] mq-site bias check (theory reviewer's sharpest catch): at the adaptive mq site,
      thresholds depend on previous draws, so unbiasedness itself is unproven for
      correlated modes. Log the buffer-error MEAN trace (rides on instrumentation);
      if ~0, claim "empirically unbiased"; if not, reframe the mq claim. (+half-day
      theory note on conditional unbiasedness at window boundaries)

### Week 1 (Aug 20-27) — the two long poles
- [ ] **Unified seed campaign** (~85 GPU-h): +2 seeds on gq4 {det,iid,qmc,strat} and
      mq4 {det,iid,qmc,strat} at 50M/fp16 (16 runs), +100M mq4/gq4 {iid,strat} incl.
      the missing 100M strat cells (longest runs, most preemption-exposed — launch
      FIRST). Report mean±σ; demote all sub-0.012 orderings to "tied".
      Do NOT seed lattice/vdc/latperm — declare the cluster and move on.
- [ ] **Implement EF with unit tests** (highest-risk code on the path; a buggy EF
      baseline that "loses" is a rebuttal landmine — extend the test_*.py culture).
      TWO variants only (panel proposed five; critic dedup'd): fp32-residual
      (quality upper bound) and one memory-honest variant (fp16/Kahan, cite
      Zamirai 2020 which makes EF's absence indefensible).

### Week 2 (Aug 28 - Sep 4) — framing deciders
- [ ] **EF runs** (~36 GPU-h): both sites × both variants × 2 seeds. Either outcome
      survives: EF wins → we're the memory-Pareto point; EF ties/loses → stronger.
      Also note (1 line): with exact EF, dither correlation provably cancels —
      unifies the LDLQ antithetic==iid PTQ result with the training story.
- [ ] **AdamW first-moment 4-bit** (~27 GPU-h): exp_avg ← Q(β₁m + (1-β₁)g), same
      no-EF structure; {det, iid, strat} × 2 seeds. Decides "optimizer state" vs
      "Muon" in the title. **Do NOT touch exp_avg_sq** (nonneg+sqrt'd, fragile,
      different paper — panel + critic unanimous).
- [ ] **Composability** (critic's catch nobody else made): gq4+mq4 quantized
      SIMULTANEOUSLY (strat + iid control, ~9-18 GPU-h, zero implementation) —
      additive or superadditive degradation? Natural headline demo of a
      "4-bit-state trainer".

### Week 3 (Sep 5-11) — robustness + rebuttal insurance
- [ ] Blockwise-absmax-128 det baseline at mq4 + strat ON TOP of the blockwise grid
      (~18 GPU-h): the honest MuonQ proxy ("orthogonal lever: randomness vs grids");
      full MuonQ/GRASP reproduction is infeasible at 50M and NOT required.
- [ ] Window sweep gq4 {iid,strat} at acc_steps {4,16} — **with the critic's confound
      fix**: fixed eff-batch confounds m with micro-batch SNR; add the fixed-micro-batch
      dual (eff-batch varies) or word the interpretation honestly (~27-45 GPU-h).
- [ ] Momentum-coefficient spot-check mq4 {iid,strat} at β=0.9 (appendix tier).
- [ ] LR-fairness nudge (rebuttal insurance): mq4-det/iid at lr×{0.75,1.25}
      (~10-14 GPU-h) — preempts "your baseline would close the gap if retuned".
- [ ] Dither-seed replicates at fixed data/init seed, gq4-strat ×3 (~14 GPU-h):
      the 0.008 noise floor is currently an n≈2 estimate used as the yardstick.
- [ ] Headroom check on the det collapse (gq4 det+strat at headroom {0.5, 2.0},
      ~18 GPU-h): the motivating +0.48 figure rests on one untested scale policy.
      Log saturation BY MICRO INDEX.
- [ ] bf16 momentum-buffer row (1 run): completes the bytes-vs-loss Pareto with the
      industry default.
- [ ] Eval-only jobs (~5 GPU-h): training-curve figure (when does the gap emerge —
      wandb data exists), cross-dataset val, one hellaswag row via src/eval_hswag.py.

### Sep 12-25 — writing only (GPUs = preemption reruns)
- [ ] Memory-accounting table with the critic's honesty items: Muon partition
      excludes wte/lm_head — at 50M the tied embedding (~38.6M/60M params) dominates,
      so experiments quantize a MINORITY of optimizer state; give the
      partition-fraction column and the 7B extrapolation where the claim is real.
      Explicit "quantization is simulated" statement + gquant deployment scenario
      (single-device/local accumulation; world_size==1 assert acknowledged).
- [ ] Scope triage: MAIN = both matrices, cross-pipeline+100M transfer, ordering
      controls, theory, EF section, mechanism figure, condensed negatives.
      APPENDIX = forward-SR ladder detail, PTQ/LDLQ, seed stability, iter_dt table.
      CUT = NS-then-round (and its det-6b control becomes moot).
- [ ] Log/reconcile orphan runs (100M qbase 3.245 / qstyle 3.261; curv 4.151 /
      curvinv 3.922) — artifact-evaluation liability.
- [ ] must_cite list from both relwork scans is in the panel output (Suresh 2022,
      MuonQ, GRASP, MBC/Stein/Owen, Zamirai, Wang 2018, Gupta 2015, Croci survey,
      SOLO, staleness paper, ECO, EF21, dither classics Schuchman/Gray-Stockham/
      Zamir-Feder, Anti-PGD/GraB, DRIVE/EDEN, ...).

## DO-NOT-DO before deadline (unanimous / critic-confirmed)

- Sighted-dither or coordinate-axis QMC research (half-baked new research)
- Real packed-int4 kernels (serves only unverified-MLSys; engineering days are the
  scarcest resource — save for resubmission)
- Scaling past 100M / new model families / 200M "leftover budget" runs (margin IS
  the leftover budget)
- More seeds on lattice-vs-vdc-vs-latperm fine ordering (unresolvable at 0.008)
- AdamW second moment; NS-round rehabilitation; more forward-arm runs vs QuEST
