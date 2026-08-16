"""Simulated low-precision gradient accumulation ("G-quantization").

Each micro-step's fresh gradient is folded into a quantized accumulator:

    G <- Q_s(G + g_micro)

There is NO error feedback -- each rounding error persists in G -- which is
exactly the regime where antithetic pairing across micro-steps can cancel what
sequential compensation (LDLQ-style) would otherwise absorb.

Grid: symmetric uniform INT grid WITH a zero level (gradients need exact zero),
per-row absmax scale sized for the whole accumulation window:

    step = headroom * acc_steps * absmax_row(g_0) / qmax,   qmax = 2^(bits-1) - 1

frozen at micro 0 of each optimizer step, so (a) G stays exactly on-grid
between adds -- each add's rounding error is that of g_micro alone -- and
(b) both members of an antithetic pair share one grid. Values are clamped to
[-qmax*step, qmax*step]; saturated coordinates are counted in .saturated.

Modes:
  det -- round-to-nearest,
  iid -- stochastic rounding, independent u every micro-step,
  qmc -- antithetic SR: micro pairs (0,1),(2,3),... share u vs 1-u per
         coordinate, reconstructed from a Generator seeded by
         (opt step, pair id, param index) -- stateless, resume-safe.
  lattice/vdc/latperm/strat -- window-level constructions over the m
         micro-steps; per coordinate the window's m draws are
           lattice: frac(u + i/m), natural order i = micro
           vdc:     frac(u + i/m), bit-reversed order (m = 2^k only)
           latperm: frac(u + pi(micro)/m), pi a fresh uniform random
                    permutation per (step, param, coord) -- SAME point set
                    and SAME base shift u as lattice (identical seed), so
                    lattice-vs-latperm isolates the point->micro-step
                    assignment, the nuisance that moved gq4 by 0.008
                    between lattice and vdc.
           strat:   (pi(micro) + xi_micro)/m, same pi stream as latperm
                    but an INDEPENDENT within-stratum jitter xi per draw
                    (Latin-hypercube along the window axis). Guarantees:
                    for ARBITRARY fixed integrands only Var <= (m/(m-1)) *
                    Var_iid (Owen; the 8/7 bound is attained by anti-aligned
                    stratum profiles). Var <= Var_iid additionally holds
                    whenever the window's integrands are monotone in u with
                    similarly-ordered stratum profiles (Chebyshev sum
                    inequality) -- true of every realizable window AT THIS
                    SITE, since the on-grid accumulator fixes frac(t_i) =
                    frac(g_i/s) and each error floor(theta+u)-theta is
                    nondecreasing in u. So strat is the safe control here;
                    the same claim does NOT formally extend to the adaptive
                    mq window in muon.py.
         All four share the Cranley-Patterson property that every single
         draw is marginally U[0,1), so each rounding is unbiased in
         isolation; draws are dependent ACROSS micro-steps, so per-step
         conditional-on-past uniformity holds only for iid.

Index-clamp bounds for THIS grid (q = round-or-floor of t = x/step) are
[-qmax, qmax]; validated against torch.round in test_gquant.py (the centered
no-zero QuEST grid has DIFFERENT bounds -- see the twice-bitten clamp bug).

DDP: accumulators are per-rank and .grad is consumed every micro-step, which
bypasses DDP's gradient reduction -- requires world_size 1 (asserted in main).
"""

import torch


class GradAccumQuantizer:
    def __init__(self, model, mode, bits=8, headroom=1.0, acc_steps=8, ef="none"):
        assert mode in ("det", "iid", "qmc", "lattice", "vdc", "latperm", "strat")
        assert ef in ("none", "fp32", "fp16"), ef
        # Error feedback: a persistent residual r is folded into every
        # micro-add and updated to the exact pre-quantization remainder,
        # r <- (G_prev + g + r) - Q(G_prev + g + r). Carries ACROSS windows
        # (standard EF semantics; the LDLQ-style sequential compensation the
        # PTQ ladder showed dominating one-shot). Costs 4 (fp32) or 2 (fp16)
        # bytes/param of extra state -- the memory-honesty comparison the
        # paper's EF-vs-stratified-SR Pareto table reports. CAVEATS: (a) the
        # clamp breaks the compressor-contraction property, so under
        # persistent saturation the residual backlog grows without bound and
        # saturation becomes self-sustaining -- the resid norms are logged in
        # mech_summary() precisely so runs are interpretable in that regime;
        # (b) the residual is NOT checkpointed, so a preemption resume drops
        # the backlog -- an O(|r|) transient, which under saturation can be
        # many windows of mass, not one.
        self.ef = ef
        if mode == "vdc":
            # bit-reversal ordering needs a power-of-two window
            assert acc_steps & (acc_steps - 1) == 0, acc_steps
        if mode == "strat":
            # jitter seed indexes (step*64 + micro); larger windows would alias
            assert acc_steps <= 64, acc_steps
        self.mode = mode
        self.bits = int(bits)
        self.qmax = 2 ** (self.bits - 1) - 1
        self.headroom = float(headroom)
        self.acc_steps = int(acc_steps)
        # Muon partition: >=2D, excluding the (un)embedding matrices
        self.params = [
            (name, p)
            for name, p in model.named_parameters()
            if p.ndim >= 2
            and not name.endswith(("wte.weight", "lm_head.weight"))
        ]
        self.accum = [None] * len(self.params)
        self.step_size = [None] * len(self.params)
        self.saturated = 0  # coords clamped in the current optimizer step
        # mechanism instrumentation (observation only, no effect on training):
        # fp32 shadow of the exact accumulation for window-error stats, plus
        # per-micro stall (swamping/round-back) counting. Shadow costs one
        # fp32 tensor per param during the window; freed at write_back.
        # Stats aggregate in float64 device tensors across windows and are
        # synced+reset only in mech_summary() (call at log intervals).
        self.shadow = [None] * len(self.params)
        self.resid = [None] * len(self.params)  # EF residual (persistent)
        self._stall_sum = None  # device tensor, per-window, lazy init
        self._stall_n = 0
        self._mech_acc = None  # cross-window device accumulators
        self._sat_acc = 0

    def _row_step(self, g):
        absmax = g.abs().flatten(1).amax(dim=1).clamp_min(1e-12)
        span = self.headroom * self.acc_steps * absmax
        return (span / self.qmax).view(-1, *([1] * (g.ndim - 1)))

    def _u(self, step, micro, idx, shape, device):
        if self.mode in ("lattice", "vdc", "latperm", "strat"):
            # Window-level constructions (see module docstring). Seed layout:
            #   ((step*100003 + idx)*8 + 5) -- base shift u, SHARED by
            #       lattice/vdc/latperm so they use the identical point set;
            #   ((step*100003 + idx)*8 + 6) -- permutation keys R, shared by
            #       latperm/strat (same pi, differing only in the offset law);
            #   ((step*64 + micro)*100003 + idx)*8 + 7 -- strat jitter, fresh
            #       per micro-step (requires acc_steps <= 64, asserted in
            #       __init__, else (step,micro) aliases into the next step).
            # The three window streams are mutually disjoint (distinct mod-8
            # residues). The iid/qmc streams use a different seed law that CAN
            # coincide with these across runs of different modes -- irrelevant
            # within a run, where only one family is ever drawn.
            m = self.acc_steps
            if self.mode in ("latperm", "strat"):
                # per-coordinate uniform permutation: rank of this micro's key
                # among the window's m keys (float32 key ties are ~2^-24 rare
                # and merely repeat a stratum for one window -- harmless)
                genp = torch.Generator(device=device)
                genp.manual_seed(((int(step) * 100003 + idx) * 8 + 6) & 0x7FFFFFFFFFFF)
                R = torch.rand((m, *shape), generator=genp, device=device, dtype=torch.float32)
                pi = (R < R[micro]).sum(0).float()
                if self.mode == "strat":
                    genj = torch.Generator(device=device)
                    genj.manual_seed((((int(step) * 64 + micro) * 100003 + idx) * 8 + 7) & 0x7FFFFFFFFFFF)
                    xi = torch.rand(shape, generator=genj, device=device, dtype=torch.float32)
                    # fp32 edge: (m-1 + xi)/m can round to exactly 1.0 with
                    # prob ~2^-25 per draw (floor then adds a full step);
                    # realized bias ~1e-15 -- accepted, not clamped, to keep
                    # streams bit-identical with the 2026-08-14 tier runs.
                    return (pi + xi) / m
                i = pi  # latperm: lattice points, permuted assignment
            else:
                i = micro
                if self.mode == "vdc":
                    nbits = self.acc_steps.bit_length() - 1
                    i = int(format(i, f"0{nbits}b")[::-1], 2) if nbits else 0
            gen = torch.Generator(device=device)
            gen.manual_seed(((int(step) * 100003 + idx) * 8 + 5) & 0x7FFFFFFFFFFF)
            u = torch.rand(shape, generator=gen, device=device, dtype=torch.float32)
            return torch.frac(u + i / m)
        if self.mode == "qmc":
            pair, flip = micro // 2, micro % 2 == 1
        else:
            pair, flip = micro, False  # iid: fresh stream every micro-step
        gen = torch.Generator(device=device)
        gen.manual_seed(((int(step) * 64 + pair) * 100003 + idx) * 2 + (self.mode == "qmc"))
        u = torch.rand(shape, generator=gen, device=device, dtype=torch.float32)
        return 1.0 - u if flip else u

    @torch.no_grad()
    def accumulate(self, step, micro):
        """Fold each param's fresh micro-gradient into its quantized accumulator
        and clear .grad so the next backward produces a fresh gradient."""
        if micro == 0:
            self.saturated = 0
            self._stall_sum = None
            self._stall_n = 0
        for idx, (_, p) in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.float()
            if micro == 0:
                self.step_size[idx] = self._row_step(g)
                self.accum[idx] = torch.zeros_like(g)
                self.shadow[idx] = torch.zeros_like(g)
            s = self.step_size[idx]
            prev = self.accum[idx]
            if self.ef != "none":
                if self.resid[idx] is None:
                    self.resid[idx] = torch.zeros_like(
                        g, dtype=torch.float16 if self.ef == "fp16" else torch.float32
                    )
                pre = prev + g + self.resid[idx].float()
            else:
                pre = prev + g
            t = pre / s
            if self.mode == "det":
                q = torch.round(t)
            else:
                q = torch.floor(t + self._u(step, micro, idx, t.shape, t.device))
            self.saturated += int((q.abs() > self.qmax).sum())
            new = q.clamp_(-self.qmax, self.qmax) * s
            if self.ef != "none":
                r = pre - new
                self.resid[idx] = r.half() if self.ef == "fp16" else r
            if micro > 0:
                # stall: the micro-add left the stored value unchanged
                # (on-grid values compare exactly; micro 0 starts from zero
                # so its "stall" would just measure zero-rounded grads)
                st = (new == prev).sum()
                self._stall_sum = st if self._stall_sum is None else self._stall_sum + st
                self._stall_n += new.numel()
            self.accum[idx] = new
            self.shadow[idx] = self.shadow[idx] + g
            p.grad = None

    @torch.no_grad()
    def write_back(self):
        """After the last micro-step: expose the quantized accumulation as .grad
        for clipping + the optimizer, and fold the window's mechanism stats
        (error vs the exact fp32 shadow in step units; stall; saturation) into
        the cross-window accumulators. No host sync here -- device tensors
        only; sync happens in mech_summary()."""
        for idx, (_, p) in enumerate(self.params):
            if self.accum[idx] is not None:
                p.grad = self.accum[idx].to(p.dtype)
                e = (self.accum[idx] - self.shadow[idx]) / self.step_size[idx]
                if self._mech_acc is None:
                    z = lambda: torch.zeros((), device=e.device, dtype=torch.float64)
                    self._mech_acc = {k: z() for k in
                                      ("err", "err_sq", "n", "stall", "stall_n",
                                       "resid_sq", "resid_max")}
                a = self._mech_acc
                a["err"] += e.sum(dtype=torch.float64)
                a["err_sq"] += (e * e).sum(dtype=torch.float64)
                a["n"] += e.numel()
                if self.ef != "none":
                    # EF backlog watch: unbounded growth here means the
                    # clamp is feeding the residual faster than it drains
                    rr = self.resid[idx].float() / self.step_size[idx]
                    a["resid_sq"] += (rr * rr).sum(dtype=torch.float64)
                    a["resid_max"] = torch.maximum(
                        a["resid_max"], rr.abs().max().double()
                    )
                self.shadow[idx] = None  # free between windows
        if self._mech_acc is not None and self._stall_sum is not None:
            self._mech_acc["stall"] += self._stall_sum
            self._mech_acc["stall_n"] += self._stall_n
        self._sat_acc += self.saturated

    def mech_summary(self):
        """Sync, return, and RESET the mechanism metrics aggregated over every
        window since the previous call (step units): err_mean, err_ms (window
        error vs exact fp32 accumulation), stall (micro-adds that left the
        stored accumulator unchanged; micro 0 excluded), sat (clamped coords).
        One GPU sync -- call at log intervals only."""
        if self._mech_acc is None:
            return None
        a = self._mech_acc
        self._mech_acc = None
        sat, self._sat_acc = self._sat_acc, 0
        n = max(float(a["n"]), 1.0)
        out = {
            "gq_err_mean": float(a["err"]) / n,
            "gq_err_ms": float(a["err_sq"]) / n,
            "gq_stall": float(a["stall"]) / max(float(a["stall_n"]), 1.0),
            "gq_sat": sat,
        }
        if self.ef != "none":
            out["gq_resid_ms"] = float(a["resid_sq"]) / n
            out["gq_resid_max"] = float(a["resid_max"])
        return out
