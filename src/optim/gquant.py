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

Index-clamp bounds for THIS grid (q = round-or-floor of t = x/step) are
[-qmax, qmax]; validated against torch.round in test_gquant.py (the centered
no-zero QuEST grid has DIFFERENT bounds -- see the twice-bitten clamp bug).

DDP: accumulators are per-rank and .grad is consumed every micro-step, which
bypasses DDP's gradient reduction -- requires world_size 1 (asserted in main).
"""

import torch


class GradAccumQuantizer:
    def __init__(self, model, mode, bits=8, headroom=1.0, acc_steps=8):
        assert mode in ("det", "iid", "qmc")
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

    def _row_step(self, g):
        absmax = g.abs().flatten(1).amax(dim=1).clamp_min(1e-12)
        span = self.headroom * self.acc_steps * absmax
        return (span / self.qmax).view(-1, *([1] * (g.ndim - 1)))

    def _u(self, step, micro, idx, shape, device):
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
        for idx, (_, p) in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.float()
            if micro == 0:
                self.step_size[idx] = self._row_step(g)
                self.accum[idx] = torch.zeros_like(g)
            s = self.step_size[idx]
            t = (self.accum[idx] + g) / s
            if self.mode == "det":
                q = torch.round(t)
            else:
                q = torch.floor(t + self._u(step, micro, idx, t.shape, t.device))
            self.saturated += int((q.abs() > self.qmax).sum())
            self.accum[idx] = q.clamp_(-self.qmax, self.qmax) * s
            p.grad = None

    @torch.no_grad()
    def write_back(self):
        """After the last micro-step: expose the quantized accumulation as .grad
        for clipping + the optimizer."""
        for idx, (_, p) in enumerate(self.params):
            if self.accum[idx] is not None:
                p.grad = self.accum[idx].to(p.dtype)
