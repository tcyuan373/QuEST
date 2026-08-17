"""AdamW with quantized first-moment STORAGE: exp_avg <- Q(exp_avg) after
every step -- the optimizer-generality arm of the QMC-SR program.

Same accumulate-without-error-feedback structure as Muon's mq site
(src/muon.py): the stored m1 is on-grid between steps, torch's step computes
m_t = beta1*Q(m_{t-1}) + (1-beta1)*g_t internally, and we re-quantize the
stored tensor afterwards. The parameter update each step consumes the
freshly-computed (unquantized) m_t; only the PERSISTENT state is low-bit --
the honest bytes/param semantics, mirroring mq where NS consumes the
quantized buffer.

Scope: quantizes exp_avg of >=2D non-embedding params only (the exact Muon
partition, for apples-to-apples with the mq tables); embeddings/norms/
exp_avg_sq stay fp32 (second moment is explicitly out of scope -- nonneg +
sqrt'd, a different problem).

Grid and streams mirror _mq_quantize: per-row absmax zero-inclusive INT
grid recomputed each step, headroom, stateless seed streams keyed by
(step counter, param index) with the same residue layout. The counter is
named _sr_step_cnt and the mech summary mq_mech_summary()/mq_mode ON
PURPOSE: src/optim/base.py's resume restore (opt._sr_step_cnt = curr_iter)
and >Mech logging then work unchanged for this optimizer.
"""

import torch

_SEED_MIX = 2654435761


class MQAdamW(torch.optim.AdamW):
    def __init__(
        self,
        params,
        m1_quant_params=None,  # iterable of params whose exp_avg is quantized
        m1_mode="none",  # "none" | "det" | "iid" | "qmc" | "strat"
        m1_bits=4,
        m1_headroom=1.0,
        **kw,
    ):
        super().__init__(params, **kw)
        assert m1_mode in ("none", "det", "iid", "qmc", "strat"), m1_mode
        self.mq_mode = m1_mode  # name shared with Muon for base.py logging
        self.m1_bits = int(m1_bits)
        self.m1_headroom = float(m1_headroom)
        self._qparams = list(m1_quant_params) if m1_quant_params is not None else []
        # fixed per-param seed offsets, same construction as Muon
        self._param_seed = {
            id(p): (idx + 1) * 0x9E3779B1 for idx, p in enumerate(self._qparams)
        }
        self._sr_step_cnt = 0  # restored on resume by src/optim/base.py
        self.mq_saturated = 0
        self._mech_acc = None

    def _u(self, cnt, ps, shape, device):
        """Dither draw for step `cnt`, param seed `ps`. Same stateless residue
        layout as Muon mq: iid *8+1, qmc *2+1, strat perm *8+5 / jitter *8+7
        (separate optimizer instance -- the streams never co-run with Muon's)."""
        if self.mq_mode == "qmc":
            pair, parity = cnt // 2, cnt % 2
            g = torch.Generator(device=device)
            g.manual_seed(((pair * _SEED_MIX) ^ (ps * 2 + 1)) & 0x7FFFFFFF)
            u = torch.rand(shape, generator=g, device=device, dtype=torch.float32)
            return 1.0 - u if parity == 1 else u
        if self.mq_mode == "strat":
            block, phase = cnt // 8, cnt % 8
            gp = torch.Generator(device=device)
            gp.manual_seed(((block * _SEED_MIX) ^ (ps * 8 + 5)) & 0x7FFFFFFF)
            R = torch.rand((8, *shape), generator=gp, device=device, dtype=torch.float32)
            pi = (R < R[phase]).sum(0).float()
            gj = torch.Generator(device=device)
            gj.manual_seed(((cnt * _SEED_MIX) ^ (ps * 8 + 7)) & 0x7FFFFFFF)
            xi = torch.rand(shape, generator=gj, device=device, dtype=torch.float32)
            return (pi + xi) / 8
        # iid
        g = torch.Generator(device=device)
        g.manual_seed(((cnt * _SEED_MIX) ^ (ps * 8 + 1)) & 0x7FFFFFFF)
        return torch.rand(shape, generator=g, device=device, dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        if self.mq_mode == "none":
            loss = super().step(closure)
            self._sr_step_cnt += 1
            return loss

        self.mq_saturated = 0
        # previous stored (quantized) m1, for the stall/staleness metric
        prevs = {
            id(p): self.state[p]["exp_avg"].clone()
            for p in self._qparams
            if p in self.state and "exp_avg" in self.state[p]
        }
        loss = super().step(closure)

        qmax = 2 ** (self.m1_bits - 1) - 1
        cnt = self._sr_step_cnt
        for p in self._qparams:
            state = self.state.get(p)
            if not state or "exp_avg" not in state:
                continue
            m = state["exp_avg"]
            absmax = (
                m.abs().flatten(1).amax(dim=1).clamp_min(1e-12)
                .view(-1, *([1] * (m.ndim - 1)))
            )
            s = self.m1_headroom * absmax.float() / qmax
            t = m.float() / s
            if self.mq_mode == "det":
                q = torch.round(t)
            else:
                u = self._u(cnt, self._param_seed[id(p)], t.shape, m.device)
                q = torch.floor(t + u)
            self.mq_saturated += int((q.abs() > qmax).sum())
            out = (q.clamp_(-qmax, qmax) * s).to(m.dtype)
            # mech (observation only): quantization error in step units,
            # aggregated across steps, drained by mq_mech_summary()
            e = (out - m) / s
            if self._mech_acc is None:
                z = lambda: torch.zeros((), device=m.device, dtype=torch.float64)
                self._mech_acc = {
                    k: z() for k in ("err", "err_sq", "n", "stall", "stall_n")
                }
            a = self._mech_acc
            a["err"] += e.sum(dtype=torch.float64)
            a["err_sq"] += (e * e).sum(dtype=torch.float64)
            a["n"] += e.numel()
            prev = prevs.get(id(p))
            if prev is not None:
                a["stall"] += ((out - prev).abs() < 0.5 * s).sum()
                a["stall_n"] += out.numel()
            m.copy_(out)
        self._sr_step_cnt += 1
        return loss

    def mq_mech_summary(self):
        """Same drain-on-read contract as Muon.mq_mech_summary."""
        if self._mech_acc is None:
            return None
        a = self._mech_acc
        self._mech_acc = None
        n = max(float(a["n"]), 1.0)
        return {
            "mq_err_mean": float(a["err"]) / n,
            "mq_err_ms": float(a["err_sq"]) / n,
            "mq_stall": float(a["stall"]) / max(float(a["stall_n"]), 1.0),
            "mq_sat": self.mq_saturated,
        }
