"""Correctness tests for G-quantized gradient accumulation (src/optim/gquant.py).

CPU-safe (gquant imports only torch -- no CUDA Hadamard). Exits nonzero on any
failure. Includes the mandatory grid-clamp validation (see the twice-bitten
clamp bug): det quantization must agree with a torch.round reference exactly.
"""
import sys
import torch

sys.path.insert(0, "src")

from optim.gquant import GradAccumQuantizer

torch.manual_seed(0)
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(32, 64, bias=False)
        self.fc2 = torch.nn.Linear(64, 8, bias=False)
        self.norm = torch.nn.Parameter(torch.ones(64))  # 1D: must be excluded
        self.wte = torch.nn.Embedding(16, 32)  # embedding: must be excluded


def run_window(gq, grads_per_micro, step=0):
    """Feed a list of per-micro grad dicts {param: tensor} through one window."""
    for micro, grads in enumerate(grads_per_micro):
        for (_, p) in gq.params:
            p.grad = grads[p].clone()
        gq.accumulate(step, micro)
    gq.write_back()


# --- param selection -------------------------------------------------------
m = Toy()
gq = GradAccumQuantizer(m, "det", bits=8, acc_steps=4)
names = [n for n, _ in gq.params]
check(
    "param selection: 2D non-embedding only",
    names == ["fc1.weight", "fc2.weight"],
    f"got {names}",
)

# --- grid / clamp validation vs torch.round reference ----------------------
ACC = 4
m = Toy()
gq = GradAccumQuantizer(m, "det", bits=4, acc_steps=ACC)
g0 = {p: torch.randn_like(p) for _, p in gq.params}
run_window(gq, [g0] + [{p: torch.zeros_like(p) for _, p in gq.params}] * (ACC - 1))
ok = True
for idx, (_, p) in enumerate(gq.params):
    s = gq.step_size[idx]
    qmax = 2 ** (4 - 1) - 1
    ref = torch.round(g0[p] / s).clamp(-qmax, qmax) * s
    ok &= torch.allclose(p.grad.float(), ref)
    idxs = torch.round(p.grad.float() / s)
    ok &= bool((idxs.abs() <= qmax).all())
check("det matches torch.round reference on the zero-inclusive grid", ok)

# zero grads stay exactly zero (grid contains 0)
gq = GradAccumQuantizer(m, "det", bits=4, acc_steps=2)
gz = {p: torch.zeros_like(p) for _, p in gq.params}
gnz = {p: torch.randn_like(p) for _, p in gq.params}
run_window(gq, [gnz, gz])  # micro 0 sets scale, micro 1 adds zeros
for micro0_only in [True]:
    pass
gq2 = GradAccumQuantizer(m, "det", bits=4, acc_steps=1)
run_window(gq2, [gz])
check(
    "zero gradient quantizes to exactly zero",
    all(bool((p.grad == 0).all()) for _, p in gq2.params),
)

# saturation: huge later micro-grads must clamp, not wrap
gq = GradAccumQuantizer(m, "det", bits=4, acc_steps=2)
run_window(gq, [gnz, {p: 100 * gnz[p] for _, p in gq.params}])
qmax = 2 ** (4 - 1) - 1
ok = gq.saturated > 0
for idx, (_, p) in enumerate(gq.params):
    ok &= bool((p.grad.float().abs() <= qmax * gq.step_size[idx] + 1e-6).all())
check("saturating adds clamp to grid edge and are counted", ok, f"saturated={gq.saturated}")

# --- high-bit det accumulation ~= fp32 sum ---------------------------------
ACC = 8
m = Toy()
gq = GradAccumQuantizer(m, "det", bits=14, acc_steps=ACC)
micros = [{p: torch.randn_like(p) / ACC for _, p in gq.params} for _ in range(ACC)]
run_window(gq, micros)
ok = True
for _, p in gq.params:
    fp32 = sum(g[p] for g in micros)
    rel = (p.grad.float() - fp32).norm() / fp32.norm()
    ok &= rel < 5e-3
check("14-bit det accumulation matches fp32 sum", ok, f"rel_err={rel:.2e}")

# --- SR unbiasedness --------------------------------------------------------
TRIALS = 1200
m = Toy()
gq = GradAccumQuantizer(m, "iid", bits=4, acc_steps=1)
g = {p: torch.randn_like(p) * 0.1 for _, p in gq.params}
acc = None
for t in range(TRIALS):
    run_window(gq, [g], step=t)  # new step -> new u
    got = torch.cat([p.grad.float().flatten() for _, p in gq.params])
    acc = got if acc is None else acc + got
mean = acc / TRIALS
target = torch.cat([g[p].flatten() for _, p in gq.params])
step_sz = torch.cat(
    [gq.step_size[i].expand_as(p).flatten() for i, (_, p) in enumerate(gq.params)]
)
# SR error per coord has sd <= step/2, so the mean of TRIALS draws has
# se <= step/(2*sqrt(TRIALS)); test the max over coords at ~4 sigma and the
# grand mean at ~4 sigma of its (much smaller) se.
se = 0.5 / TRIALS**0.5
per_coord = (mean - target) / step_sz
check(
    "iid SR is unbiased (max coord)",
    per_coord.abs().max() < 4.5 * se,
    f"max |bias|/step = {per_coord.abs().max():.4f}, 4.5se = {4.5*se:.4f}",
)
check(
    "iid SR is unbiased (grand mean)",
    per_coord.mean().abs() < 4 * se / per_coord.numel() ** 0.5,
    f"mean bias/step = {per_coord.mean():.2e}",
)

# --- antithetic pairing: u1 = 1 - u0, and variance beats iid ----------------
m = Toy()
gq = GradAccumQuantizer(m, "qmc", bits=8, acc_steps=2)
p0 = gq.params[0][1]
u0 = gq._u(3, 0, 0, p0.shape, p0.device)
u1 = gq._u(3, 1, 0, p0.shape, p0.device)
check("qmc pair uses u and 1-u", torch.allclose(u0 + u1, torch.ones_like(u0)))

m2 = Toy()
gqi = GradAccumQuantizer(m2, "iid", bits=8, acc_steps=2)
ui0 = gqi._u(3, 0, 0, p0.shape, p0.device)
ui1 = gqi._u(3, 1, 0, p0.shape, p0.device)
check("iid micros draw independent u", not torch.allclose(ui0 + ui1, torch.ones_like(ui0)))

# identical grad in both pair members: antithetic error should collapse
def window_err(mode, trials=100):
    m = Toy()
    gq = GradAccumQuantizer(m, mode, bits=4, acc_steps=2)
    g = {p: torch.randn_like(p) * 0.05 for _, p in gq.params}
    errs = []
    for t in range(trials):
        run_window(gq, [g, g], step=t)
        got = torch.cat([p.grad.float().flatten() for _, p in gq.params])
        tgt = torch.cat([2 * g[p].flatten() for _, p in gq.params])
        errs.append(((got - tgt) ** 2).mean())
    return torch.stack(errs).mean()

e_qmc, e_iid = window_err("qmc"), window_err("iid")
check(
    "antithetic beats iid on correlated pair (identical micro-grads)",
    e_qmc < 0.7 * e_iid,
    f"mse qmc={e_qmc:.3e} iid={e_iid:.3e}",
)

# --- determinism / statelessness -------------------------------------------
def one_window(mode):
    m = Toy()
    torch.manual_seed(7)
    gq = GradAccumQuantizer(m, mode, bits=4, acc_steps=4)
    micros = [{p: torch.randn_like(p) for _, p in gq.params} for _ in range(4)]
    run_window(gq, micros, step=5)
    return torch.cat([p.grad.float().flatten() for _, p in gq.params])

check("qmc window is reproducible (stateless seeding)", torch.equal(one_window("qmc"), one_window("qmc")))

# --- .grad handling ---------------------------------------------------------
m = Toy()
gq = GradAccumQuantizer(m, "det", bits=8, acc_steps=2)
g = {p: torch.randn_like(p) for _, p in gq.params}
for _, p in gq.params:
    p.grad = g[p].clone()
gq.accumulate(0, 0)
check(".grad cleared after each accumulate", all(p.grad is None for _, p in gq.params))
gq_dtype_ok = True
for _, p in gq.params:
    p.grad = g[p].clone()
gq.accumulate(0, 1)
gq.write_back()
check("write_back restores .grad in param dtype", all(p.grad is not None and p.grad.dtype == p.dtype for _, p in gq.params))

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL PASS")
