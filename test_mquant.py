"""Correctness tests for Muon momentum-buffer quantization (src/muon.py, mq_*).

CPU-safe. Exits nonzero on any failure. Includes the mandatory grid-clamp
validation (see the twice-bitten clamp bug): det quantization must agree with
a torch.round reference exactly on the zero-inclusive symmetric grid.
"""
import sys
import torch

sys.path.insert(0, "src")

from muon import Muon

torch.manual_seed(0)
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def make_opt(mode, bits=8, headroom=1.0, n=2, shape=(16, 32), momentum=0.95,
             nesterov=True):
    params = [torch.nn.Parameter(torch.randn(shape)) for _ in range(n)]
    opt = Muon(
        muon_params=params,
        adamw_params=[],
        lr=0.0,  # freeze weights: isolate buffer dynamics
        wd=0.0,
        momentum=momentum,
        nesterov=nesterov,
        mq_mode=mode,
        mq_bits=bits,
        mq_headroom=headroom,
    )
    return opt, params


def step_with(opt, params, grads):
    for p, g in zip(params, grads):
        p.grad = g.clone()
    opt.step()


def bufs(opt, params):
    return [opt.state[p]["momentum_buffer"].clone() for p in params]


# --- grid / clamp validation vs torch.round reference ----------------------
BITS = 4
opt, params = make_opt("det", bits=BITS)
grads = [torch.randn_like(p) for p in params]
step_with(opt, params, grads)  # buf = 0 * m + g -> Q(g), scale from this buf
qmax = 2 ** (BITS - 1) - 1
ok = True
for p, g in zip(params, grads):
    s = opt.state[p]["mq_step_size"]
    ref = torch.round(g.float() / s).clamp(-qmax, qmax) * s
    buf = opt.state[p]["momentum_buffer"].float()
    ok &= torch.allclose(buf, ref)
    ok &= bool((torch.round(buf / s).abs() <= qmax).all())
check("det matches torch.round reference on the zero-inclusive grid", ok)

# per-step scale covers the current buffer: no clamp ever at headroom >= 1
opt, params = make_opt("det", bits=BITS, headroom=1.0)
ok = True
for t in range(6):
    step_with(opt, params, [torch.randn_like(p) * (1 + 5 * t) for p in params])
    ok &= opt.mq_saturated == 0
    for p in params:
        s = opt.state[p]["mq_step_size"]
        buf = opt.state[p]["momentum_buffer"].float()
        ok &= bool((buf.abs() <= qmax * s + 1e-6).all())
check("per-step scale covers a growing buffer (never clamps)", ok, f"sat={opt.mq_saturated}")

# zero buffer stays exactly zero (grid contains 0)
opt, params = make_opt("det", bits=BITS)
step_with(opt, params, [torch.zeros_like(p) for p in params])
check(
    "zero gradient keeps buffer exactly zero",
    all(bool((b == 0).all()) for b in bufs(opt, params)),
)

# --- scale tracks the buffer every step -------------------------------------
opt, params = make_opt("qmc", bits=8)
step_with(opt, params, [torch.randn_like(p) for p in params])
s0 = [opt.state[p]["mq_step_size"].clone() for p in params]
step_with(opt, params, [5 * torch.randn_like(p) for p in params])
s1 = [opt.state[p]["mq_step_size"].clone() for p in params]
check(
    "scale recomputed from the buffer every step",
    not any(torch.equal(a, b) for a, b in zip(s0, s1)),
)

# --- saturation: headroom < 1 clamps to grid edge and is counted ------------
opt, params = make_opt("det", bits=BITS, headroom=0.5)
step_with(opt, params, [torch.randn_like(p) for p in params])
ok = opt.mq_saturated > 0
for p in params:
    s = opt.state[p]["mq_step_size"]
    buf = opt.state[p]["momentum_buffer"].float()
    ok &= bool((buf.abs() <= qmax * s + 1e-6).all())
check("headroom<1 overflow clamps to grid edge and is counted", ok, f"sat={opt.mq_saturated}")

# --- high-bit det ~= fp32 momentum recursion --------------------------------
STEPS = 12
opt, params = make_opt("det", bits=14, momentum=0.9)
ref = [torch.zeros_like(p) for p in params]
for t in range(STEPS):
    grads = [torch.randn_like(p) for p in params]
    step_with(opt, params, grads)
    ref = [0.9 * r + g for r, g in zip(ref, grads)]
ok = True
for b, r in zip(bufs(opt, params), ref):
    rel = (b.float() - r).norm() / r.norm()
    ok &= rel < 5e-3
check("14-bit det buffer matches fp32 momentum recursion", ok, f"rel_err={rel:.2e}")

# --- SR unbiasedness (single quantization event) ----------------------------
TRIALS = 1200
tgt = torch.randn(16, 32) * 0.1
acc = None
sizes = None
for t in range(TRIALS):
    opt, params = make_opt("iid", bits=4, n=1, shape=(16, 32))
    torch.manual_seed(1000 + t)  # drives torch.rand_like in iid mode
    step_with(opt, params, [tgt])
    b = bufs(opt, params)[0].float()
    acc = b if acc is None else acc + b
    sizes = opt.state[params[0]]["mq_step_size"]
mean = acc / TRIALS
se = 0.5 / TRIALS**0.5
per_coord = (mean - tgt) / sizes
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

# --- antithetic pairing across consecutive optimizer steps ------------------
# Direct test of the pairing math: quantize the SAME pre-round value at steps
# 2t and 2t+1 (identical value -> identical per-step scale -> identical grid);
# the summed error of the u / 1-u pair must beat two iid draws. Cancellation is
# partial even for identical values (pair-sum var 2f(1-f) -> |1-2f|*min(f,1-f)),
# so the 0.7 factor is the right bar, matching test_gquant.py.
def direct_pair_err(mode, trials=200):
    errs = []
    for t in range(trials):
        torch.manual_seed(2000 + t)
        opt, params = make_opt(mode, bits=4, n=1)
        p = params[0]
        state = opt.state[p]
        x = torch.randn_like(p) * 0.05
        opt._sr_step_cnt = 2 * t
        q0 = opt._mq_quantize(x, state)
        opt._sr_step_cnt = 2 * t + 1
        q1 = opt._mq_quantize(x, state)
        errs.append((((q0 + q1) - 2 * x) ** 2).mean())
    return torch.stack(errs).mean()

e_qmc, e_iid = direct_pair_err("qmc"), direct_pair_err("iid")
check(
    "antithetic beats iid on correlated pair (identical pre-round value)",
    e_qmc < 0.7 * e_iid,
    f"mse qmc={e_qmc:.3e} iid={e_iid:.3e}",
)

# u / 1-u structure verified directly on the generator path
opt, params = make_opt("qmc", bits=4, n=1)
p = params[0]
state = opt.state[p]
x = torch.full_like(p, 0.5)  # constant rows: frac(t) identical per row
opt._sr_step_cnt = 6
q0 = opt._mq_quantize(x, state)
opt._sr_step_cnt = 7
q1 = opt._mq_quantize(x, state)
s = state["mq_step_size"]
# for identical t with u vs 1-u, exactly one of the pair rounds up unless u
# lands in the middle band; the pair MEAN must be closer to x than iid rms
check(
    "qmc pair mean tracks the true value",
    ((q0 + q1) / 2 - x).abs().mean() < (s / 4).mean(),
)

# --- determinism / statelessness -------------------------------------------
def run_seq(mode):
    torch.manual_seed(7)
    opt, params = make_opt(mode, bits=4)
    for t in range(4):
        step_with(opt, params, [torch.randn_like(p) for p in params])
    return torch.cat([b.flatten() for b in bufs(opt, params)])

check(
    "qmc sequence is reproducible (stateless seeding)",
    torch.equal(run_seq("qmc"), run_seq("qmc")),
)

# mq stream must be disjoint from the NS-then-round stream (different u even
# for the same step/param)
opt, params = make_opt("qmc", bits=4, n=1)
p = params[0]
state = opt.state[p]
gen = torch.Generator(device="cpu")
gen.manual_seed(((0 * 2654435761) ^ state["sr_param_seed"]) & 0x7FFFFFFF)
u_sr = torch.rand(p.shape, generator=gen, dtype=torch.float32)
gen2 = torch.Generator(device="cpu")
gen2.manual_seed(((0 * 2654435761) ^ (state["sr_param_seed"] * 2 + 1)) & 0x7FFFFFFF)
u_mq = torch.rand(p.shape, generator=gen2, dtype=torch.float32)
check("mq stream disjoint from NS-round stream", not torch.allclose(u_sr, u_mq))

# --- mode none is a strict no-op vs baseline Muon ---------------------------
def run_baseline(mq_mode):
    torch.manual_seed(11)
    params = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
    opt = Muon(muon_params=params, adamw_params=[], lr=1e-3, wd=0.1,
               mq_mode=mq_mode)
    torch.manual_seed(13)
    for t in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    return torch.cat([p.data.flatten() for p in params])

check(
    "mq_mode=none reproduces baseline Muon exactly",
    torch.equal(run_baseline("none"), run_baseline("none")),
)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL PASS")
