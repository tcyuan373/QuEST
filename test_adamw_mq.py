"""Correctness tests for MQAdamW (src/adamw_mq.py): quantized first-moment
storage. CPU-safe. Exits nonzero on any failure."""
import sys
import torch

sys.path.insert(0, "src")

from adamw_mq import MQAdamW, _SEED_MIX

torch.manual_seed(0)
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def make(mode, n=2, shape=(16, 32), bits=4, seed=11):
    torch.manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(shape)) for _ in range(n)]
    opt = MQAdamW(
        params, m1_quant_params=params, m1_mode=mode, m1_bits=bits,
        lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1,
    )
    return opt, params


def steps(opt, params, n_steps, gseed=13):
    torch.manual_seed(gseed)
    for _ in range(n_steps):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()


# --- mode=none is bit-identical to torch.optim.AdamW -------------------------
torch.manual_seed(11)
ref_params = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
ref = torch.optim.AdamW(ref_params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
torch.manual_seed(13)
for _ in range(5):
    for p in ref_params:
        p.grad = torch.randn_like(p)
    ref.step()
opt0, params0 = make("none")
steps(opt0, params0, 5)
check(
    "m1_mode=none is bit-identical to torch.optim.AdamW",
    all(torch.equal(a, b) for a, b in zip(params0, ref_params))
    and all(
        torch.equal(opt0.state[a]["exp_avg"], ref.state[b]["exp_avg"])
        for a, b in zip(params0, ref_params)
    ),
)

# --- stored exp_avg is on-grid after every step ------------------------------
for mode in ("det", "iid", "qmc", "strat"):
    opt, params = make(mode)
    steps(opt, params, 5)
    qmax = 2 ** (4 - 1) - 1
    ok = True
    for p in params:
        m = opt.state[p]["exp_avg"]
        absmax = m.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        # reconstruct: on-grid means m/s is (near-)integer with |q|<=qmax.
        # NOTE absmax post-quantization equals qmax*s only if some row
        # saturates the grid -- instead verify integrality against the
        # smallest positive stored magnitude... simpler: re-quantizing with
        # det must be a fixed point (Q(m)==m for on-grid m).
        s = 1.0 * absmax.float() / qmax
        q = m.float() / s
        ok = ok and torch.allclose(q, torch.round(q), atol=1e-4)
        ok = ok and bool((q.abs() <= qmax + 1e-4).all())
    check(f"{mode}: stored exp_avg is on-grid (int levels, |q|<=qmax)", ok)

# --- iid draws independent of global RNG; sequences reproducible -------------
outs = []
for gseed in (1, 2):
    opt, params = make("iid")
    torch.manual_seed(999)
    grads = [torch.randn_like(p) for p in params]
    torch.manual_seed(gseed)  # perturb global RNG differently
    for p, g in zip(params, grads):
        p.grad = g.clone()
    opt.step()
    outs.append(torch.cat([opt.state[p]["exp_avg"].flatten() for p in params]))
check("iid draws independent of global RNG state", torch.equal(outs[0], outs[1]))

for mode in ("iid", "qmc", "strat"):
    def run(m=mode):
        opt, params = make(m)
        steps(opt, params, 6)
        return torch.cat([opt.state[p]["exp_avg"].flatten() for p in params])
    check(f"{mode} sequence reproducible", torch.equal(run(), run()))

# --- realized draws match the documented seed contract (iid, strat) ----------
opt, params = make("iid")
p = params[0]
torch.manual_seed(21)
g0 = {q: torch.randn_like(q) for q in params}
for q in params:
    q.grad = g0[q].clone()
# reference: plain AdamW step to get the pre-quant exp_avg
optr, paramsr = make("none")
for q, qr in zip(params, paramsr):
    qr.grad = g0[q].clone()
optr.step()
opt.step()
ps = opt._param_seed[id(p)]
gen = torch.Generator()
gen.manual_seed(((0 * _SEED_MIX) ^ (ps * 8 + 1)) & 0x7FFFFFFF)
u = torch.rand(p.shape, generator=gen, dtype=torch.float32)
mref = optr.state[paramsr[0]]["exp_avg"]
qmax = 7
absmax = mref.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
s = absmax.float() / qmax
want = (torch.floor(mref.float() / s + u).clamp(-qmax, qmax) * s).to(mref.dtype)
check(
    "iid realized quantization matches seed contract",
    torch.equal(opt.state[p]["exp_avg"], want),
)

# --- partition scoping: params outside m1_quant_params untouched -------------
torch.manual_seed(11)
pa = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
opt_scoped = MQAdamW(
    pa, m1_quant_params=pa[:1], m1_mode="iid", m1_bits=4,
    lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1,
)
torch.manual_seed(11)
pb = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
opt_ref2 = torch.optim.AdamW(pb, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
torch.manual_seed(13)
for _ in range(3):
    for q1, q2 in zip(pa, pb):
        g = torch.randn_like(q1)
        q1.grad = g.clone()
        q2.grad = g.clone()
    opt_scoped.step()
    opt_ref2.step()
check(
    "unquantized param's exp_avg matches plain AdamW exactly... (diverges via"
    " params only after quantized partner shifts shared loss -- here grads are"
    " independent, so exact)",
    torch.equal(opt_scoped.state[pa[1]]["exp_avg"], opt_ref2.state[pb[1]]["exp_avg"]),
)

# --- counter restore hook (resume path) --------------------------------------
opt, params = make("strat")
steps(opt, params, 4)
sd = opt.state_dict()
opt2, params2 = make("strat")
opt2.load_state_dict(sd)
check("_sr_step_cnt attribute present for base.py resume restore",
      hasattr(opt2, "_sr_step_cnt"))
opt2._sr_step_cnt = 4  # what base.py does with curr_iter
for p in params2:
    p.grad = torch.randn_like(p)
opt2.step()
check("post-restore step runs and advances counter", opt2._sr_step_cnt == 5)

# --- mech summary: populated, sane, drains -----------------------------------
opt, params = make("iid")
steps(opt, params, 3)
m = opt.mq_mech_summary()
check(
    "mech summary populated, sane, drains",
    m is not None
    and 0.0 <= m["mq_stall"] <= 1.0
    and m["mq_err_ms"] >= 0.0
    and abs(m["mq_err_mean"]) < 0.1
    and opt.mq_mech_summary() is None,
    str(m),
)
opt0, params0 = make("none")
steps(opt0, params0, 2)
check("mech is None when mode=none", opt0.mq_mech_summary() is None)

# --- MC unbiasedness of a single quantization event --------------------------
TRIALS = 800
opt, params = make("det")  # build once for the reference exp_avg
torch.manual_seed(31)
gfix = {p: torch.randn_like(p) for p in params}
optr, paramsr = make("none")
for p, pr in zip(params, paramsr):
    pr.grad = gfix[p].clone()
optr.step()
mref = optr.state[paramsr[0]]["exp_avg"].float()
absmax = mref.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
s = absmax / 7
acc = None
for t in range(TRIALS):
    opt_t, params_t = make("iid")
    opt_t._sr_step_cnt = t
    for p, pt in zip(params, params_t):
        pt.grad = gfix[p].clone()
    opt_t.step()
    b = opt_t.state[params_t[0]]["exp_avg"].float()
    acc = b if acc is None else acc + b
bias = ((acc / TRIALS - mref) / s)
se = 0.5 / TRIALS**0.5
check(
    "iid m1 quantization is unbiased (max coord)",
    bias.abs().max() < 4.5 * se,
    f"max |bias|/step = {bias.abs().max():.4f} (thr {4.5*se:.4f})",
)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL PASS")
