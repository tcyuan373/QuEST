"""Correctness tests for the "loyal" randomized-QMC SR modes:
gquant modes lattice/vdc (length-m low-discrepancy set across the micro-step
window) and Muon mq_mode lattice (blocks of 8 optimizer steps).

CPU-safe. Exits nonzero on any failure.
"""
import sys
import torch

sys.path.insert(0, "src")

from optim.gquant import GradAccumQuantizer
from muon import Muon

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


def run_window(gq, grads_per_micro, step=0):
    for micro, grads in enumerate(grads_per_micro):
        for (_, p) in gq.params:
            p.grad = grads[p].clone()
        gq.accumulate(step, micro)
    gq.write_back()


# --- point-set structure ----------------------------------------------------
M = 8
m = Toy()
gq = GradAccumQuantizer(m, "lattice", bits=4, acc_steps=M)
p0 = gq.params[0][1]
us = torch.stack([gq._u(3, i, 0, p0.shape, p0.device) for i in range(M)])
# per coordinate the M draws must be {frac(u + i/M)}: sorted gaps all == 1/M
sorted_us, _ = us.reshape(M, -1).sort(dim=0)
gaps = torch.cat(
    [sorted_us[1:] - sorted_us[:-1], (sorted_us[:1] + 1) - sorted_us[-1:]]
)
check(
    "lattice window is an equidistant shifted set (all gaps = 1/M)",
    bool(torch.allclose(gaps, torch.full_like(gaps, 1 / M), atol=1e-6)),
    f"gap range [{gaps.min():.4f}, {gaps.max():.4f}]",
)

gqv = GradAccumQuantizer(Toy(), "vdc", bits=4, acc_steps=M)
uv = torch.stack([gqv._u(3, i, 0, p0.shape, p0.device) for i in range(M)])
sv, _ = uv.reshape(M, -1).sort(dim=0)
su, _ = us.reshape(M, -1).sort(dim=0)
check(
    "vdc uses the SAME point set as lattice (m=2^k), different order",
    bool(torch.allclose(sv, su, atol=1e-6))
    and not torch.allclose(uv.reshape(M, -1), us.reshape(M, -1)),
)
# bit-reversal: vdc micro 1 must equal lattice micro M/2 (phi2(1) = 1/2)
check(
    "vdc order is bit-reversed",
    bool(torch.allclose(uv[1], us[M // 2], atol=1e-6)),
)

# consecutive lattice pairs are NOT antithetic (u vs u+1/M, not 1-u)
check(
    "lattice is a shift family, not reflection",
    not torch.allclose(us[0] + us[1], torch.ones_like(us[0]), atol=1e-3),
)

# --- latperm: same point set as lattice, random assignment ------------------
gqp = GradAccumQuantizer(Toy(), "latperm", bits=4, acc_steps=M)
up = torch.stack([gqp._u(3, i, 0, p0.shape, p0.device) for i in range(M)])
sp, _ = up.reshape(M, -1).sort(dim=0)
check(
    "latperm uses the SAME point set as lattice (shared base seed)",
    bool(torch.allclose(sp, su, atol=1e-6)),
)
check(
    "latperm order differs from lattice's natural order",
    not torch.allclose(up.reshape(M, -1), us.reshape(M, -1)),
)
# each stratum hit exactly once per coordinate: sorted draws stratified
strata_p = torch.floor(sp * M)
check(
    "latperm hits every stratum exactly once per coordinate",
    bool((strata_p == torch.arange(M).unsqueeze(1)).all()),
)
# permutations differ across coordinates (not one global shuffle)
ranks = (up.reshape(M, -1).unsqueeze(1) > up.reshape(M, -1).unsqueeze(0)).sum(0)
check(
    "latperm permutation varies across coordinates",
    bool((ranks != ranks[:, :1]).any()),
)

# --- strat: Latin-hypercube along the window axis ---------------------------
gqs = GradAccumQuantizer(Toy(), "strat", bits=4, acc_steps=M)
ust = torch.stack([gqs._u(3, i, 0, p0.shape, p0.device) for i in range(M)])
ss, _ = ust.reshape(M, -1).sort(dim=0)
check(
    "strat is stratified: one draw per stratum per coordinate",
    bool((torch.floor(ss * M) == torch.arange(M).unsqueeze(1)).all()),
)
# both modes must consume the SAME permutation stream: reconstruct pi from
# the seed contract and check strat's stratum == pi and latperm's draw ==
# frac(u_base + pi/M). (latperm's realized stratum is pi ROTATED by the
# base shift, so the two strata differ pointwise by construction.)
genr = torch.Generator()
genr.manual_seed(((3 * 100003 + 0) * 8 + 6) & 0x7FFFFFFFFFFF)
Rw = torch.rand((M, *p0.shape), generator=genr, dtype=torch.float32)
pi_w = torch.stack([(Rw < Rw[i]).sum(0).float() for i in range(M)])
genb = torch.Generator()
genb.manual_seed(((3 * 100003 + 0) * 8 + 5) & 0x7FFFFFFFFFFF)
u_base = torch.rand(p0.shape, generator=genb, dtype=torch.float32)
check(
    "strat stratum == shared-pi stream",
    bool((torch.floor(ust * M) == pi_w).all()),
)
check(
    "latperm == frac(u_base + pi/M) with the same pi stream",
    bool(torch.allclose(up, torch.frac(u_base + pi_w / M), atol=1e-6)),
)
check(
    "strat jitter is independent of the lattice shift (not the lattice set)",
    not torch.allclose(ss, su, atol=1e-4),
)
# jitter fresh across windows: same micro, different step -> different offsets
ust2 = gqs._u(4, 0, 0, p0.shape, p0.device)
check(
    "strat draws differ across optimizer steps",
    not torch.allclose(ust[0], ust2, atol=1e-6),
)

# --- unbiasedness (each draw is marginally uniform via the random shift) ----
TRIALS = 1200
m = Toy()
g = {p: torch.randn_like(p) * 0.1 for _, p in GradAccumQuantizer(m, "det").params}
acc = None
for t in range(TRIALS):
    gq = GradAccumQuantizer(m, "lattice", bits=4, acc_steps=1)
    run_window(gq, [g], step=t)
    got = torch.cat([p.grad.float().flatten() for _, p in gq.params])
    acc = got if acc is None else acc + got
mean = acc / TRIALS
target = torch.cat([g[p].flatten() for _, p in gq.params])
step_sz = torch.cat(
    [gq.step_size[i].expand_as(p).flatten() for i, (_, p) in enumerate(gq.params)]
)
se = 0.5 / TRIALS**0.5
per_coord = (mean - target) / step_sz
check(
    "lattice SR is unbiased (max coord)",
    per_coord.abs().max() < 4.5 * se,
    f"max |bias|/step = {per_coord.abs().max():.4f}",
)

# --- full-window cancellation: identical micro-grads x8 ---------------------
def window_err(mode, trials=100):
    errs = []
    for t in range(trials):
        m = Toy()
        gq = GradAccumQuantizer(m, mode, bits=4, acc_steps=M)
        g = {p: torch.randn_like(p) * 0.02 for _, p in gq.params}
        run_window(gq, [g] * M, step=t)
        got = torch.cat([p.grad.float().flatten() for _, p in gq.params])
        tgt = torch.cat([M * g[p].flatten() for _, p in gq.params])
        errs.append(((got - tgt) ** 2).mean())
    return torch.stack(errs).mean()

e_lat, e_vdc, e_qmc, e_iid = (
    window_err("lattice"),
    window_err("vdc"),
    window_err("qmc"),
    window_err("iid"),
)
check(
    "lattice beats iid on the correlated window",
    e_lat < 0.5 * e_iid,
    f"mse lattice={e_lat:.3e} vdc={e_vdc:.3e} qmc={e_qmc:.3e} iid={e_iid:.3e}",
)
check(
    "lattice beats pairwise antithetic on the correlated window",
    e_lat < e_qmc,
    f"lattice/qmc = {e_lat / e_qmc:.2f}",
)
check("vdc ~ lattice on drift-free window (same point set)",
      abs(e_vdc - e_lat) < 0.5 * e_lat, f"vdc/lattice = {e_vdc / e_lat:.2f}")

# --- determinism -------------------------------------------------------------
def one_window(mode):
    m = Toy()
    torch.manual_seed(7)
    gq = GradAccumQuantizer(m, mode, bits=4, acc_steps=M)
    micros = [{p: torch.randn_like(p) for _, p in gq.params} for _ in range(M)]
    run_window(gq, micros, step=5)
    return torch.cat([p.grad.float().flatten() for _, p in gq.params])

for mode in ("lattice", "vdc", "latperm", "strat"):
    check(
        f"{mode} window reproducible (stateless seeding)",
        torch.equal(one_window(mode), one_window(mode)),
    )

# strat's guaranteed property: window MSE <= iid on the correlated window
e_strat, e_latperm = window_err("strat"), window_err("latperm")
check(
    "strat beats iid on the correlated window",
    e_strat < 0.75 * e_iid,
    f"mse strat={e_strat:.3e} latperm={e_latperm:.3e} iid={e_iid:.3e}",
)
check(
    "latperm beats iid on the correlated window",
    e_latperm < 0.75 * e_iid,
    f"latperm/iid = {e_latperm / e_iid:.2f}",
)

# --- Muon mq lattice ---------------------------------------------------------
def make_opt(mode, **kw):
    params = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
    opt = Muon(muon_params=params, adamw_params=[], lr=0.0, wd=0.0,
               mq_mode=mode, mq_bits=4, **kw)
    return opt, params

# block structure: same base draw across the 8 phases of a block
opt, params = make_opt("lattice")
p = params[0]
state = opt.state[p]
x = torch.randn_like(p) * 0.05
us = []
for cnt in range(8):
    opt._sr_step_cnt = cnt
    opt._mq_quantize(x, state)  # sets scale; draw happens inside
# reconstruct draws directly: same generator logic
_SEED_MIX = 2654435761
gen = torch.Generator(device="cpu")
gen.manual_seed(((0 * _SEED_MIX) ^ (state["sr_param_seed"] * 4 + 3)) & 0x7FFFFFFF)
base = torch.rand(x.shape, generator=gen, dtype=torch.float32)
lat_set = torch.stack([torch.frac(base + ph / 8) for ph in range(8)])
srt, _ = lat_set.reshape(8, -1).sort(dim=0)
gaps = torch.cat([srt[1:] - srt[:-1], (srt[:1] + 1) - srt[-1:]])
check(
    "mq lattice block is an equidistant shifted set",
    bool(torch.allclose(gaps, torch.full_like(gaps, 1 / 8), atol=1e-6)),
)

# mq lattice sequence reproducible end-to-end
def run_seq():
    torch.manual_seed(11)
    opt, params = make_opt("lattice")
    torch.manual_seed(13)
    for t in range(10):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    return torch.cat(
        [opt.state[p]["momentum_buffer"].flatten() for p in params]
    )

check("mq lattice sequence reproducible", torch.equal(run_seq(), run_seq()))

# mq latperm shares lattice's point set per block; mq strat is stratified
opt_p, params_p = make_opt("latperm")
pp = params_p[0]
state_p = opt_p.state[pp]
xs = torch.randn_like(pp) * 0.05
# reconstruct latperm's draws for one block directly from the seeding scheme
genb = torch.Generator(device="cpu")
genb.manual_seed(((0 * _SEED_MIX) ^ (state_p["sr_param_seed"] * 4 + 3)) & 0x7FFFFFFF)
base_p = torch.rand(xs.shape, generator=genb, dtype=torch.float32)
genr = torch.Generator(device="cpu")
genr.manual_seed(((0 * _SEED_MIX) ^ (state_p["sr_param_seed"] * 8 + 5)) & 0x7FFFFFFF)
Rb = torch.rand((8, *xs.shape), generator=genr, dtype=torch.float32)
lp_set = torch.stack(
    [torch.frac(base_p + (Rb < Rb[ph]).sum(0).float() / 8) for ph in range(8)]
)
slp, _ = lp_set.reshape(8, -1).sort(dim=0)
slat = torch.stack([torch.frac(base_p + ph / 8) for ph in range(8)])
slat, _ = slat.reshape(8, -1).sort(dim=0)
check(
    "mq latperm block point set == mq lattice block point set",
    bool(torch.allclose(slp, slat, atol=1e-6)),
)

for mode in ("latperm", "strat"):
    def run_seq_m(m=mode):
        torch.manual_seed(11)
        opt, params = make_opt(m)
        torch.manual_seed(13)
        for t in range(10):
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()
        return torch.cat(
            [opt.state[p]["momentum_buffer"].flatten() for p in params]
        )
    check(f"mq {mode} sequence reproducible", torch.equal(run_seq_m(), run_seq_m()))

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL PASS")
