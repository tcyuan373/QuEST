"""Error-feedback correctness tests (gquant ef / Muon mq_ef). CPU-safe.

The load-bearing property is the EF telescoping invariant:
    stored + residual == exact uncompensated value   (exactly in fp32;
up to residual-storage rounding in fp16), which holds regardless of
clamping because r <- pre - stored absorbs clamp error too.
Exits nonzero on any failure.
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


M = 8

# --- gquant EF invariant: accum + resid == shadow (+ window-start resid) -----
for mode in ("det", "iid"):
    for ef in ("fp32", "fp16"):
        torch.manual_seed(5)
        gq = GradAccumQuantizer(Toy(), mode, bits=4, acc_steps=M, ef=ef)
        gs = {p: torch.randn_like(p) * 0.3 for _, p in gq.params}
        for micro in range(M):
            for (_, p) in gq.params:
                p.grad = gs[p].clone()
            gq.accumulate(0, micro)
        # invariant BEFORE write_back frees the shadow
        atol = 1e-4 if ef == "fp32" else 0.05
        inv = all(
            torch.allclose(
                gq.accum[i] + gq.resid[i].float(), gq.shadow[i], atol=atol
            )
            for i, (_, p) in enumerate(gq.params)
        )
        gq.write_back()
        check(f"gquant {mode}+ef-{ef} invariant accum+resid==exact", inv)

# --- EF recovers swamped mass across windows (det, the canonical failure) ----
def run_windows(gq, big, tiny, n_windows):
    delivered = None
    for w in range(n_windows):
        for micro in range(M):
            for (_, p) in gq.params:
                p.grad = (big[p] if micro == 0 else tiny[p]).clone()
            gq.accumulate(w, micro)
        gq.write_back()
        d = {p: p.grad.float().clone() for (_, p) in gq.params}
        delivered = d if delivered is None else {p: delivered[p] + d[p] for p in d}
    return delivered


W = 5
errs = {}
for ef in ("none", "fp32"):
    torch.manual_seed(21)
    gq = GradAccumQuantizer(Toy(), "det", bits=4, acc_steps=M, ef=ef)
    big = {p: torch.randn_like(p) for _, p in gq.params}
    tiny = {p: torch.full_like(p, 0.1) for _, p in gq.params}
    got = run_windows(gq, big, tiny, W)
    exact = {p: W * (big[p] + (M - 1) * tiny[p]) for _, p in gq.params}
    errs[ef] = sum(float((got[p] - exact[p]).abs().sum()) for _, p in gq.params) / sum(
        p.numel() for _, p in gq.params
    )
check(
    "det+EF recovers swamped micro-grad mass across windows",
    errs["fp32"] < 0.5 * errs["none"],
    f"mean |err|: ef {errs['fp32']:.3f} vs none {errs['none']:.3f}",
)

# --- gquant ef=none leaves no residual state (default path untouched) --------
gq0 = GradAccumQuantizer(Toy(), "iid", bits=4, acc_steps=M)
check("gquant ef=none allocates no residual", all(r is None for r in gq0.resid))

# --- Muon mq EF invariant: buf + resid tracks the exact fp32 recursion -------
def make_opt(mq_ef, mode="det"):
    torch.manual_seed(11)
    params = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
    opt = Muon(
        muon_params=params, adamw_params=[], lr=0.0, wd=0.0,
        mq_mode=mode, mq_bits=4, mq_ef=mq_ef,
    )
    return opt, params


for mode in ("det", "iid"):
    for ef in ("fp32", "fp16"):
        opt, params = make_opt(ef, mode)
        shadow = {p: torch.zeros_like(p) for p in params}
        torch.manual_seed(13)
        ok = True
        for t in range(5):
            grads = {p: torch.randn_like(p) for p in params}
            for p in params:
                p.grad = grads[p].clone()
                shadow[p] = 0.95 * shadow[p] + grads[p]
            opt.step()
            atol = 1e-4 if ef == "fp32" else 0.05
            for p in params:
                st = opt.state[p]
                ok = ok and torch.allclose(
                    st["momentum_buffer"].float() + st["mq_resid"].float(),
                    shadow[p], atol=atol,
                )
        check(f"mq {mode}+ef-{ef} invariant buf+resid==exact recursion", ok)

# --- bf16 buffer + EF: master must track the EXACT fp32 recursion ------------
# (the compensated master is formed in fp32 from prev_stored + r, so bf16
# buffer arithmetic never contaminates it)
torch.manual_seed(11)
params_be = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
opt_be = Muon(
    muon_params=params_be, adamw_params=[], lr=0.0, wd=0.0,
    mq_mode="det", mq_bits=4, mq_ef="fp32", buf_dtype="bf16",
)
shadow_be = {p: torch.zeros_like(p) for p in params_be}
torch.manual_seed(13)
ok = True
for t in range(5):
    for p in params_be:
        g_ = torch.randn_like(p)
        p.grad = g_.clone()
        shadow_be[p] = 0.95 * shadow_be[p] + g_
    opt_be.step()
    for p in params_be:
        st = opt_be.state[p]
        ok = ok and torch.allclose(
            st["momentum_buffer"].float() + st["mq_resid"].float(),
            shadow_be[p], atol=1e-4,
        )
check("bf16 buffer + fp32 EF: master tracks exact fp32 recursion", ok)

# --- residual-backlog logging (the runaway watch) ----------------------------
gq_r = GradAccumQuantizer(Toy(), "det", bits=4, acc_steps=M, ef="fp32")
gr = {p: torch.randn_like(p) * 0.3 for _, p in gq_r.params}
for micro in range(M):
    for (_, p) in gq_r.params:
        p.grad = gr[p].clone()
    gq_r.accumulate(0, micro)
gq_r.write_back()
mr = gq_r.mech_summary()
check(
    "gq EF logs residual norms (backlog watch)",
    "gq_resid_ms" in mr and mr["gq_resid_ms"] >= 0.0 and "gq_resid_max" in mr,
    f"resid_ms={mr.get('gq_resid_ms')}, resid_max={mr.get('gq_resid_max')}",
)
gq_n = GradAccumQuantizer(Toy(), "det", bits=4, acc_steps=M)
for micro in range(M):
    for (_, p) in gq_n.params:
        p.grad = gr[p].clone() if p in gr else torch.randn_like(p)
    gq_n.accumulate(0, micro)
gq_n.write_back()
check("no resid keys when ef=none", "gq_resid_ms" not in gq_n.mech_summary())
opt_rl, params_rl = make_opt("fp32", "det")
for p in params_rl:
    p.grad = torch.randn_like(p)
opt_rl.step()
mrl = opt_rl.mq_mech_summary()
check(
    "mq EF logs residual norms",
    "mq_resid_ms" in mrl and mrl["mq_resid_ms"] >= 0.0,
)

# --- mq residual lives in optimizer state (checkpointed -> resume-safe) ------
opt, params = make_opt("fp32")
for p in params:
    p.grad = torch.randn_like(p)
opt.step()
sd = opt.state_dict()
check(
    "mq_resid is checkpointed via state_dict",
    any("mq_resid" in v for v in sd["state"].values()),
)
opt2, params2 = make_opt("fp32")
for p in params2:
    p.grad = torch.randn_like(p)
opt2.step()
opt2.load_state_dict(sd)
same = all(
    torch.equal(opt2.state[p2]["mq_resid"], opt.state[p1]["mq_resid"])
    for p1, p2 in zip(params, params2)
)
check("mq_resid round-trips through load_state_dict", same)

# --- bf16 buffer-storage knob (the 2 B/param Pareto anchor row) --------------
torch.manual_seed(11)
params_b = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
opt_b = Muon(
    muon_params=params_b, adamw_params=[], lr=0.0, wd=0.0, buf_dtype="bf16"
)
torch.manual_seed(13)
for t in range(3):
    for p in params_b:
        p.grad = torch.randn_like(p)
    opt_b.step()
check(
    "buf_dtype=bf16 stores the momentum buffer in bfloat16",
    all(opt_b.state[p]["momentum_buffer"].dtype == torch.bfloat16 for p in params_b),
)
sd_b = opt_b.state_dict()
check(
    "bf16 buffer round-trips state_dict with dtype",
    all(
        v["momentum_buffer"].dtype == torch.bfloat16
        for v in sd_b["state"].values()
        if "momentum_buffer" in v
    ),
)
# THE resume trap: load_state_dict casts state to the PARAM dtype (fp32), so
# without the re-cast guard the anchor arm silently becomes fp32 after any
# preemption resume. Assert the POST-load, post-step dtype.
torch.manual_seed(11)
params_r = [torch.nn.Parameter(torch.randn(16, 32)) for _ in range(2)]
opt_r = Muon(muon_params=params_r, adamw_params=[], lr=0.0, wd=0.0, buf_dtype="bf16")
opt_r.load_state_dict(sd_b)
for p in params_r:
    p.grad = torch.randn_like(p)
opt_r.step()
check(
    "bf16 buffer stays bf16 AFTER load_state_dict + step (resume trap)",
    all(opt_r.state[p]["momentum_buffer"].dtype == torch.bfloat16 for p in params_r),
)
# bf16 buffer tracks the fp32 recursion to bf16 resolution
shadow_b = {p: torch.zeros_like(p) for p in params_b}
torch.manual_seed(13)
opt_b2, = [Muon(muon_params=params_b, adamw_params=[], lr=0.0, wd=0.0, buf_dtype="bf16")]
torch.manual_seed(13)
for t in range(3):
    for p in params_b:
        g_ = torch.randn_like(p)
        p.grad = g_.clone()
        shadow_b[p] = 0.95 * shadow_b[p] + g_
    opt_b2.step()
check(
    "bf16 buffer tracks fp32 recursion to bf16 resolution",
    all(
        torch.allclose(
            opt_b2.state[p]["momentum_buffer"].float(), shadow_b[p],
            rtol=0.02, atol=0.05,
        )
        for p in params_b
    ),
)

# --- default-off leaves no residual state ------------------------------------
opt0, params0 = make_opt("none")
for p in params0:
    p.grad = torch.randn_like(p)
opt0.step()
check(
    "mq ef=none stores no residual",
    all("mq_resid" not in opt0.state[p] for p in params0),
)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1)
print("ALL PASS")
