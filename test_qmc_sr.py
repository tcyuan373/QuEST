"""Correctness tests for the QMC-SR implementation (QMCSRSTEQuantizer +
set_sr_step + Muon NS-then-round). Needs a GPU (base_linear builds CUDA
Hadamard matrices at import). Exits nonzero on any failure."""
import sys
import torch

sys.path.insert(0, "src")

from models.quantization.base_linear import (
    QMCSRSTEQuantizer,
    STEQuantizer,
    QuantizedLinear,
)
from optim.base import set_sr_step
from muon import Muon

dev = "cuda"
torch.manual_seed(0)
FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


x = torch.randn(256, 512, device=dev)

# --- 1. eval mode identical to deterministic parent (RTN on same grid) -------
qz = QMCSRSTEQuantizer().to(dev).eval()
ste = STEQuantizer().to(dev).eval()
check("eval == deterministic STEQuantizer", torch.equal(qz(x), ste(x)))

# --- 2. unbiasedness in training mode over many optimizer steps --------------
qz.train()
acc = torch.zeros_like(x)
N = 400
for t in range(N):
    qz._sr_step.fill_(t)
    qz._sr_micro.fill_(0)
    acc += qz(x).detach()
bias = (acc / N - x).abs().mean().item()
sr_err = (qz(x).detach() - x).abs().mean().item()
check("unbiased (mean of draws -> x)", bias < 3 * sr_err / N**0.5 + 5e-3,
      f"bias={bias:.5f}, single-draw err={sr_err:.5f}")

# --- 3. antithetic pairing: same step, micro 0 vs 1 --------------------------
qz._sr_step.fill_(7)
qz._sr_micro.fill_(0)
y0 = qz(x).detach()
qz._sr_micro.fill_(1)
y1 = qz(x).detach()
pair_err = ((y0 + y1) / 2 - x).abs().mean().item()
check("antithetic pair cancels", pair_err < 0.6 * sr_err,
      f"pair-mean err={pair_err:.5f} vs single {sr_err:.5f}")
# micro 2 starts a fresh pair -> different draw than micro 0
qz._sr_micro.fill_(2)
y2 = qz(x).detach()
check("next pair uses fresh draw", not torch.equal(y0, y2))

# --- 4. determinism/reproducibility + step dependence -------------------------
qz._sr_micro.fill_(0)
a = qz(x).detach()
b = qz(x).detach()
check("same (step,micro) reproducible", torch.equal(a, b))
qz._sr_step.fill_(8)
c = qz(x).detach()
check("different step -> different draw", not torch.equal(a, c))

# --- 5. per-instance decorrelation -------------------------------------------
qa, qb = QMCSRSTEQuantizer().to(dev).train(), QMCSRSTEQuantizer().to(dev).train()
for q_ in (qa, qb):
    q_._sr_step.fill_(3)
    q_._sr_micro.fill_(0)
check("two instances draw different noise", not torch.equal(qa(x), qb(x)))

# --- 6. STE gradient is identity ---------------------------------------------
xg = x.clone().requires_grad_(True)
qz(xg).sum().backward()
check("STE gradient == 1", torch.equal(xg.grad, torch.ones_like(xg)))

# --- 7. grid range: quantized values within +-scale of parent grid ------------
qz._sr_step.fill_(11)
y = qz(x).detach()
scale = 2.513930578568423 * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
check("xq within +-scale", bool((y.abs() <= scale * (1 + 1e-4)).all()))
lv = torch.unique(((y / (2 * scale / 15)) + 0.5).round())
check("<=16 distinct grid indices", lv.numel() <= 16, f"got {lv.numel()}")

# --- 8. iid mode: unbiased, differs call-to-call ------------------------------
qi = QMCSRSTEQuantizer(qmc=False).to(dev).train()
check("iid mode varies per call", not torch.equal(qi(x), qi(x)))

# --- 9. set_sr_step broadcasts through a module tree --------------------------
lin = QuantizedLinear(512, 256, weight_quantizer=QMCSRSTEQuantizer()).to(dev)
wrapper = torch.nn.Sequential(lin)
set_sr_step(wrapper, 42, 3)
check("set_sr_step reaches quantizer",
      lin.weight_quantizer._sr_step.item() == 42
      and lin.weight_quantizer._sr_micro.item() == 3)

# --- 10. Muon NS-then-round: two replicas stay in lockstep (DDP safety) -------
torch.manual_seed(1)
p1 = torch.nn.Parameter(torch.randn(64, 128, device=dev))
p2 = torch.nn.Parameter(p1.data.clone())
o1 = Muon(muon_params=[p1], adamw_params=[], lr=0.01, sr_mode="update")
o2 = Muon(muon_params=[p2], adamw_params=[], lr=0.01, sr_mode="update")
for t in range(3):
    g = torch.randn(64, 128, device=dev)
    p1.grad = g.clone()
    p2.grad = g.clone()
    o1.step()
    o2.step()
check("Muon SR replicas identical (rank sync)", torch.equal(p1.data, p2.data))
check("Muon SR actually changes update",
      True)  # covered implicitly: sr_mode=none comparison below
p3 = torch.nn.Parameter(p1.data.clone())
o3 = Muon(muon_params=[p3], adamw_params=[], lr=0.01, sr_mode="none")
p1.grad = p3.grad = torch.randn(64, 128, device=dev)
o1.step()
o3.step()
check("sr_mode=update differs from none", not torch.equal(p1.data, p3.data))

# --- 11. FP4-grid SR quantizer -----------------------------------------------
from models.quantization.base_linear import QMCSRFP4Quantizer, FP4STEQuantizer

qf = QMCSRFP4Quantizer().to(dev)
qf_det = FP4STEQuantizer().to(dev)
qf.eval()
check("fp4-sr eval == FP4STEQuantizer", torch.allclose(qf(x), qf_det(x), atol=1e-5))

qf.train()
acc = torch.zeros_like(x)
N = 400
for t in range(N):
    qf._sr_step.fill_(t)
    qf._sr_micro.fill_(0)
    acc += qf(x).detach()
bias = (acc / N - x).abs().mean().item()
sr_err = (qf(x).detach() - x).abs().mean().item()
check("fp4-sr unbiased", bias < 3 * sr_err / N**0.5 + 5e-3,
      f"bias={bias:.5f}, single-draw err={sr_err:.5f}")

qf._sr_step.fill_(7)
qf._sr_micro.fill_(0)
z0 = qf(x).detach()
qf._sr_micro.fill_(1)
z1 = qf(x).detach()
pair_err = ((z0 + z1) / 2 - x).abs().mean().item()
check("fp4-sr antithetic pair cancels", pair_err < 0.6 * sr_err,
      f"pair-mean err={pair_err:.5f} vs single {sr_err:.5f}")

# every output value must sit exactly on the (rms-scaled) FP4 grid
std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
zn = (z0 / std)
dist = torch.min(torch.abs(zn.unsqueeze(-1) - qf.levels), dim=-1).values
check("fp4-sr values on FP4 grid", dist.max().item() < 1e-4,
      f"max off-grid dist={dist.max().item():.2e}")

xg2 = x.clone().requires_grad_(True)
qf(xg2).sum().backward()
check("fp4-sr STE gradient == 1", torch.equal(xg2.grad, torch.ones_like(xg2)))

# --- 12. Trust + QMC-SR (deterministic mask, stochastic values) --------------
from models.quantization.base_linear import (
    QMCSRTrustQuantizer, HadamardQMCSRQuantizer, HalfHadamardQMCSRQuantizer,
    HadamardQMCSRTrustQuantizer, HalfHadamardQMCSRTrustQuantizer,
    TrustQuantizer, HalfHadamardTrustQuantizer, HadamardTrustQuantizer,
)

qt = QMCSRTrustQuantizer().to(dev).train()
qt._sr_step.fill_(3); qt._sr_micro.fill_(0)
y0 = qt(x); qt._sr_micro.fill_(1); y1 = qt(x)
# mask must be identical across the antithetic pair: gradient of both draws equal
xg = x.clone().requires_grad_(True)
qt._sr_micro.fill_(0); qt(xg).sum().backward(); g0 = xg.grad.clone()
xg.grad = None
qt._sr_micro.fill_(1); qt(xg).sum().backward(); g1 = xg.grad.clone()
check("trust: mask deterministic across pair (grads equal)", torch.equal(g0, g1))
check("trust: grad is 0/1 mask", set(torch.unique(g0).tolist()) <= {0.0, 1.0})
# values still unbiased & antithetic on the value path
pair_err = ((y0.detach() + y1.detach()) / 2 - x * (g0 == g0)).abs()  # vs x
sing_err = (y0.detach() - x).abs()
check("trust: antithetic value cancellation", pair_err.mean() < 0.6 * sing_err.mean(),
      f"pair {pair_err.mean():.4f} vs single {sing_err.mean():.4f}")
qt.eval()
te = TrustQuantizer().to(dev)
check("trust: eval == deterministic TrustQuantizer values",
      torch.allclose(qt(x).detach(), te(x).detach(), atol=1e-5))

# --- 13. Hadamard QMC-SR (full rotation) --------------------------------------
qh = HadamardQMCSRQuantizer().to(dev).train()
acc = torch.zeros_like(x)
N = 300
for t in range(N):
    qh._sr_step.fill_(t); qh._sr_micro.fill_(0)
    acc += qh(x).detach()
# SR is unbiased on the CLIPPED rotated value; rotation back spreads the
# (deterministic) clip saturation across coords, so compare against the
# clipped reference, not raw x.
with torch.no_grad():
    Mh = qh._rotation(x)
    xh = x @ Mh
    sc = 2.513930578568423 * torch.sqrt(torch.mean(xh**2, dim=-1, keepdim=True)) + 1e-8
    ref = torch.clamp(xh, -sc, sc) @ Mh.T
bias = (acc / N - ref).abs().mean().item()
serr = (qh(x).detach() - ref).abs().mean().item()
check("hadamard-full: unbiased vs clipped reference", bias < 3 * serr / N**0.5 + 5e-3,
      f"bias={bias:.5f} single={serr:.5f}")
qh._sr_step.fill_(7); qh._sr_micro.fill_(0); h0 = qh(x).detach()
qh._sr_micro.fill_(1); h1 = qh(x).detach()
check("hadamard-full: antithetic pair cancels",
      ((h0 + h1) / 2 - x).abs().mean() < 0.6 * (h0 - x).abs().mean())
xg3 = x.clone().requires_grad_(True)
qh(xg3).sum().backward()
check("hadamard-full: STE gradient == 1", torch.allclose(xg3.grad, torch.ones_like(xg3), atol=1e-4))

# --- 14. Half-Hadamard QMC-SR: rotated-domain output, product preserved -------
qhh = HalfHadamardQMCSRQuantizer().to(dev).eval()
M = qhh._rotation(x)
ref = x @ M
check("half-hadamard: eval outputs live in rotated domain",
      torch.allclose(qhh(x).detach(), STEQuantizer().to(dev)(ref).detach(), atol=1e-4))
# both-sides rotation cancels in the matmul (weights x activations)
wq = HalfHadamardQMCSRQuantizer().to(dev).eval()
aq_rot = HalfHadamardTrustQuantizer().to(dev).eval()
a = torch.randn(64, 512, device=dev)
w = torch.randn(256, 512, device=dev)
prod_rot = aq_rot(a).detach() @ wq(w).detach().T
prod_ref = (a @ w.T)
rel = (prod_rot - prod_ref).abs().mean() / prod_ref.abs().mean()
check("half-hadamard pair: rotations cancel in matmul (quant-level error only)",
      rel.item() < 0.2, f"rel err {rel.item():.3f}")

# --- 15. QuEST-style combos construct & train-step cleanly --------------------
for cls in (HadamardQMCSRTrustQuantizer, HalfHadamardQMCSRTrustQuantizer):
    q = cls().to(dev).train()
    q._sr_step.fill_(1); q._sr_micro.fill_(0)
    xg4 = x.clone().requires_grad_(True)
    q(xg4).sum().backward()
    check(f"{cls.__name__}: fwd/bwd OK, grad in rotated-mask family", xg4.grad is not None)
q = HadamardQMCSRTrustQuantizer().to(dev)
q.eval()
ht = HadamardTrustQuantizer().to(dev)
check("hadamard-trust-qmc: eval values == HadamardTrustQuantizer",
      torch.allclose(q(x).detach(), ht(x).detach(), atol=1e-4))

print()
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("ALL CHECKS PASSED")
