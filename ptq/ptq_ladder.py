"""PTQ ladder on a trained QuEST-repo checkpoint.

Rungs (each isolates one ingredient, all on the SAME rowwise centered uniform
grid used by the QAT STEQuantizer family — OPTIMAL_GAUSSIAN_SCALES[bits] * rms):

  rtn       round-to-nearest, no error feedback
  ldlq      GPTQ/LDLQ: sequential columns w/ Hessian error feedback, det rounding
  ldlq_iid  LDLQ with iid stochastic rounding
  ldlq_qmc  LDLQ with ANTITHETIC stochastic rounding: column pairs (2t, 2t+1)
            share a per-row uniform draw u vs 1-u, so error-feedback-coupled
            adjacent columns get negatively correlated rounding noise
  tcq       (phase 2) trellis-coded quantization inside LDLQ, cf. QTIP

Hessian proxy per linear layer: H = sum X^T X over calibration batches
(train.bin windows), GPTQ-style 1% damping. Eval: fixed sequential windows of
val.bin, same CE the training loop reports.

Usage (GPU node):
  python ptq/ptq_ladder.py --exp-dir exps/qmc_c4_fp16_50M_job827911 \
      --dataset-dir datasets_full/c4slice --bits 4 3 2 \
      --methods rtn ldlq ldlq_iid ldlq_qmc --sr-seeds 3
"""

import argparse
import copy
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.utils import get_model  # noqa: E402
from models.quantization.base_linear import OPTIMAL_GAUSSIAN_SCALES  # noqa: E402


# ---------------------------------------------------------------- model + data


def load_model(exp_dir, device):
    with open(os.path.join(exp_dir, "summary.json")) as f:
        args = types.SimpleNamespace(**json.load(f)["args"])
    model = get_model(args)
    ck = torch.load(
        os.path.join(exp_dir, "ckpts", "latest", "main.pt"),
        map_location="cpu",
        weights_only=False,
    )
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, args


def batches(bin_path, n_batches, bs, seq, offset=0):
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    for b in range(n_batches):
        start = offset + b * bs * (seq + 1)
        chunk = torch.from_numpy(
            data[start:start + bs * (seq + 1)].astype(np.int64)
        ).view(bs, seq + 1)
        yield chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, val_bin, n_batches, bs, seq, device):
    losses = []
    for x, y in batches(val_bin, n_batches, bs, seq):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device != "cpu")):
            out = model(x, targets=y)
        losses.append(out["loss"].item())
    return float(np.mean(losses))


def target_linears(model):
    """All transformer-block linear layers (skip embeddings / lm_head)."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".h." in name:
            out[name] = mod
    return out


@torch.no_grad()
def collect_hessians(model, layers, train_bin, n_batches, bs, seq, device):
    """H = sum over calib tokens of x x^T per layer (float32, in_features^2)."""
    H = {n: torch.zeros(m.in_features, m.in_features, device=device)
         for n, m in layers.items()}
    hooks = []

    def mk(name):
        def hook(mod, inp, _out):
            x = inp[0].reshape(-1, mod.in_features).float()
            H[name].add_(x.T @ x)
        return hook

    for n, m in layers.items():
        hooks.append(m.register_forward_hook(mk(n)))
    for x, y in batches(train_bin, n_batches, bs, seq):
        x = x.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device != "cpu")):
            model(x)
    for h in hooks:
        h.remove()
    return H


# ---------------------------------------------------------------- quantizers


def row_grid(W, bits):
    """QAT-consistent rowwise grid: scale = OPT[bits]*rms(row), centered."""
    scale = OPTIMAL_GAUSSIAN_SCALES[bits] * W.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8
    step = 2 * scale / (2 ** bits - 1)
    return scale, step


def q_col(w, scale, step, bits, u=None):
    """Quantize one column (values w, per-row scale/step). u=None -> det RTN;
    else stochastic with per-row uniforms u."""
    w_c = torch.clamp(w, -scale, scale)
    t = w_c / step + 0.5          # continuous level index
    if u is None:
        idx = torch.round(t)
    else:
        lo = torch.floor(t)
        idx = lo + (u < (t - lo)).to(t.dtype)
    # centered grid: t in [1 - 2^(b-1), 2^(b-1)] (e.g. [-7, 8] at 4 bits);
    # clamping to [0, 2^b - 1] here is the legacy StoRounding bug that
    # collapses all negatives to -step/2
    idx = torch.clamp(idx, 1 - 2 ** (bits - 1), 2 ** (bits - 1))
    return idx * step - step / 2


def rtn(W, H, bits, gen=None, mode="det"):
    scale, step = row_grid(W, bits)
    scale, step = scale.squeeze(1), step.squeeze(1)
    out = torch.empty_like(W)
    for j in range(W.shape[1]):
        out[:, j] = q_col(W[:, j], scale, step, bits)
    return out


def ldlq(W, H, bits, gen=None, mode="det"):
    """GPTQ-form LDLQ: natural column order, Cholesky-of-inverse feedback.
    mode: det | iid | qmc (antithetic column pairs, u vs 1-u)."""
    m, n = W.shape
    W = W.clone().float()
    Hd = H.double()
    Hd = 0.5 * (Hd + Hd.T)  # float accumulation leaves tiny asymmetry
    dead = torch.diag(Hd) == 0
    Hd[dead, dead] = 1.0
    W[:, dead] = 0
    eye = torch.eye(n, device=Hd.device, dtype=Hd.dtype)
    mean_diag = torch.diag(Hd).mean()
    U = None
    for damp in (0.01, 0.1, 1.0):
        try:
            Hc = Hd + damp * mean_diag * eye
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hc))
            Hinv = 0.5 * (Hinv + Hinv.T)
            U = torch.linalg.cholesky(Hinv, upper=True)
            break
        except torch._C._LinAlgError:
            continue
    if U is None:
        raise RuntimeError("H not factorizable even at damp=1.0")
    U = U.float()  # upper triangular

    scale, step = row_grid(W, bits)
    scale, step = scale.squeeze(1), step.squeeze(1)
    Q = torch.empty_like(W)
    u_prev = None
    for j in range(n):
        if mode == "det":
            u = None
        elif mode == "iid":
            u = torch.rand(m, device=W.device, generator=gen)
        elif mode == "qmc":
            if j % 2 == 0:
                u_prev = torch.rand(m, device=W.device, generator=gen)
                u = u_prev
            else:
                u = 1.0 - u_prev
        else:
            raise ValueError(mode)
        q = q_col(W[:, j], scale, step, bits, u)
        Q[:, j] = q
        err = (W[:, j] - q) / U[j, j]
        W[:, j + 1:] -= err.unsqueeze(1) * U[j, j + 1:].unsqueeze(0)
    return Q


METHODS = {
    "rtn": (rtn, "det"),
    "ldlq": (ldlq, "det"),
    "ldlq_iid": (ldlq, "iid"),
    "ldlq_qmc": (ldlq, "qmc"),
}


# ---------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--dataset-dir", default="datasets_full/c4slice")
    ap.add_argument("--methods", nargs="+", default=["rtn", "ldlq", "ldlq_iid", "ldlq_qmc"])
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 3, 2])
    ap.add_argument("--calib-batches", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--sr-seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "tcq" in args.methods:
        raise NotImplementedError("tcq rung is phase 2 -- see /home/ty373/workspace/qtip")

    model, margs = load_model(args.exp_dir, device)
    layers = target_linears(model)
    train_bin = os.path.join(args.dataset_dir, "train.bin")
    val_bin = os.path.join(args.dataset_dir, "val.bin")

    fp_loss = evaluate(model, val_bin, args.eval_batches, args.batch_size, args.seq, device)
    print(f"[fp16 reference] val_loss={fp_loss:.4f}", flush=True)

    print("collecting Hessians...", flush=True)
    H = collect_hessians(model, layers, train_bin, args.calib_batches,
                         args.batch_size, args.seq, device)

    orig = {n: l.weight.data.clone() for n, l in layers.items()}
    results = {"fp16": fp_loss, "config": vars(args)}
    for bits in args.bits:
        for meth in args.methods:
            fn, mode = METHODS[meth]
            seeds = range(args.sr_seeds) if mode != "det" else [0]
            losses = []
            for seed in seeds:
                gen = torch.Generator(device=device).manual_seed(1000 + seed)
                for n, l in layers.items():
                    l.weight.data = fn(orig[n].float(), H[n], bits,
                                       gen=gen, mode=mode).to(orig[n].dtype)
                loss = evaluate(model, val_bin, args.eval_batches,
                                args.batch_size, args.seq, device)
                losses.append(loss)
                print(f"bits={bits} {meth} seed={seed}: val_loss={loss:.4f}", flush=True)
            results[f"b{bits}_{meth}"] = losses
    for n, l in layers.items():
        l.weight.data = orig[n]

    out = args.out or os.path.join("ptq", "ptq_ladder_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("\n=== SUMMARY (val_loss, mean over seeds) ===")
    print(f"fp16: {fp_loss:.4f}")
    for k, v in results.items():
        if k.startswith("b"):
            arr = np.array(v)
            print(f"{k}: {arr.mean():.4f}" + (f" +- {arr.std():.4f}" if len(arr) > 1 else ""))


if __name__ == "__main__":
    main()
