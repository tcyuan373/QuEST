import math
import torch


# Copied from models.quantization.base_linear (importing that module pulls in
# fast_hadamard_transform and builds CUDA tensors at import time, so we keep
# muon.py self-contained).
OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}

# Knuth multiplicative hash constant, for mixing (step, param) into a seed.
_SEED_MIX = 2654435761


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        sr_mode="none",  # "none" | "update" | "weight" -- NS-then-round scheme
        sr_bits=4,
        sr_qmc=True,  # antithetic pairing across consecutive optimizer steps
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not

        assert sr_mode in ("none", "update", "weight"), sr_mode
        self.sr_mode = sr_mode
        self.sr_bits = sr_bits
        self.sr_qmc = sr_qmc
        self._sr_step_cnt = 0  # optimizer-step counter driving the QMC sequence

        for idx, p in enumerate(muon_params):
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
            # fixed per-parameter seed offset so QMC streams are independent
            # across parameters but reproducible across the antithetic pair
            self.state[p]["sr_param_seed"] = (idx + 1) * 0x9E3779B1
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def _qmc_sr_round(self, x, param_seed):
        """NS-then-round: stochastically round `x` onto a rowwise Gaussian-optimal
        centered grid (same grid as STEQuantizer in base_linear.py).

        QMC via antithetic pairing across consecutive optimizer steps: step 2k
        draws u from a generator seeded by (pair_id, param), step 2k+1 reuses the
        SAME u flipped to 1-u. With momentum=0.95 consecutive NS outputs are
        strongly correlated, so the linear SR-error terms approximately cancel
        pairwise along the weight trajectory (exact cancellation would require
        identical inputs across the pair, as in weight-quantizer antithetics).
        """
        n_levels = 2 ** self.sr_bits
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.sr_bits]
            * torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True))
            + 1e-8
        )
        step_sz = 2 * scale / (n_levels - 1)
        r = torch.clamp(x.float(), -scale, scale) / step_sz + 0.5

        if self.sr_qmc:
            pair_id, parity = self._sr_step_cnt // 2, self._sr_step_cnt % 2
            g = torch.Generator(device=x.device)
            g.manual_seed(((pair_id * _SEED_MIX) ^ param_seed) & 0x7FFFFFFF)
            u = torch.rand(x.shape, generator=g, device=x.device, dtype=torch.float32)
            if parity == 1:
                u = 1.0 - u
        else:
            u = torch.rand_like(r)

        # centered grid indices q = round(x/step + 1/2) live in
        # [-(n/2 - 1), n/2], NOT [0, n-1] (that clamp collapses all negatives)
        q = torch.floor(r + u).clamp(-(n_levels // 2 - 1), n_levels // 2)
        xq = q * step_sz - step_sz / 2
        return xq.to(x.dtype)

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # NS-then-round: stochastically round the orthogonalized update
                if self.sr_mode == "update":
                    u = self._qmc_sr_round(u, state["sr_param_seed"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

                # alternative: round the master weights after the update
                if self.sr_mode == "weight":
                    p.data.copy_(self._qmc_sr_round(p.data, state["sr_param_seed"]))

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        # advance the QMC/antithetic sequence once per optimizer step
        self._sr_step_cnt += 1

        return loss
