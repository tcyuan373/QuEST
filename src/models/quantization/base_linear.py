import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform
from torch.autograd import Function

class ClippedSTE(Function):
    @staticmethod
    def forward(ctx, x, xq, x_min, x_max):
        ctx.save_for_backward(x, x_min, x_max)
        return xq

    @staticmethod
    def backward(ctx, grad_output):
        x, x_min, x_max = ctx.saved_tensors
        mask = (x >= x_min) & (x <= x_max)
        grad_x = grad_output * mask
        # no grad for xq, x_min, x_max
        return grad_x, None, None, None


class BaseQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x


class UniformQuantizer(BaseQuantizer):
    def forward(self, x):
        if not self.training:
            return x
        scale = torch.max(torch.abs(x), dim=-1, keepdim=True) + 1e-8
        step = scale * 2 / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        return x + (xq - x).detach()


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


class STEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, centered=True):
        super().__init__(bits)
        self.centered = centered

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, -scale * (self.n_levels - 2) / self.n_levels, scale)
            xq = torch.round(x_clip / step) * step

        return x + (xq - x).detach()


class ClipQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.clip_scale = clip_scale

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(x) <= scale * self.clip_scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = (
                (neg_scale * self.clip_scale <= x) & (x <= scale * self.clip_scale)
            ).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardClipQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardClipQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardTrustQuantizer(STEQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True)
        self.matrix = None
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class TrustQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, trust=None):
        super().__init__(bits, centered)

        # in terms of std
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HadamardTrustQuantizer(TrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, True, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class GaussianSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4):
        super().__init__(bits)
        self.register_buffer("levels", self._compute_gaussian_levels())

    def _compute_gaussian_levels(self):
        levels = np.linspace(-3, 3, self.n_levels)
        boundaries = np.zeros(self.n_levels + 1)

        for _ in range(20):
            boundaries[1:-1] = (levels[1:] + levels[:-1]) / 2
            boundaries[0] = -float("inf")
            boundaries[-1] = float("inf")

            new_levels = []
            for i in range(self.n_levels):
                b_left, b_right = boundaries[i], boundaries[i + 1]

                def f(x):
                    return x * norm.pdf(x)

                integral_num = integrate.quad(f, b_left, b_right)[0]
                integral_den = integrate.quad(norm.pdf, b_left, b_right)[0]
                if integral_den > 1e-10:
                    new_levels.append(integral_num / integral_den)
                else:
                    new_levels.append(levels[i])
            levels = np.array(new_levels)
        return torch.tensor(levels, dtype=torch.float32)

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        return x + (xq - x).detach()


class GaussianClipQuantizer(GaussianSTEQuantizer):
    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (x_norm.abs() <= self.levels[-1]).float()
        return x * mask + (xq - x * mask).detach()


class GaussianTrustQuantizer(GaussianSTEQuantizer):
    def __init__(self, bits=4, trust=None):
        super().__init__(bits)
        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            xq = xq @ self.matrix.T
            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


FP4_LEVELS = [
    -2.92247856,
    -1.94831904,
    -1.46123928,
    -0.97415952,
    -0.73061964,
    -0.48707976,
    -0.24353988,
    0.0,
    0.0,
    0.24353988,
    0.48707976,
    0.73061964,
    0.97415952,
    1.46123928,
    1.94831904,
    2.92247856,
]


class FP4STEQuantizer(GaussianSTEQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))


class FP4ClipQuantizer(GaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class FP4TrustQuantizer(GaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HalfHadamardFP4ClipQuantizer(HalfHadamardGaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HadamardFP4ClipQuantizer(HadamardGaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )


class HalfHadamardFP4TrustQuantizer(HalfHadamardGaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )

        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2


class HadamardFP4TrustQuantizer(HadamardGaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer(
            "levels",
            torch.tensor(
                [
                    -2.92247856,
                    -1.94831904,
                    -1.46123928,
                    -0.97415952,
                    -0.73061964,
                    -0.48707976,
                    -0.24353988,
                    0.0,
                    0.0,
                    0.24353988,
                    0.48707976,
                    0.73061964,
                    0.97415952,
                    1.46123928,
                    1.94831904,
                    2.92247856,
                ]
            ),
        )

        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2


class FourEightMaskedQuantizer(BaseQuantizer):
    def __init__(self, p=2.0):
        super().__init__(16)
        self.p = p

    def forward(self, x):
        x_reshaped = x.reshape(-1, 4, 2)
        _, idx = x_reshaped.norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        mask = torch.ones_like(x_reshaped, dtype=torch.bool)
        mask[torch.arange(x_reshaped.size(0)).repeat(2, 1).T, idx, :] = False
        mask = mask.reshape(x.shape).float()

        return x * mask


class FourEightSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, p: float = 2.0):
        super().__init__(bits)
        self.p = p

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        return x + (xq - x).detach()


class FourEightClipQuantizer(FourEightSTEQuantizer):
    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        mask = (torch.abs(x) <= scale).float()
        return x * mask + (xq - x * mask).detach()


class FourEightTrustQuantizer(FourEightSTEQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, p)
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, trust)
        self.p = p

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4, 2)
                .norm(p=self.p, dim=-1)
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4, 2)
            xq[
                torch.arange(xq.size(0)).repeat(2, 1).T,
                idx,
            ] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, trust)
        self.p = p

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4, 2)
                .norm(p=self.p, dim=-1)
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4, 2)
            xq[
                torch.arange(xq.size(0)).repeat(2, 1).T,
                idx,
            ] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


# torch._dynamo.config.optimize_ddp=False # uncommend if actually using ErfClipQuantizer
class ErfFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xq, buffer, mask):
        ctx.save_for_backward(buffer, mask)
        return xq

    @staticmethod
    def backward(ctx, grad_output):
        buffer, mask = ctx.saved_tensors
        mask = mask.float()

        return (
            (grad_output + buffer) * mask,
            None,
            grad_output * (1 - mask) - buffer * mask,
            None,
        )


class ErfClipQuantizer(ClipQuantizer):
    def __init__(self, bits=4, acc_dtype=torch.float32):
        super().__init__(bits, True)
        self.acc_dtype = acc_dtype
        self.register_parameter("acc", None)

    def forward(self, x):
        with torch.no_grad():
            if self.acc is None:
                self.acc = nn.Parameter(
                    torch.zeros_like(x, dtype=self.acc_dtype), requires_grad=True
                )
            elif self.acc.grad is not None:
                self.acc.data += self.acc.grad
                self.acc.grad = None

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        mask = (torch.abs(x) <= scale).float()

        return ErfFn().apply(x, xq, self.acc, mask)


class FlushAccFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, acc):
        ctx.save_for_backward(acc)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (acc,) = ctx.saved_tensors
        return grad_output + acc, None


class ClipAccQuantizer(STEQuantizer):
    def __init__(
        self,
        bits=4,
        centered=True,
        flush_every: int = 64,
        acc_dtype=torch.float32,
        scale: float = None,
    ):
        super().__init__(bits, centered)

        if scale is None:
            scale = 1 / flush_every

        self.acc_dtype = acc_dtype
        self.flush_every = flush_every
        self.counter = 0
        self.scale = scale
        self.register_buffer("acc", None)

    def forward(self, x):
        with torch.no_grad():
            if self.counter == 0:
                if self.acc is None:
                    self.acc = torch.zeros_like(x, dtype=self.acc_dtype)
                else:
                    self.acc.data = torch.zeros_like(x, dtype=self.acc_dtype)

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(x) <= scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = ((neg_scale <= x) & (x <= scale)).float()

        self.counter += 1
        if self.counter == self.flush_every:
            self.counter = 0
            grad_flow_output = FlushAccFn().apply(
                x * mask + x * (1 - mask) * self.scale,
                (self.acc * self.scale).to(x.dtype),
            )
        else:
            grad_flow_output = x * mask + self.acc * (1 - mask)

        return grad_flow_output + (xq - grad_flow_output).detach()

class SRSTEQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True):
        super().__init__(bits, centered) 

    def round_stoch_fp4(self, x):
        SCALE = torch.tensor((1 << 23), dtype=torch.int32).view(torch.float32)
        SCALE_INV = SCALE.reciprocal()
        MASK = (SCALE * (-6)).view(torch.int32)
        x = x.to(torch.float32)
        x = x.clamp(min=-6, max=6)
        x = x * SCALE
        x = x.view(torch.int32)
        x = x + torch.randint_like(x, low=0, high=(1<<22))
        x = x & MASK
        x = x.view(torch.float32)
        return x * SCALE_INV
    
    def forward(self, x, clamp_grad=False):
        with torch.no_grad():
            xq = self.round_stoch_fp4(x)
        y = x + (xq - x).detach()
        if not clamp_grad:
            return y
        # Optional: zero gradient outside clamp range (since forward clamps anyway)
        mask = (x >= -6) & (x <= 6)
        # This keeps forward identical to y but scales gradient by mask
        return x + (y - x).detach() * 1.0 + (x - x.detach()) * mask.to(x.dtype)
    
class SR2STEQuantizer(STEQuantizer):
    def __init__(self, bits=4, eps=1e-8, centered=True):
        super().__init__(bits, centered) 
        self.eps = eps

    def round_stoch_fp4(self, x):
        SCALE = torch.tensor((1 << 23), dtype=torch.int32).view(torch.float32)
        SCALE_INV = SCALE.reciprocal()
        MASK = (SCALE * (-6)).view(torch.int32)
        x = x.to(torch.float32)
        x = x.clamp(min=-6, max=6)
        x = x * SCALE
        x = x.view(torch.int32)
        x = x + torch.randint_like(x, low=0, high=(1<<22))
        x = x & MASK
        x = x.view(torch.float32)
        return x * SCALE_INV
    
    def forward(self, x, clamp_grad=False):
        scale = torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + self.eps)
        with torch.no_grad():
            x = x / (scale + self.eps)
            xq_norm = self.round_stoch_fp4(x)
            xq = xq_norm * scale
        
        return x + (xq - x).detach()
        


class StochasticSTEQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, stochastic_eval=False):
        super().__init__(bits, centered)
        self.stochastic_eval = stochastic_eval

    def forward(self, x):

        if self.centered:
            max_val = torch.quantile(x.abs().to(torch.float32), 0.999, dim=-1, keepdim=True).to(x.dtype)
            scale = max_val + 1e-8
        else:
            max_val = x.amax(dim=-1, keepdim=True)
            min_val = x.amin(dim=-1, keepdim=True)
            scale = (max_val - min_val) + 1e-8

        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            
            x_clip = torch.clamp(x, -scale, scale)
            val_to_round = x_clip / step + 0.5
        else:
            step = scale / self.n_levels 
            x_clip = torch.clamp(x, min_val, max_val)
            val_to_round = (x_clip - min_val) / step

        if self.training or self.stochastic_eval:
            x_int = self.stochastic_round(val_to_round)
        else:
            x_int = torch.floor(val_to_round)

        if self.centered:
            xq = x_int * step - step / 2
        else:
            xq = x_int * step + min_val

        return x + (xq - x).detach()


    def stochastic_round(self, x):
        return torch.floor(x + torch.rand_like(x))

class StoRoundingSTEQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, stochastic=True, eval_deterministic=True):
        """
        stochastic        : enable stochastic rounding during training
        eval_deterministic: if True, fall back to deterministic rounding in eval mode
                            (useful if you want stable eval metrics)
        """
        super().__init__(bits=bits, centered=centered)
        self.stochastic = stochastic
        self.eval_deterministic = eval_deterministic

    @staticmethod
    def _stochastic_round(r: torch.Tensor,
                          qmin: float = None,
                          qmax: float = None) -> torch.Tensor:
        """
        Stochastic rounding:
          r = real-valued index
          q = floor(r) with prob (1 - frac), ceil(r) with prob frac
        Optionally clamp to [qmin, qmax].
        """
        lower = torch.floor(r)
        frac = r - lower  # in [0, 1)
        u = torch.rand_like(frac)
        upper = lower + 1.0

        choose_upper = (u < frac).to(r.dtype)
        q = lower + choose_upper

        if qmin is not None or qmax is not None:
            if qmin is None:
                qmin = -float("inf")
            if qmax is None:
                qmax = float("inf")
            q = q.clamp(qmin, qmax)

        return q

    def forward(self, x):
        # fall back to original deterministic STE in eval mode (optional)
        if (not self.training and self.eval_deterministic) or (not self.stochastic):
            return super().forward(x)

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        if self.centered:
            # centered symmetric quantization:
            #   step = 2*scale/(n_levels-1)
            #   original: xq = round(x_clip/step + 0.5)*step - step/2
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)

            r = x_clip / step + 0.5  # "index" before rounding
            # valid indices are [0, n_levels-1]
            q = self._stochastic_round(
                r,
                qmin=0.0,
                qmax=float(self.n_levels - 1),
            )
            xq = q * step - step / 2

        else:
            # uncentered (e.g. for activations):
            #   step = 2*scale/n_levels
            #   original: xq = round(x_clip/step)*step
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(
                x,
                -scale * (self.n_levels - 2) / self.n_levels,
                scale,
            )

            r = x_clip / step
            # roughly valid index range: [-(n_levels-2)/2, n_levels/2]
            q = self._stochastic_round(
                r,
                qmin=-(self.n_levels // 2),
                qmax=self.n_levels // 2,
            )
            xq = q * step

        return x + (xq - x).detach()


class PartialSRSTEQuantizer(STEQuantizer):
    def __init__(
        self,
        bits=4,
        centered=True,
        stochastic=True,
        eval_deterministic=True,
        clipping=True,
        sr_prob: float = 0.1,   # fraction of entries using SR
    ):
        """
        stochastic        : enable stochastic rounding during training
        eval_deterministic: use deterministic quant at eval time
        sr_prob           : probability that a given entry uses SR instead of round()
                            sr_prob=0 => pure deterministic
                            sr_prob=1 => all entries SR
        """
        super().__init__(bits=bits, centered=centered)
        self.stochastic = stochastic
        self.eval_deterministic = eval_deterministic
        self.sr_prob = float(sr_prob)
        self.clipping = clipping
        
    @staticmethod
    def _stochastic_round(r: torch.Tensor,
                          qmin: float = None,
                          qmax: float = None) -> torch.Tensor:
        """
        Elementwise SR:
          r = real-valued index
          q = floor(r) w.p. (1 - frac), ceil(r) w.p. frac
        """
        lower = torch.floor(r)
        frac = r - lower                           # in [0,1)
        u = torch.rand_like(frac)                  # elementwise RNG
        choose_upper = (u < frac).to(r.dtype)
        q = lower + choose_upper

        if qmin is not None or qmax is not None:
            if qmin is None:
                qmin = -float("inf")
            if qmax is None:
                qmax = float("inf")
            q = q.clamp(qmin, qmax)
        return q


    @staticmethod
    def _det_round(r: torch.Tensor,
                   qmin: float = None,
                   qmax: float = None) -> torch.Tensor:
        q = torch.round(r)
        if qmin is not None or qmax is not None:
            if qmin is None:
                qmin = -float("inf")
            if qmax is None:
                qmax = float("inf")
            q = q.clamp(qmin, qmax)
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pure deterministic paths – these are Python bools, safe for Dynamo
        if (not self.training and self.eval_deterministic) or (not self.stochastic):
            return super().forward(x)

        if self.sr_prob <= 0.0:
            return super().forward(x)

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        if self.centered:
            # centered symmetric grid
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)

            # original index: round(x_clip/step + 0.5)
            r = x_clip / step + 0.5
            qmin, qmax = 0.0, float(self.n_levels - 1)
        else:
            # uncentered grid
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(
                x,
                -scale * (self.n_levels - 2) / self.n_levels,
                scale,
            )
            r = x_clip / step
            qmin, qmax = -(self.n_levels // 2), self.n_levels // 2

        q_sr  = self._stochastic_round(r, qmin=qmin, qmax=qmax)
        q_det = self._det_round(r, qmin=qmin, qmax=qmax)

        mask = (torch.rand_like(r) < self.sr_prob).to(r.dtype)
        q = mask * q_sr + (1.0 - mask) * q_det

        if self.centered:
            xq = q * step - step / 2
        else:
            xq = q * step

        if self.clipping:
            x_min = -scale if self.centered else -scale * (self.n_levels - 2) / self.n_levels
            x_max =  scale
            
            y = ClippedSTE.apply(x, xq, x_min, x_max)
            return y
        
        else:
            return x + (xq - x).detach()


class PartialRowwiseSRSTEQuantizer(STEQuantizer):
    def __init__(
        self,
        bits=4,
        centered=True,
        stochastic=True,
        eval_deterministic=True,
        clipping=True,
        sr_prob: float = 0.1,   # fraction of entries using SR
    ):
        """
        stochastic        : enable stochastic rounding during training
        eval_deterministic: use deterministic quant at eval time
        sr_prob           : probability that a given entry uses SR instead of round()
                            sr_prob=0 => pure deterministic
                            sr_prob=1 => all entries SR
        """
        super().__init__(bits=bits, centered=centered)
        self.stochastic = stochastic
        self.eval_deterministic = eval_deterministic
        self.sr_prob = float(sr_prob)
        self.clipping = clipping

    @staticmethod
    def _stochastic_round_rowwise(r: torch.Tensor,
                                  qmin: float = None,
                                  qmax: float = None) -> torch.Tensor:
        """
        r: (..., D) - we share the noise across the last dimension.
        """
        lower = torch.floor(r)
        frac = r - lower                           # (..., D)
        # sample one u per row (all but last dim)
        noise_shape = r.shape[:-1] + (1,)
        u = torch.rand(noise_shape, device=r.device, dtype=r.dtype)
        choose_upper = (u < frac).to(r.dtype)      # broadcast over last dim
        q = lower + choose_upper
        if qmin is not None or qmax is not None:
            if qmin is None:
                qmin = -float("inf")
            if qmax is None:
                qmax = float("inf")
            q = q.clamp(qmin, qmax)
        return q


    @staticmethod
    def _det_round(r: torch.Tensor,
                   qmin: float = None,
                   qmax: float = None) -> torch.Tensor:
        q = torch.round(r)
        if qmin is not None or qmax is not None:
            if qmin is None:
                qmin = -float("inf")
            if qmax is None:
                qmax = float("inf")
            q = q.clamp(qmin, qmax)
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training and self.eval_deterministic) or (not self.stochastic):
            return super().forward(x)

        if self.sr_prob <= 0.0:
            return super().forward(x)

        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        if self.centered:
            # centered symmetric grid
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)

            # original index: round(x_clip/step + 0.5)
            r = x_clip / step + 0.5
            qmin, qmax = 0.0, float(self.n_levels - 1)
        else:
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(
                x,
                -scale * (self.n_levels - 2) / self.n_levels,
                scale,
            )
            r = x_clip / step
            qmin, qmax = -(self.n_levels // 2), self.n_levels // 2

        q_sr  = self._stochastic_round_rowwise(r, qmin=qmin, qmax=qmax)
        q_det = self._det_round(r, qmin=qmin, qmax=qmax)
        mask = (torch.rand_like(r) < self.sr_prob).to(r.dtype)
        q = mask * q_sr + (1.0 - mask) * q_det

        if self.centered:
            xq = q * step - step / 2
        else:
            xq = q * step
            
            
            
        if self.clipping:
            x_min = -scale if self.centered else -scale * (self.n_levels - 2) / self.n_levels
            x_max =  scale
            
            y = ClippedSTE.apply(x, xq, x_min, x_max)
            return y
        
        else:
            return x + (xq - x).detach()



class LSQQuantizer(nn.Module):
    """
    Implementation of LSQ quantizer from https://arxiv.org/abs/1902.08153
    LSQ uses a learnable step size for quantization. This learnable step size(alpha) is initialized using the optimal gaussian scale
    ans must be normalized with a weight decay.
    """

    def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
        super().__init__()
        # NOTE: raise_zero should never be used with FP quantization

        self.bits = bits
        self.n_levels = 2**bits
        self.all_positive = all_positive
        self.raise_zero = raise_zero

        self.q_min, self.q_max = self.get_dtype_bounds()

        self.is_alpha_init = False
        self.alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def get_dtype_bounds(self):
        if not self.all_positive:
            q_min = -self.n_levels / 2
            q_max = self.n_levels / 2 - 1
        else:
            q_min = 0
            q_max = self.n_levels - 1
        return q_min, q_max

    def cast(self, x):
        # This method can be inherited to use any casting, e.g. int, fp(e2m1, e1m2,...), optimal gaussian, etc.
        # NOTE: raise_zero should never be used with FP quantization
        return x.round()

    def ste_cast(self, x):
        return (self.cast(x) - x).detach() + x

    def grad_scale(self, x, scale):
        return (x - x * scale).detach() + x * scale

    @torch.no_grad()
    def get_initial_step_value(self, x):
        return (
            torch.mean(torch.abs(x.detach())) * 2 / (np.sqrt(self.q_max))
        )  # LSQ initialization

    def get_learnable_step(self, x):
        if not self.is_alpha_init:
            with torch.no_grad():
                step = self.get_initial_step_value(x)
                self.alpha_weight.data.multiply_(
                    torch.tensor(
                        step,
                        dtype=self.alpha_weight.dtype,
                        device=self.alpha_weight.device,
                    )
                )
            self.is_alpha_init = True
        return self.alpha_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
        xq = xscr * step

        return xq + step * 1e-9  # extra term to ensure gradient flow


class LSQPlusWeightQuantizer(LSQQuantizer):
    @torch.no_grad()
    def get_initial_step_value(self, x):
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * torch.sqrt(torch.mean(x**2)) + 1e-8
        step = 2 * scale / (self.n_levels - 1)
        return step


class LSQPlusActivationQuantizer(LSQPlusWeightQuantizer):
    def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
        super().__init__(bits, raise_zero, all_positive, **kwargs)
        self.beta_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.is_beta_init = False

    @torch.no_grad()
    def get_initial_bias_value(self, x):
        return x.min() - self.alpha_weight * self.q_min

    def get_learnable_bias(self, x):
        if not self.is_beta_init:
            with torch.no_grad():
                bias = self.get_initial_bias_value(x)
                self.beta_weight.data.add_(
                    torch.tensor(
                        bias,
                        dtype=self.beta_weight.dtype,
                        device=self.beta_weight.device,
                    )
                )
            self.is_beta_init = True
        return self.beta_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        bias = self.get_learnable_bias(x)
        bias = self.grad_scale(bias, 1.0 / np.sqrt(x.numel() * self.q_max))
        xs = (x - bias) / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
        xq = xscr * step + bias
        return xq + step * 1e-9  # extra term to ensure gradient flow


class PACTQuantizer(LSQQuantizer):
    """
    Implementation of PACT quantizer from https://arxiv.org/abs/1805.06085
    PACT and LSQ are quite similar and do the same thing for forward pass.
    The difference is in the backward pass where PACT does not perform a full gradient flow.
    """

    def forward(self, x):
        step = self.get_learnable_step(x)
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            with torch.no_grad():
                clamp_mask = ~torch.isclose(xsc, xs - 1 / 2)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            with torch.no_grad():
                clamp_mask = ~torch.isclose(xsc, xs)
            xscr = self.ste_cast(xsc)
        xq = xscr * step
        xq = xq * clamp_mask + (xq - xq * clamp_mask).detach()
        return xq + step * 1e-9  # extra term to ensure gradient flow


QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "UniformQuantizer": UniformQuantizer,
    "STEQuantizer": STEQuantizer,
    "ClipQuantizer": ClipQuantizer,
    "HalfHadamardClipQuantizer": HalfHadamardClipQuantizer,
    "HadamardClipQuantizer": HadamardClipQuantizer,
    "TrustQuantizer": TrustQuantizer,
    "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
    "HadamardTrustQuantizer": HadamardTrustQuantizer,
    "GaussianSTEQuantizer": GaussianSTEQuantizer,
    "GaussianClipQuantizer": GaussianClipQuantizer,
    "GaussianTrustQuantizer": GaussianTrustQuantizer,
    "HadamardGaussianClipQuantizer": HadamardGaussianClipQuantizer,
    "HalfHadamardGaussianTrustQuantizer": HalfHadamardGaussianTrustQuantizer,
    "HadamaardGaussianTrustQuantizer": HadamardGaussianTrustQuantizer,
    "FP4STEQuantizer": FP4STEQuantizer,
    "FP4ClipQuantizer": FP4ClipQuantizer,
    "FP4TrustQuantizer": FP4TrustQuantizer,
    "HalfHadamardFP4ClipQuantizer": HalfHadamardFP4ClipQuantizer,
    "HadamardFP4ClipQuantizer": HadamardFP4ClipQuantizer,
    "HalfHadamardFP4TrustQuantizer": HalfHadamardFP4TrustQuantizer,
    "HadamardFP4TrustQuantizer": HadamardFP4TrustQuantizer,
    "FourEightMaskedQuantizer": FourEightMaskedQuantizer,
    "FourEightSTEQuantizer": FourEightSTEQuantizer,
    "FourEightClipQuantizer": FourEightClipQuantizer,
    "FourEightTrustQuantizer": FourEightTrustQuantizer,
    "HalfHadamardFourEightTrustQuantizer": HalfHadamardFourEightTrustQuantizer,
    "HadamardFourEightTrustQuantizer": HadamardFourEightTrustQuantizer,
    "ErfClipQuantizer": ErfClipQuantizer,
    "ClipAccQuantizer": ClipAccQuantizer,
    "PACTQuantizer": PACTQuantizer,
    "LSQQuantizer": LSQQuantizer,
    "LSQPlusActivationQuantizer": LSQPlusActivationQuantizer,
    "LSQPlusWeightQuantizer": LSQPlusWeightQuantizer,
    "StochasticFP4STEQuantizer": SRSTEQuantizer,
    "SRSTEQuantizer": SR2STEQuantizer,
    "PartialSRSTEQuantizer": PartialSRSTEQuantizer,
    "PartialRowwiseSRSTEQuantizer": PartialRowwiseSRSTEQuantizer,
}


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        weight_quantizer=None,
        activation_quantizer=None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def forward(self, x):
        x = self.activation_quantizer(x)
        w = self.weight_quantizer(self.weight)
        return F.linear(x, w, self.bias)
