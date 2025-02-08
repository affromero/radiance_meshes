import torch
import numpy as np
from utils import safe_math
from icecream import ic

def contract_mean_std(x, std):
    eps = torch.finfo(x.dtype).eps
    # eps = 1e-3
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x_mag_sqrt = torch.sqrt(x_mag_sq)
    mask = x_mag_sq <= 1
    z = torch.where(mask, x, (safe_math.safe_div(2 * torch.sqrt(x_mag_sq) - 1, x_mag_sq)) * x)
    # det_13 = ((1 / x_mag_sq) * ((2 / x_mag_sqrt - 1 / x_mag_sq) ** 2)) ** (1 / 3)
    top_det = safe_math.safe_pow(2 * x_mag_sqrt - 1, 1/3)
    det_13 = (top_det / x_mag_sqrt) ** 2
    # ic(det_13.min(), det_13.max(), x_mag_sqrt.min(), x_mag_sqrt.max(), top_det.min(), top_det.max())

    std = torch.where(mask[..., 0], std, det_13[..., 0] * std)
    return z, std


# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch3d.transforms
import torch
from icecream import ic
from utils.safe_math import safe_div

def l2_normalize_th(x, eps=torch.finfo(torch.float32).eps, dim=-1):
    """Normalize x to unit length along last axis."""
    return x / torch.sqrt(
        torch.clip(torch.sum(x**2, dim=dim, keepdim=True), eps, None)
    )

def contract_points(x, eps=torch.finfo(torch.float32).eps, dim=-1):
    mag = torch.sqrt(
        torch.clip(torch.sum(x**2, dim=dim, keepdim=True), eps, None)
    )
    return torch.where(mag <= 1, x, (2 - (1 / mag)) * (x / mag))


def inv_contract_points(z, eps=torch.finfo(torch.float32).eps):
    z_mag_sq = torch.sum(z**2, dim=-1, keepdims=True)
    z_mag_sq = torch.maximum(torch.ones_like(z_mag_sq), z_mag_sq)
    inv_scale = 2 * torch.sqrt(z_mag_sq.clip(min=eps)) - z_mag_sq
    x = safe_div(z, inv_scale)
    return x


def track_gaussians(fn, means, covs, densities):
    jc_means = torch.vmap(torch.func.jacrev(fn))(means.view(-1, means.shape[-1]))
    jc_means = jc_means.view(list(means.shape) + [means.shape[-1]])

    # Only update covariances on positions outside the unit sphere
    mag = means.norm(dim=-1)
    mask = mag >= 1
    covs = covs.clone()
    covs[mask] = jc_means[mask] @ covs[mask] @ torch.transpose(jc_means[mask], -2, -1)

    # densities[mask] = densities[mask] * torch.linalg.det(jc_means)[mask].abs()

    return fn(means), covs, densities


def contract_gaussians(means, covs, densities):
    return track_gaussians(contract_points, means, covs, densities)


def inv_contract_gaussians(means, covs, densities):
    return track_gaussians(inv_contract_points, means, covs, densities)


def to_cov(scale, quat):
    R = pytorch3d.transforms.quaternion_to_matrix(quat)
    S2 = torch.zeros_like(R)
    S2[:, 0, 0] = scale[:, 0] ** 2
    S2[:, 1, 1] = scale[:, 1] ** 2
    S2[:, 2, 2] = scale[:, 2] ** 2
    return torch.bmm(torch.bmm(R.permute(0, 2, 1), S2), R)


def from_covs(Ms):
    eig = torch.linalg.eig(Ms)
    scales2 = eig.eigenvalues.real.clip(min=1e-10).sqrt()
    R2 = eig.eigenvectors.real.permute(0, 2, 1)
    R2 = R2 * torch.linalg.det(R2).reshape(-1, 1, 1)
    q2 = pytorch3d.transforms.matrix_to_quaternion(R2)
    return scales2, q2


def inv_contract_gaussians_decomposed(means, scales, quats, densities):
    covs = to_cov(scales, quats)
    new_means, new_covs, new_densities = inv_contract_gaussians(means, covs, densities)
    new_scales, new_quats = from_covs(new_covs)
    return new_means, new_scales, new_quats, new_densities

