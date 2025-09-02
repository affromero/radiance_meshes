import torch
import time
import math
from data.camera import Camera
from utils import optim
from gdel3d import Del
from torch import nn
from icecream import ic
from utils.safe_math import safe_exp, safe_div, safe_sqrt, safe_pow, safe_cos, safe_sin, remove_zero, safe_arctan2
from utils.contraction import contract_mean_std
from utils.contraction import contract_points, inv_contract_points
from utils.train_util import get_expon_lr_func
from utils import topo_utils
from utils.graphics_utils import l2_normalize_th
from typing import List
from utils import hashgrid
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from pathlib import Path
import numpy as np
from utils.args import Args
import tinyplypy
from utils.phong_shading import compute_vert_color
from utils.model_util import RGB2SH, iNGPD
from sh_slang.eval_sh import eval_sh
from scipy.spatial import  Delaunay, ConvexHull
from typing import Optional, Tuple
import gc

def init_weights(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

@torch.jit.script
def pre_calc_cell_values(vertices, indices, center, scene_scaling: float, per_level_scale: float, L: int, scale_multi: float, base_resolution: float):
    device = vertices.device
    circumcenter, radius = topo_utils.calculate_circumcenters_torch(vertices[indices].double())
    normalized = (circumcenter - center) / scene_scaling
    cv, cr = contract_mean_std(normalized, radius / scene_scaling)
    cr = cr.float() * scale_multi
    n = torch.arange(L, device=device).reshape(1, 1, -1)
    erf_x = safe_div(torch.tensor(1.0, device=device), safe_sqrt(per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
    scaling = torch.erf(erf_x)
    # sphere_area = 4/3*math.pi*cr**3
    # scaling = safe_div(base_resolution * per_level_scale**n, sphere_area.reshape(-1, 1, 1)).clip(max=1)
    return cv.float(), scaling


class FrozenTetModel(nn.Module):
    def __init__(self,
                 vertices: torch.Tensor,
                 vertex_base_color: torch.Tensor,
                 vertex_shs: torch.Tensor,
                 indices: torch.Tensor,
                 densities: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 scale_multi=0.5,
                 density_offset=-1,
                 current_sh_deg=2,
                 max_sh_deg=2,
                 ablate_circumsphere=False,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.device = vertices.device
        self.density_offset = density_offset
        self.chunk_size = 508576
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = current_sh_deg
        self.feature_dim = 1
        self.mask_values = True
        self.frozen = True
        self.linear = True
        self.alpha = 0
        self.ablate_circumsphere = ablate_circumsphere

        self.backbone = torch.compile(iNGPD(**kwargs)).to(self.device)

        self.densities = nn.Parameter(densities)
        self.vertex_base_color = nn.Parameter(vertex_base_color)
        self.vertex_shs = nn.Parameter(vertex_shs)

        self.register_buffer('indices', indices.int())
        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling), device=self.device))
        self.contracted_vertices = nn.Parameter(vertices.detach().float())
        self.update_triangulation()

    def calc_tet_density(self):
        return self.densities

    @torch.no_grad
    def save2ply(self, path, sample_camera):
        """
        Convert the old save2ply function (which used 'plyfile'),
        so it uses our new tinyply-based library via pybind11.
        """

        # Ensure the output directory exists
        path.parent.mkdir(exist_ok=True, parents=True)

        # 1. Gather vertex positions
        xyz = self.vertices.detach().cpu().numpy().astype(np.float32)
        rgb = self.vertex_base_color.detach().cpu().numpy().astype(np.float32)

        # For tinyply, we store them as one dictionary for the "vertex" element:
        #   { "x": array([...]), "y": ..., "z": ... }
        # Make sure to cast to a concrete dtype (e.g. float32).
        vertex_dict = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "r": rgb[:, 0],
            "g": rgb[:, 1],
            "b": rgb[:, 2],
        }
        vertex_shs = self.vertex_shs.detach().cpu().numpy().astype(np.float32)
        for i in range(self.num_shs):
            offset = i*6
            vertex_dict[f"l{i}_r"]         = np.ascontiguousarray(vertex_shs[:, offset + 0])
            vertex_dict[f"l{i}_g"]         = np.ascontiguousarray(vertex_shs[:, offset + 1])
            vertex_dict[f"l{i}_b"]         = np.ascontiguousarray(vertex_shs[:, offset + 2])
            vertex_dict[f"l{i}_roughness"] = np.ascontiguousarray(vertex_shs[:, offset + 3])
            vertex_dict[f"l{i}_phi"]       = np.ascontiguousarray(vertex_shs[:, offset + 4])
            vertex_dict[f"l{i}_theta"]     = np.ascontiguousarray(vertex_shs[:, offset + 5])

        # 2. Compute your RGBA / lighting data per tetrahedron
        #    (same logic as in your code: iterative chunking, gather, etc.)
        N = self.indices.shape[0]
        densities = np.zeros((N), dtype=np.float32)

        vertices = self.vertices
        indices = self.indices
        # e.g., chunk-based processing (adapt as you did in your original code)
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])

            output = self.compute_batch_features(vertices, indices, start, end)
            density = torch.exp(output[:, 0] + self.density_offset)
            densities[start:end] = density.cpu().numpy().astype(np.float32)

        # 3. Build the dictionary for your "tetrahedron" element
        #    'vertex_indices' is a 2D array (N,4) for the tetra indices
        #    plus 'r', 'g', 'b', 's', and the per-light properties
        tetra_dict = {}

        # Indices: shape (N, 4). Must be stored as an unsigned int (common for face/tet indices).
        tetra_dict["vertex_indices"] = self.indices.cpu().numpy().astype(np.int32)

        # The first 4 columns in rgbs are [r, g, b, s].
        tetra_dict["density"] = np.ascontiguousarray(densities)

        # 4. Final data structure:
        # data_dict[element_name][property_name] = numpy_array
        data_dict = {
            "vertex": vertex_dict,
            "tetrahedron": tetra_dict,
        }

        tinyplypy.write_ply(str(path), data_dict, is_binary=True)

    def inv_contract(self, points):
        return inv_contract_points(points) * self.scene_scaling + self.center

    def contract(self, points):
        return contract_points((points - self.center) / self.scene_scaling)

    @property
    def vertices(self):
        return self.contracted_vertices


    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        return self.densities[start:end]

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
        pass

    def get_cell_values(self, camera: Camera, mask=None,
                        circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        densities = self.densities[mask]

        vertex_color_raw = eval_sh(
            vertices,
            self.vertex_base_color,
            self.vertex_shs.reshape(-1, (self.max_sh_deg+1)**2 - 1, 3),
            camera.camera_center.cuda(),
            self.current_sh_deg)
        vertex_color = torch.nn.functional.softplus(vertex_color_raw, beta=10)
        return vertex_color, densities.reshape(-1, 1)

    def __len__(self):
        return self.vertices.shape[0]
        
    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg+1)

    def calc_tet_alpha(self, mode="min", density=None):
        alpha_list = []
        start = 0
        
        verts = self.vertices
        inds = self.indices
        v0, v1, v2, v3 = verts[inds[:, 0]], verts[inds[:, 1]], verts[inds[:, 2]], verts[inds[:, 3]]
        
        edge_lengths = torch.stack([
            torch.norm(v0 - v1, dim=1), torch.norm(v0 - v2, dim=1), torch.norm(v0 - v3, dim=1),
            torch.norm(v1 - v2, dim=1), torch.norm(v1 - v3, dim=1), torch.norm(v2 - v3, dim=1)
        ], dim=0)
        if mode == "min":
            el = edge_lengths.min(dim=0)[0]
        elif mode == "max":
            el = edge_lengths.max(dim=0)[0]
        elif mode == "mean":
            el = edge_lengths.mean(dim=0)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'min', 'max', or 'mean'.")
        
        density = self.calc_tet_density() if density is None else density
        alpha = 1 - torch.exp(-density.reshape(-1) * el.reshape(-1))
        return alpha


class FrozenTetOptimizer:
    def __init__(self,
                 model: FrozenTetModel,
                 density_lr: float=1e-3,
                 final_density_lr: float=1e-5,
                 color_lr: float=1e-3,
                 final_color_lr: float=1e-5,
                 shs_lr: float=1e-4,
                 final_shs_lr: float=1e-6,
                 vertices_lr: float=4e-4,
                 final_vertices_lr: float=4e-7,
                 vertices_lr_delay_multi: float=0.01,
                 vertices_lr_max_steps: int=5000,
                 weight_decay=1e-10,
                 net_weight_decay=1e-3,
                 split_std: float = 0.5,
                 vertices_beta: List[float] = [0.9, 0.99],
                 iterations=10000,
                 freeze_start=5000,
                 lr_delay: int = 500,
                 **kwargs):
        self.weight_decay = weight_decay
        self.optim = optim.CustomAdam([
            {"params": model.densities, "lr": density_lr},
        ])
        self.base_color_optim = optim.CustomAdam([
            {"params": model.vertex_base_color, "lr": color_lr, "name": "base_color"},
        ])
        self.sh_optim = optim.CustomAdam([
            {"params": [model.vertex_shs], "lr": shs_lr, "name": "vertex_shs"},
        ])
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = None
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None
        self.split_std = split_std
        self.freeze_start = freeze_start

        self.density_scheduler_args = get_expon_lr_func(
            lr_init=density_lr,
            lr_final=final_density_lr,
            # lr_delay_mult=1e-8,
            lr_delay_steps=0,
            max_steps=iterations - self.freeze_start
        )

        self.color_scheduler_args = get_expon_lr_func(
            lr_init=color_lr,
            lr_final=final_color_lr,
            # lr_delay_mult=1e-8,
            lr_delay_steps=0,
            max_steps=iterations - self.freeze_start
        )

        self.shs_scheduler_args = get_expon_lr_func(
            lr_init=shs_lr,
            lr_final=final_shs_lr,
            # lr_delay_mult=1e-8,
            lr_delay_steps=0,
            max_steps=iterations - self.freeze_start
        )

    def update_learning_rate(self, iteration):

        for param_group in self.base_color_optim.param_groups:
            if param_group["name"] == "base_color":
                lr = self.color_scheduler_args(iteration - self.freeze_start)
                param_group['lr'] = lr

        for param_group in self.optim.param_groups:
            lr = self.density_scheduler_args(iteration - self.freeze_start)
            param_group['lr'] = lr

        # shs
        for param_group in self.sh_optim.param_groups:
            if param_group["name"] == "vertex_shs":
                lr = self.shs_scheduler_args(iteration - self.freeze_start)
                param_group['lr'] = lr

    def remove_points(self, mask: torch.Tensor):
        pass

    def add_points(self, new_verts: torch.Tensor, new_vertex_base_color: torch.Tensor, new_vertex_shs: torch.Tensor):
        pass

    @torch.no_grad()
    def split(self, clone_indices, split_point, split_mode, split_std, **kwargs):
        pass

    def main_step(self):
        self.optim.step()
        self.base_color_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()
        self.base_color_optim.zero_grad()

    def regularizer(self, render_pkg):
        return 0

    def update_triangulation(self, **kwargs):
        pass


# =============================================================================
# 2.  BAKING UTILITY                                                         |
# =============================================================================

@torch.no_grad()
def bake_from_model(base_model, *, detach: bool = True, chunk_size: int = 408_576) -> FrozenTetModel:
    """Convert an existing neural‑field `Model` into a parameter‑only
    `FrozenTetModel`.  All per‑tet features are *evaluated once* through the
    network and stored explicitly so that no backbone is needed afterwards."""
    device = base_model.device

    vertices_full = base_model.vertices.detach() if detach else base_model.vertices
    int_vertices  = vertices_full
    indices       = base_model.indices.detach() if detach else base_model.indices

    d_list = []
    for start in range(0, indices.shape[0], chunk_size):
        end = min(start + chunk_size, indices.shape[0])
        density = base_model.compute_batch_features(
            vertices_full, indices, start, end
        )
        d_list.append(density)

    density  = torch.cat(d_list, 0)

    return FrozenTetModel(
        vertices=int_vertices.to(device),
        indices=indices.to(device),
        densities=density.to(device),
        vertex_base_color=base_model.vertex_base_color,
        vertex_shs = base_model.vertex_shs,
        center=base_model.center.detach().to(device),
        scene_scaling=base_model.scene_scaling.detach().to(device),
        density_offset=base_model.density_offset,
        current_sh_deg=base_model.current_sh_deg,
        max_sh_deg=base_model.max_sh_deg,
        chunk_size=chunk_size,
    )


def _offload_model_to_cpu(model: nn.Module):
    """Move every parameter & buffer to CPU and drop gradients to free GPU VRAM."""
    if model is None:
        return
    for p in model.parameters(recurse=True):
        p.grad = None
        p.data = p.data.cpu()
    for b in model.buffers(recurse=True):
        b.data = b.data.cpu()
    torch.cuda.empty_cache()

@torch.no_grad()
def freeze_model(
    base_model,
    *,
    weight_decay: float = 1e-10,
    lambda_tv:    float = 0.0,
    lambda_density: float = 0.0,
    detach: bool = True,
    chunk_size: int = 408_576,
    **kwargs
) -> Tuple[FrozenTetModel, FrozenTetOptimizer]:
    """Utility wrapper to *freeze* a trained neural‑field `Model`, produce the
    corresponding `FrozenTetModel`, and return a ready‑to‑use
    `FrozenTetOptimizer` so training can continue seamlessly.

    Returns
    -------
    FrozenTetModel
        Parameter‑only representation of the field.
    FrozenTetOptimizer
        Optimiser bound to the frozen model.
    """
    print("Freezing model")
    frozen_model = bake_from_model(base_model, detach=detach, chunk_size=chunk_size)

    frozen_optim = FrozenTetOptimizer(
        frozen_model,
        weight_decay=weight_decay,
        lambda_tv=lambda_tv,
        lambda_density=lambda_density,
        **kwargs
    )

    # free GPU memory used by the big backbone (optional but handy)
    _offload_model_to_cpu(base_model)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return frozen_model, frozen_optim
