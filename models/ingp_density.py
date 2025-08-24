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


class Model(nn.Module):
    def __init__(self,
                 vertices: torch.Tensor,
                 vertex_base_color: torch.Tensor,
                 vertex_shs: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 hashmap_dim=4,
                 hidden_dim=64,
                 density_offset=-1,
                 current_sh_deg=2,
                 max_sh_deg=2,
                 ablate_circumsphere=False,
                 **kwargs):
        super().__init__()
        self.scale_multi = scale_multi
        self.L = L
        self.dim = hashmap_dim
        self.device = vertices.device
        self.density_offset = density_offset
        self.per_level_scale = per_level_scale
        self.base_resolution = base_resolution
        self.chunk_size = 508576
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = current_sh_deg
        self.feature_dim = 1
        self.mask_values = True
        self.frozen = False
        self.linear = True
        self.alpha = 0
        self.ablate_circumsphere = ablate_circumsphere

        self.backbone = torch.compile(iNGPD(**kwargs)).to(self.device)

        self.vertex_base_color = nn.Parameter(vertex_base_color)
        self.vertex_shs = nn.Parameter(vertex_shs)

        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling), device=self.device))
        self.contracted_vertices = nn.Parameter(vertices.detach().float())
        self.update_triangulation()

    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "alldata.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)
        vertices = ckpt['contracted_vertices']
        print(f"Loaded {vertices.shape[0]} vertices")
        model = Model(vertices.to(device), ckpt['center'], ckpt['scene_scaling'], **config.as_dict())
        model.load_state_dict(ckpt)
        return model

    def calc_tet_density(self):
        densities = []
        verts = self.vertices
        for start in range(0, self.indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.indices.shape[0])
            
            density = self.compute_batch_features(verts, self.indices, start, end)

            densities.append(density.reshape(-1))
        return torch.cat(densities)

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

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, max_sh_deg,
                      voxel_size=0.00, **kwargs):
        torch.manual_seed(2)
        N = point_cloud.points.shape[0]
        # N = 1000
        vertices = torch.as_tensor(point_cloud.points)[:N]

        ccenters = torch.stack([c.camera_center.reshape(3).cuda() for c in cameras], dim=0)
        minv = ccenters.min(dim=0, keepdim=True).values
        maxv = ccenters.max(dim=0, keepdim=True).values
        # center = (minv + (maxv-minv)/2).to(device)
        # scaling = (maxv-minv).max().to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(ccenters - center.reshape(1, 3), dim=1, ord=torch.inf).max()
        # ic(center1, center, scaling1, scaling)

        vertices = vertices + torch.randn(*vertices.shape) * 1e-3
        v = Del(vertices.shape[0])
        indices_np, prev = v.compute(vertices.detach().cpu())
        indices_np = indices_np.numpy()
        indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
        vertices = vertices[indices_np].mean(dim=1)
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        # repeats = 3
        # vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
        # vertices = vertices + torch.randn(*vertices.shape) * 1e-1
        # vertices = vertices.reshape(-1, 3)

        # vertex_base_color = torch.as_tensor(point_cloud.colors).float().to(device)
        # vertex_base_color = vertex_base_color.reshape(-1, 1, 3).expand(-1, repeats, 3).reshape(-1, 3)
        vertex_base_color = torch.ones_like(vertices, device=device).float() * 0.0
        vertex_sh = torch.zeros(
            (vertices.shape[0], ((max_sh_deg+1)**2 - 1) * 3)
        ).to(device)

        vertices = nn.Parameter(vertices.cuda())
        model = Model(vertices, vertex_base_color, vertex_sh, center, scaling, max_sh_deg=max_sh_deg, **kwargs)
        return model

    def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
        tets = vertices[indices[start:end]]
        if circumcenters is None:
            circumcenter, radius = topo_utils.calculate_circumcenters_torch(tets.double())
        else:
            circumcenter = circumcenters[start:end]
        if self.ablate_circumsphere:
            circumcenter = tets.mean(dim=1)
        if self.training:
            circumcenter += self.alpha*torch.rand_like(circumcenter)
        normalized = (circumcenter - self.center) / self.scene_scaling
        radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)
        cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
        x = (cv/2 + 1)/2
        output = checkpoint(self.backbone, x, cr, use_reentrant=True)
        return output

    @torch.no_grad()
    def update_triangulation(self, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
        torch.cuda.empty_cache()
        verts = self.vertices
        if high_precision:
            indices_np = Delaunay(verts.detach().cpu().numpy()).simplices.astype(np.int32)
            # self.indices = torch.tensor(indices_np, device=verts.device).int().cuda()
        else:
            v = Del(verts.shape[0])
            indices_np, prev = v.compute(verts.detach().cpu().double())
            indices_np = indices_np.numpy()
            indices_np = indices_np[(indices_np < verts.shape[0]).all(axis=1)]
            del prev
        

        # Ensure volume is positive
        indices = torch.as_tensor(indices_np).cuda()
        vols = topo_utils.tet_volumes(verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        # Cull tets with low density
        self.indices = indices
        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.calc_tet_density()
            tet_alpha = self.calc_tet_alpha(mode="min", density=tet_density)
            mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)
            self.indices = self.indices[mask]
            
        torch.cuda.empty_cache()

    def get_cell_values(self, camera: Camera, mask=None,
                        circumcenters=None, radii=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices

        densities = torch.empty((indices.shape[0]), device=self.device)
        # densities = []
        for start in range(0, indices.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, indices.shape[0])
            output = self.compute_batch_features(vertices, indices, start, end)
            density = safe_exp(output[:, 0]+self.density_offset)
            densities[start:end] = density

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


class TetOptimizer:
    def __init__(self,
                 model: Model,
                 encoding_lr: float=1e-2,
                 final_encoding_lr: float=1e-2,
                 color_lr: float=1e-2,
                 final_color_lr: float=1e-2,  # <-- Add final LR for base color
                 network_lr: float=1e-3,
                 final_network_lr: float=1e-3,
                 shs_lr: float=1e-4,
                 final_shs_lr: float=1e-4,  # <-- Add final LR for shs
                 vertices_lr: float=4e-4,
                 final_vertices_lr: float=4e-7,
                 vertices_lr_delay_multi: float=0.01,
                 vertices_lr_max_steps: int=5000,
                 weight_decay=1e-10,
                 net_weight_decay=1e-3,
                 split_std: float = 0.5,
                 vertices_beta: List[float] = [0.9, 0.99],
                 lr_delay: int = 500,
                 **kwargs):
        self.weight_decay = weight_decay
        self.optim = optim.CustomAdam([
            {"params": model.backbone.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], betas=[0.9, 0.999], eps=1e-15)
        self.net_optim = optim.CustomAdam([
            {"params": model.backbone.density_net.parameters(),   "lr": network_lr,  "name": "density"},
        ], ignore_param_list=[], betas=[0.9, 0.999])
        self.base_color_optim = optim.CustomAdam([
            {"params": model.vertex_base_color, "lr": color_lr, "name": "base_color"},
        ])
        self.sh_optim = optim.CustomAdam([
            {"params": [model.vertex_shs], "lr": shs_lr, "name": "vertex_shs"},
        ])
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": self.vert_lr_multi*vertices_lr, "name": "contracted_vertices"},
        ])
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None
        self.split_std = split_std

        self.net_scheduler_args = get_expon_lr_func(lr_init=network_lr,
                                                lr_final=final_network_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=10000)
        self.encoder_scheduler_args = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=1e-8,
                                                lr_delay_steps=lr_delay,
                                                max_steps=10000)
        self.vertex_scheduler_args = get_expon_lr_func(lr_init=self.vert_lr_multi*vertices_lr,
                                                lr_final=self.vert_lr_multi*final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_multi,
                                                max_steps=10000,
                                                lr_delay_steps=lr_delay)
        self.color_scheduler_args = get_expon_lr_func(
            lr_init=color_lr,
            lr_final=final_color_lr,
            # lr_delay_mult=1e-8,
            lr_delay_steps=0,
            max_steps=10000
        )
        self.shs_scheduler_args = get_expon_lr_func(
            lr_init=shs_lr,
            lr_final=final_shs_lr,
            # lr_delay_mult=1e-8,
            lr_delay_steps=0,
            max_steps=10000
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.net_optim.param_groups:
            if param_group["name"] == "network":
                lr = self.net_scheduler_args(iteration)
                param_group['lr'] = lr
        for param_group in self.optim.param_groups:
            if param_group["name"] == "encoding":
                lr = self.encoder_scheduler_args(iteration)
                param_group['lr'] = lr
        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "contracted_vertices":
                lr = self.vertex_scheduler_args(iteration)
                param_group['lr'] = lr
        # Base color
        for param_group in self.base_color_optim.param_groups:
            if param_group["name"] == "base_color":
                lr = self.color_scheduler_args(iteration)
                param_group['lr'] = lr

        # shs
        for param_group in self.sh_optim.param_groups:
            if param_group["name"] == "vertex_shs":
                lr = self.shs_scheduler_args(iteration)
                param_group['lr'] = lr

    def remove_points(self, mask: torch.Tensor):
        new_tensors = self.optim.prune_optimizer(mask)
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(mask)['contracted_vertices']
        self.model.update_triangulation()

    def add_points(self, new_verts: torch.Tensor, new_vertex_base_color: torch.Tensor, new_vertex_shs: torch.Tensor):
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = new_verts
        ))['contracted_vertices']
        self.model.vertex_shs = self.sh_optim.cat_tensors_to_optimizer(dict(
            vertex_shs = new_vertex_shs
        ))['vertex_shs']
        self.model.vertex_base_color = self.base_color_optim.cat_tensors_to_optimizer(dict(
            base_color = new_vertex_base_color
        ))['base_color']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, clone_indices, split_point, split_mode, split_std, **kwargs):
        device = self.model.device

        if split_mode == "barycentric":
            barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
            barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
            new_vertex_base_color = (self.model.vertex_base_color[clone_indices] * barycentric_weights).sum(dim=1)
            new_vertex_shs = (self.model.vertex_shs[clone_indices] * barycentric_weights).sum(dim=1)
        elif split_mode == "split_point":
            tets = self.model.vertices[clone_indices]
            _, radius = topo_utils.calculate_circumcenters_torch(tets)
            split_point += (split_std * radius.reshape(-1, 1)).clip(min=1e-3, max=3) * torch.randn(*split_point.shape, device=self.model.device)
            barycentric_weights = topo_utils.calc_barycentric(split_point, tets).clip(min=0)
            barycentric_weights = barycentric_weights / (1e-3+barycentric_weights.sum(dim=1, keepdim=True))

            print(barycentric_weights.shape)
            barycentric_weights = barycentric_weights.reshape(-1, 4, 1)
            new_vertex_base_color = (self.model.vertex_base_color[clone_indices] * barycentric_weights).sum(dim=1)
            new_vertex_shs = (self.model.vertex_shs[clone_indices] * barycentric_weights).sum(dim=1)
            new_vertex_location = split_point
        else:
            raise Exception(f"Split mode: {split_mode} not supported")
        self.add_points(new_vertex_location, new_vertex_base_color, new_vertex_shs)

    def main_step(self):
        self.optim.step()
        self.net_optim.step()
        self.base_color_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()
        self.net_optim.zero_grad()
        self.base_color_optim.zero_grad()

    def regularizer(self, render_pkg):
        return self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.backbone.encoding.embeddings])

    def update_triangulation(self, **kwargs):
        self.model.update_triangulation(**kwargs)
