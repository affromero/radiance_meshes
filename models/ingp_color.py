import torch
import time
import math
from data.camera import Camera
from utils import optim
from sh_slang.eval_sh import eval_sh
# from delaunay_rasterization.internal.alphablend_tiled_slang import AlphaBlendTiledRender, render_alpha_blend_tiles_slang_raw
# from delaunay_rasterization.internal.render_grid import RenderGrid
# from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
from gDel3D.build.gdel3d import Del
from torch import nn
from icecream import ic
from utils.train_util import RGB2SH
import tinycudann as tcnn
from utils.topo_utils import calculate_circumcenters_torch
from utils.safe_math import safe_exp, safe_div, safe_sqrt
from utils.contraction import contract_mean_std
from torch_ema import ExponentialMovingAverage
from utils.contraction import contract_points, inv_contract_points
from utils.train_util import RGB2SH, safe_exp, get_expon_lr_func, sample_uniform_in_sphere
from utils import topo_utils
from utils.graphics_utils import l2_normalize_th
from typing import List
from utils import hashgrid
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

def forward_in_chunks(forward, x, chunk_size=548576):
# def forward_in_chunks(self, x, chunk_size=65536):
    """
    Same as forward(), but processes 'x' in chunks to reduce memory usage.
    """
    outputs = []
    start = 0
    while start < x.shape[0]:
        end = min(start + chunk_size, x.shape[0])
        x_chunk = x[start:end]
        outputs.append(forward(x_chunk))
        start = end
    return torch.cat(outputs, dim=0)

def next_multiple(value, multiple):
    """Round `value` up to the nearest multiple of `multiple`."""
    return ((value + multiple - 1) // multiple) * multiple

def grid_scale(level, per_level_scale, base_resolution):
    return math.ceil(math.exp2(level * math.log2(per_level_scale)) * base_resolution - 1) + 1

def compute_grid_offsets(cfg, N_POS_DIMS=3):
    """
    Translates the C++ snippet's logic into Python, returning:
      - offset_table: list of offsets per level
      - total_params: sum of all params_in_level

    cfg is a dictionary containing:
      - otype: "HashGrid" / "DenseGrid" / "TiledGrid" etc.
      - n_levels
      - n_features_per_level
      - log2_hashmap_size
      - base_resolution
      - per_level_scale
    """

    # Unpack configuration
    otype               = cfg["otype"]  # e.g. "HashGrid"
    n_levels            = cfg["n_levels"]
    n_features_per_level = cfg["n_features_per_level"]
    log2_hashmap_size   = cfg["log2_hashmap_size"]
    base_resolution     = cfg["base_resolution"]
    per_level_scale     = cfg["per_level_scale"]

    # (Optional checks, similar to C++ throws)
    # e.g., check if n_levels <= some MAX_N_LEVELS
    # if n_levels > 16:
    #     raise ValueError(f"n_levels={n_levels} exceeds maximum allowed")

    offset_table = []
    offset = 0

    # Simulate the "max_params" check for 32-bit safety
    # C++ used std::numeric_limits<uint32_t>::max() / 2
    max_params_32 = (1 << 31) - 1

    for level in range(n_levels):
        # 1) Compute resolution for this level
        resolution = grid_scale(level, per_level_scale, base_resolution)

        # 2) params_in_level = resolution^N_POS_DIMS (capped by max_params_32)
        grid_size = resolution ** N_POS_DIMS
        # params_in_level = grid_size if grid_size <= max_params_32 else max_params_32
        params_in_level = min(grid_size, max_params_32)

        # 3) Align to multiple of 8
        # ic(params_in_level, next_multiple(params_in_level, 8), resolution, max_params_32)
        params_in_level = next_multiple(params_in_level, 8)

        # 4) Adjust based on grid type
        if otype == "DenseGrid":
            # No-op
            pass
        elif otype == "TiledGrid":
            # Tiled can’t exceed base_resolution^N_POS_DIMS
            tiled_max = (base_resolution ** N_POS_DIMS)
            params_in_level = min(params_in_level, tiled_max)
        elif otype == "HashGrid":
            # Hash grid can't exceed 2^log2_hashmap_size
            params_in_level = min(params_in_level, (1 << log2_hashmap_size))
        else:
            raise RuntimeError(f"Invalid grid type '{otype}'")

        params_in_level = params_in_level * n_features_per_level
        # 5) Store offset for this level and increment
        offset_table.append(offset)
        offset += params_in_level

        # (Optional debug print)
        # print(f"Level={level}, resolution={resolution}, params_in_level={params_in_level}, offset={offset}")

    # offset now points past the last level’s parameters
    return offset_table, offset

# def exponential_decay_scheduler(optimizer, decay_start, decay_interval, decay_base):
#     def lr_lambda(step):
#         if step < decay_start:
#             return 1.0
#         else:
#             return decay_base ** ((step - decay_start) // decay_interval)
#     return LambdaLR(optimizer, lr_lambda)
def init_weights(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

@torch.jit.script
def pre_calc_cell_values(vertices, indices, center, scene_scaling: float, per_level_scale: float, L: int, scale_multi: float):
    device = vertices.device
    circumcenter, radius = calculate_circumcenters_torch(vertices[indices])
    normalized = (circumcenter - center) / scene_scaling
    cv, cr = contract_mean_std(normalized, radius / scene_scaling)
    cr = cr * scale_multi
    n = torch.arange(L, device=device).reshape(1, 1, -1)
    erf_x = safe_div(torch.tensor(1.0, device=device), safe_sqrt(per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
    scaling = torch.erf(erf_x)
    return cv, scaling

class Model:
    def __init__(self,
                 vertices: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 scale_multi=0.5,
                 log2_hashmap_size=16,
                 base_resolution=16,
                 per_level_scale=2,
                 L=10,
                 density_offset=-1,
                 **kwargs):
        self.scale_multi = scale_multi
        self.L = L
        self.dim = 4
        self.device = vertices.device
        self.density_offset = density_offset
        config = dict(
            otype="HashGrid",
            n_levels=self.L,
            n_features_per_level=self.dim,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            # base_resolution=32,
            per_level_scale=per_level_scale
        )
        self.per_level_scale = per_level_scale
        self.encoding = tcnn.Encoding(3, config).to(self.device)

        # self.encoding = torch.compile(hashgrid.HashEmbedderOptimized(
        #     [torch.zeros((3), device=self.device), torch.ones((3), device=self.device)],
        #     self.L, n_features_per_level=self.dim,
        #     log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
        #     finest_resolution=base_resolution*per_level_scale**self.L)).to(self.device)


        self.network = tcnn.Network(self.encoding.n_output_dims, 4, dict(
            # otype="CutlassMLP",
            otype="FullyFusedMLP",
            # activation="Swish",
            activation="ReLU",
            output_activation="None",
            n_neurons=64,
            n_hidden_layers=1,
        ))
        offsets, pred_total = compute_grid_offsets(config, 3)
        total = list(self.encoding.parameters())[0].shape[0]
        # ic(offsets, pred_total, total)
        # assert total == pred_total, f"Pred #params: {pred_total} vs {total}"
        resolution = grid_scale(L-1, per_level_scale, base_resolution)
        self.different_size = 0
        self.nominal_offset_size = offsets[-1] - offsets[-2]
        for o1, o2 in zip(offsets[:-1], offsets[1:]):
            if o2 - o1 == self.nominal_offset_size:
                break
            else:
                self.different_size += 1
        self.offsets = offsets

        self.center = center.reshape(1, 3)
        self.scene_scaling = scene_scaling
        self.contracted_vertices = nn.Parameter(self.contract(vertices.detach()))
        self.update_triangulation()


    def inv_contract(self, points):
        return inv_contract_points(points) * self.scene_scaling + self.center

    def contract(self, points):
        return contract_points((points - self.center) / self.scene_scaling)

    @property
    def vertices(self):
        return self.inv_contract(self.contracted_vertices)

    @staticmethod
    def init_from_pcd(point_cloud, cameras, device, **kwargs):
        torch.manual_seed(2)
        N = point_cloud.points.shape[0]
        # N = 1000
        vertices = torch.as_tensor(point_cloud.points)[:N]

        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0)
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

        vertices = nn.Parameter(vertices.cuda())
        model = Model(vertices, center, scaling, **kwargs)
        return model

    def sh_up(self):
        pass

    def update_triangulation(self):
        verts = self.vertices
        v = Del(verts.shape[0])
        indices_np, prev = v.compute(verts.detach().cpu())
        indices_np = indices_np.numpy()
        indices_np = indices_np[(indices_np < verts.shape[0]).all(axis=1)]
        self.indices = torch.as_tensor(indices_np).cuda()
        
    def get_cell_values(self, camera: Camera, mask=None):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices
        # circumcenter, radius = calculate_circumcenters_torch(vertices[indices])
        # normalized = (circumcenter - self.center) / self.scene_scaling
        # cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
        # cr = cr * self.scale_multi
        # n = torch.arange(self.L, device=self.device).reshape(1, 1, -1)
        # erf_x = safe_div(torch.tensor(1.0, device=self.device), safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1, 1, 1)))
        # scaling = torch.erf(erf_x)
        cv, scaling =  pre_calc_cell_values(
            vertices, indices, self.center, self.scene_scaling, self.per_level_scale, self.L, self.scale_multi)

        # output = checkpoint(self.encoding.forward_in_chunks, (cv/2 + 1)/2, use_reentrant=True).float()
        # output = self.encoding((cv/2 + 1)/2).float()
        output = forward_in_chunks(self.encoding, (cv/2 + 1)/2).float()
        # output = self.encoding((cv/2 + 1)/2).float()
        output = output.reshape(-1, self.dim, self.L)
        output = output * scaling
        output = self.network(output.reshape(-1, self.L * self.dim)).float()

        features = torch.cat([
            torch.nn.functional.softplus(output[:, :3]), safe_exp(output[:, 3:4]+self.density_offset)], dim=1)
        return features

    def __len__(self):
        return self.vertices.shape[0]
        

class TetOptimizer:
    def __init__(self,
                 model: Model,
                 encoding_lr: float=1e-2,
                 final_encoding_lr: float=1e-2,
                 network_lr: float=1e-3,
                 final_network_lr: float=1e-3,
                 vertices_lr: float=4e-4,
                 final_vertices_lr: float=4e-7,
                 vertices_lr_delay_mult: float=0.01,
                 vertices_lr_max_steps: int=5000,
                 weight_decay=1e-10,
                 net_weight_decay=1e-3,
                 split_std: float = 0.1,
                 vertices_beta: List[float] = [0.9, 0.99],
                 **kwargs):
        self.weight_decay = weight_decay
        self.optim = optim.CustomAdam([
            {"params": model.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], eps=1e-15, betas=vertices_beta)
        self.net_optim = optim.CustomAdam([
            {"params": model.network.parameters(), "lr": network_lr, "name": "network"},
        ], ignore_param_list=["encoding", "network"], weight_decay=net_weight_decay)
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.contracted_vertices], "lr": vertices_lr, "name": "contracted_vertices"},
        ])
        self.ema = ExponentialMovingAverage(list(model.network.parameters()) + list(model.encoding.parameters()), decay=0.99)
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None
        self.split_std = split_std

        self.net_scheduler_args = get_expon_lr_func(lr_init=network_lr,
                                                lr_final=final_network_lr,
                                                lr_delay_mult=vertices_lr_delay_mult,
                                                max_steps=vertices_lr_max_steps)
        self.encoder_scheduler_args = get_expon_lr_func(lr_init=encoding_lr,
                                                lr_final=final_encoding_lr,
                                                lr_delay_mult=vertices_lr_delay_mult,
                                                max_steps=vertices_lr_max_steps)
        self.vertex_scheduler_args = get_expon_lr_func(lr_init=vertices_lr,
                                                lr_final=final_vertices_lr,
                                                lr_delay_mult=vertices_lr_delay_mult,
                                                max_steps=vertices_lr_max_steps)

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

    def update_ema(self):
        self.ema.update()

    def add_points(self, new_verts: torch.Tensor):
        self.model.contracted_vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            contracted_vertices = self.model.contract(new_verts)
        ))['contracted_vertices']
        self.model.update_triangulation()

    def remove_points(self, mask: torch.Tensor):
        new_tensors = self.optim.prune_optimizer(mask)
        self.model.contracted_vertices = self.vertex_optim.prune_optimizer(mask)['contracted_vertices']
        self.model.update_triangulation()

    @torch.no_grad()
    def split(self, clone_indices, split_mode):
        device = self.model.device
        clone_vertices = self.model.vertices[clone_indices]

        if split_mode == "circumcenter":
            circumcenters, radius = topo_utils.calculate_circumcenters_torch(clone_vertices)
            radius = radius.reshape(-1, 1)
            circumcenters = circumcenters.reshape(-1, 3)
            sphere_loc = sample_uniform_in_sphere(circumcenters.shape[0], 3).to(device)
            r = torch.randn((clone_indices.shape[0], 1), device=self.model.device)
            r[r.abs() < 1e-2] = 1e-2
            sampled_radius = (r * self.split_std + 1) * radius
            new_vertex_location = l2_normalize_th(sphere_loc) * sampled_radius + circumcenters
        elif split_mode == "barycentric":
            barycentric = torch.rand((clone_indices.shape[0], clone_indices.shape[1], 1), device=device).clip(min=0.01, max=0.99)
            barycentric_weights = barycentric / (1e-3+barycentric.sum(dim=1, keepdim=True))
            new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
        else:
            raise Exception(f"Split mode: {split_mode} not supported")
        self.add_points(new_vertex_location)

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()
        self.net_optim.zero_grad()

    @property
    def sh_optim(self):
        optim = lambda x: x
        optim.step = lambda x=1: x
        optim.zero_grad = lambda x=1: x
        return optim

    def regularizer(self):
        # return self.weight_decay * sum([(embed.weight**2).mean() for embed in self.model.encoding.embeddings])
        # l2_reg = 1e-6 * torch.linalg.norm(list(self.model.encoding.parameters())[0], ord=2)
        # return l2_reg
        # split params
        param = list(self.model.encoding.parameters())[0]
        weight_decay = 0
        ind = 0
        for i in range(self.model.different_size):
            o = self.model.offsets[i+1] - self.model.offsets[i]
            weight_decay = weight_decay + (param[ind:self.model.offsets[i+1]]**2).mean()
            ind += o
        weight_decay = weight_decay + (param[ind:].reshape(-1, self.model.nominal_offset_size)**2).mean(dim=1).sum()
        
        return self.weight_decay * weight_decay# + l2_reg