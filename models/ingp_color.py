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

def next_multiple(value, multiple):
    """Round `value` up to the nearest multiple of `multiple`."""
    return ((value + multiple - 1) // multiple) * multiple

def grid_scale(level, per_level_scale, base_resolution):
    """
    Python equivalent of:
       grid_scale(i, std::log2(per_level_scale), base_resolution)
    If per_level_scale=2, this effectively becomes base_resolution * 2^level.
    """
    return int(base_resolution * (per_level_scale ** level))

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
        raw_params = resolution ** N_POS_DIMS
        params_in_level = raw_params if raw_params <= max_params_32 else max_params_32

        # 3) Align to multiple of 8
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
            hashed_max = (1 << log2_hashmap_size)
            params_in_level = min(params_in_level, hashed_max)
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

class Model:
    def __init__(self,
                 vertices: torch.Tensor,
                 vertex_sh_param: torch.Tensor,
                 center: torch.Tensor,
                 scene_scaling: float,
                 active_sh: int,
                 sh_deg: int,
                 min_scale_multi=0.5,
                 max_scale_multi=2):
        self.scale_multi = min_scale_multi
        self.L = 16
        self.dim = 4
        self.device = vertices.device
        self.min_scale_multi = min_scale_multi
        self.max_scale_multi = max_scale_multi
        config = dict(
            otype="HashGrid",
            n_levels=self.L,
            n_features_per_level=self.dim,
            log2_hashmap_size=14,
            base_resolution=16,
            # base_resolution=32,
            per_level_scale=2
        )
        self.encoding = tcnn.Encoding(3, config).to(self.device)
        self.network = tcnn.Network(self.encoding.n_output_dims, 4, dict(
            # otype="CutlassMLP",
            otype="FullyFusedMLP",
            activation="Softplus",
            # activation="Sigmoid",
            # activation="Sigmoid",
            # activation="Squareplus",
            # activation="ReLU",
            output_activation="None",
            n_neurons=64,
            n_hidden_layers=2,
        ))
        offsets, _ = compute_grid_offsets(config, 3)
        self.different_size = 0
        self.nominal_offset_size = offsets[-1] - offsets[-2]
        for o1, o2 in zip(offsets[:-1], offsets[1:]):
            if o2 - o1 == self.nominal_offset_size:
                break
            else:
                self.different_size += 1
        self.offsets = offsets
        ic(self.nominal_offset_size, self.different_size, offsets)

        self.center = center.reshape(1, 3)
        self.scene_scaling = scene_scaling
        self.vertices = vertices
        self.vertex_sh_param = vertex_sh_param
        self.active_sh = active_sh
        self.sh_deg = sh_deg
        self.update_triangulation()

    @staticmethod
    def init_from_pcd(point_cloud, cameras, sh_deg, device):
        ccenters = torch.stack([c.camera_center.reshape(3) for c in cameras], dim=0)
        dim = 1 + 3*(sh_deg+1)**2
        torch.manual_seed(2)
        N = point_cloud.points.shape[0]
        # N = 1000
        vertices = torch.as_tensor(point_cloud.points)[:N]
        # minv = vertices.min(dim=0, keepdim=True).values
        # maxv = vertices.max(dim=0, keepdim=True).values
        minv = ccenters.min(dim=0, keepdim=True).values
        maxv = ccenters.max(dim=0, keepdim=True).values
        center = (minv + (maxv-minv)/2).to(device)
        scaling = (maxv-minv).max().to(device)
        repeats = 3
        vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
        vertices = vertices + torch.randn(*vertices.shape) * 1e-1
        vertices = vertices.reshape(-1, 3)
        vertices = nn.Parameter(vertices.cuda())
        vertex_sh_param = torch.zeros((vertices.shape[0], 3*(sh_deg+1)**2 - 3), device=device)
        model = Model(vertices, vertex_sh_param, center, scaling, 0, sh_deg)
        return model

    def sh_up(self):
        self.active_sh = min(self.active_sh+1, self.sh_deg)

    def update_triangulation(self):
        v = Del(self.vertices.shape[0])
        indices_np, prev = v.compute(self.vertices.detach().cpu())
        indices_np = indices_np.numpy()
        self.indices_np = indices_np[(indices_np < self.vertices.shape[0]).all(axis=1)]
        self.indices = torch.as_tensor(self.indices_np).cuda()
        
    def get_cell_values(self, camera: Camera, mask=None):
        indices = self.indices[mask] if mask is not None else self.indices
        new_vertex_location, radius = calculate_circumcenters_torch(self.vertices[indices])
        new_vertex_location = self.vertices[indices].sum(dim=1)
        normalized = (new_vertex_location - self.center) / self.scene_scaling
        cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
        cr = cr * self.scale_multi
        # cv, cr = normalized, radius
        # ic(cv, normalized, new_vertex_location, self.scene_scaling)
        # ic(normalized, normalized.max(), normalized.min(), self.center, self.scene_scaling)
        output = self.encoding((cv + 1)/2).float()
        output = output.reshape(-1, self.dim, self.L)
        n = torch.arange(self.L, device=self.device).reshape(1, 1, -1)
        erf_x = safe_div(torch.tensor(1.0, device=self.device), safe_sqrt(8*n*cr.reshape(-1, 1, 1)))
        scaling = torch.erf(erf_x)
        # ic(erf_x.min(), erf_x.max(), scaling.min(), scaling.max(), output.min(), output.max())
        output = output * scaling
        # output = output.sum(dim=1)
        output = self.network(output.reshape(-1, self.L * self.dim)).float()
        # directions = l2_normalize_th(self.vertices - camera.camera_center.reshape(1, 3))
        features = torch.cat([
            torch.nn.functional.softplus(output[:, :3]), safe_exp(output[:, 3:4]-2)], dim=1)
        p = list(self.encoding.parameters())[0]
        # ic(p.max(), p.min(), features[:, 3].max(), features.min(), features.max())
        # ic(scaling, features[:, 3:4])
        return features

    def regularizer(self):
        # split params
        param = list(self.encoding.parameters())[0]
        weight_decay = 0
        ind = 0
        for i in range(self.different_size):
            o = self.offsets[i+1] - self.offsets[i]
            weight_decay = weight_decay + (param[ind:self.offsets[i+1]]**2).mean()
            ind += o
        weight_decay = weight_decay + (param[ind:].reshape(-1, self.nominal_offset_size)**2).mean(dim=1).sum()
        
        l2_reg = 1e-6 * torch.linalg.norm(list(self.encoding.parameters())[0], ord=2)
        return 0.01 * weight_decay + l2_reg

    def __len__(self):
        return self.vertices.shape[0]
        

class TetOptimizer:
    def __init__(self,
                 model: Model,
                 encoding_lr: float=2e-2,
                 network_lr: float=1e-3,
                 sh_param_lr: float=0.00025,
                 vertices_lr: float=4e-4):
        self.optim = optim.CustomAdam([
            # {"params": net.parameters(), "lr": 1e-3},
            {"params": model.encoding.parameters(), "lr": encoding_lr, "name": "encoding"},
        ], ignore_param_list=["encoding", "network"], eps=1e-15)
        self.net_optim = optim.CustomAdam([
            {"params": model.network.parameters(), "lr": network_lr, "name": "network"},
        ], ignore_param_list=["encoding", "network"])
        self.ema = ExponentialMovingAverage(list(model.network.parameters()) + list(model.encoding.parameters()), decay=0.99)
        self.sh_optim = optim.CustomAdam([
            {"params": [model.vertex_sh_param], "lr": sh_param_lr, "name": "vertex_sh_param"},
        ])
        self.vertex_optim = optim.CustomAdam([
            {"params": [model.vertices], "lr": vertices_lr, "name": "vertices"},
        ])
        self.model = model
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None

    def update_ema(self):
        self.ema.update()

    def add_points(self,
                   new_verts: torch.Tensor,
                   new_vert_sh: torch.Tensor):
        new_tensors = self.sh_optim.cat_tensors_to_optimizer(dict(
            vertex_sh_param = new_vert_sh,
        ))
        self.model.vertex_sh_param = new_tensors["vertex_sh_param"]
        self.model.vertices = self.vertex_optim.cat_tensors_to_optimizer(dict(
            vertices = new_verts
        ))['vertices']
        self.model.update_triangulation()

    def remove_points(self, mask: torch.Tensor):
        new_tensors = self.optim.prune_optimizer(mask)
        new_tensors = self.sh_optim.prune_optimizer(mask)
        self.model.vertex_sh_param = new_tensors["vertex_sh_param"]
        self.model.vertices = self.vertex_optim.prune_optimizer(mask)['vertices']
        self.model.update_triangulation()

    def split(self, clone_indices, barycentric_weights):
        new_vertex_location = (self.model.vertices[clone_indices] * barycentric_weights).sum(dim=1)
        # new_s = (self.model.vertex_s_param[clone_indices] * barycentric_weights).sum(dim=1)
        # new_rgb = (self.model.vertex_rgb_param[clone_indices] * barycentric_weights).sum(dim=1)
        new_sh = (self.model.vertex_sh_param[clone_indices] * barycentric_weights).sum(dim=1)
        self.add_points(new_vertex_location, new_sh)

    def track_gradients(self):
        # grad = 
        if self.vertex_rgbs_param_grad is not None:
            self.vertex_rgbs_param_grad += self.model.vertex_rgbs_param.grad
            self.vertex_grad += self.model.vertices.grad
        else:
            self.vertex_rgbs_param_grad = self.model.vertex_rgbs_param.grad
            self.vertex_grad = self.model.vertices.grad

        self.tracker_n += 1

    def get_tracker_predicates(self):
        if self.vertex_rgbs_param_grad is not None:
            grads = self.vertex_rgbs_param_grad / self.tracker_n
            vgrads = self.vertex_grad / self.tracker_n
            return grads.abs().sum(dim=-1), vgrads.abs().sum(dim=-1)
        else:
            return torch.zeros((len(self.model)), dtype=bool, device=self.model.device), torch.zeros((len(self.model)), dtype=bool, device=self.model.device)

    def reset_tracker(self):
        self.tracker_n = 0
        self.vertex_rgbs_param_grad = None
        self.vertex_grad = None

    def main_step(self):
        self.optim.step()
        self.net_optim.step()

    def main_zero_grad(self):
        self.optim.zero_grad()
        self.net_optim.zero_grad()