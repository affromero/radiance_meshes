# %%
import os
from pathlib import Path
import sys
# sys.path.append(str(Path(os.path.abspath('')).parent))
# print(str(Path(os.path.abspath('')).parent))
import math
import torch
from torch import Tensor
from delaunay_rasterization.internal.alphablend_tiled_slang import render_alpha_blend_tiles_slang_raw
from scipy.spatial import Voronoi, Delaunay
from torch import nn
import matplotlib.pyplot as plt
import mediapy
from icecream import ic
from data import loader
import random
import time
import tinycudann as tcnn
from utils.contraction import contract_mean_std
from utils import topo_utils
from tqdm import tqdm
import numpy as np
# from dtet import DelaunayTriangulation
from dtet.build.dtet import DelaunayTriangulation
from scipy.spatial import KDTree

K = 20

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_slang_projection_matrix(znear, zfar, fy, fx, height, width, device):
    tanHalfFovX = width/(2*fx)
    tanHalfFovY = height/(2*fy)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    z_sign = 1.0

    P = torch.tensor([
       [2.0 * znear / (right - left),     0.0,                          (right + left) / (right - left), 0.0 ],
       [0.0,                              2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0 ],
       [0.0,                              0.0,                          z_sign * zfar / (zfar - znear),  -(zfar * znear) / (zfar - znear) ],
       [0.0,                              0.0,                          z_sign,                          0.0 ]
    ], device=device)

    return P

def common_camera_properties_from_gsplat(viewmats, Ks, height, width):
  """ Fetches all the Camera properties from the inria defined object"""
  zfar = 100.0
  znear = 0.01
  
  world_view_transform = viewmats
  fx = Ks[0,0]
  fy = Ks[1,1]
  projection_matrix = get_slang_projection_matrix(znear, zfar, fy, fx, height, width, Ks.device)
  fovx = focal2fov(fx, width)
  fovy = focal2fov(fy, height)

  cam_pos = viewmats.inverse()[:, 3]

  return world_view_transform, projection_matrix, cam_pos, fovy, fovx

# %%
train_cameras, test_cameras, scene_info = loader.load_dataset("/data/nerf_datasets/360/bicycle", "images_8", data_device="cuda", eval=True)

# %%

torch.manual_seed(2)
N = scene_info.point_cloud.points.shape[0]
vertices = torch.as_tensor(scene_info.point_cloud.points)[:N]
minv = vertices.min(dim=0, keepdim=True).values
maxv = vertices.max(dim=0, keepdim=True).values
repeats = 10
vertices = vertices.reshape(-1, 1, 3).expand(-1, repeats, 3)
vertices = vertices + torch.randn(*vertices.shape) * 1e-1
N = 50000
vertices = torch.cat([
  vertices.reshape(-1, 3),
  torch.rand((N, 3)) * (maxv - minv) + minv
], dim=0)
vertices = nn.Parameter(vertices.cuda())

device = torch.device('cuda')
encoding = tcnn.Encoding(3, dict(
    otype="HashGrid",
    n_levels=16,
    n_features_per_level=2,
    log2_hashmap_size=14,
    base_resolution=1,
    per_level_scale=1.5
))
network = tcnn.Network(encoding.n_output_dims, 4, dict(
    # otype="CutlassMLP",
    otype="FullyFusedMLP",
    activation="ReLU",
    output_activation="None",
    n_neurons=64,
    n_hidden_layers=2,
))
net = torch.nn.Sequential(
    encoding, network
).to(device)


def safe_exp(x):
  return x.clip(max=5).exp()

def safe_trig_helper(x, fn, t=100 * torch.pi):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  return fn(torch.nan_to_num(torch.where(torch.abs(x) < t, x, x % t)))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, torch.sin)

def rgbs_fn(xyz):
  cxyz, _ = contract_mean_std(xyz, torch.ones_like(xyz[..., 0]))
  rgbs_raw = net((cxyz/2 + 1)/2).float()
  rgbs = torch.cat([torch.sigmoid(rgbs_raw[:, :3]), safe_exp(rgbs_raw[:, 3:]-3)], dim=1)
  return rgbs


# %%
camera = train_cameras[0]
print(camera.projection_matrix)

# %%
def render(camera, indices, vertices):
    fy = fov2focal(camera.fovy, camera.image_height)
    fx = fov2focal(camera.fovx, camera.image_width)
    K = torch.tensor([
    [fx, 0, camera.image_width/2],
    [0, fy, camera.image_height/2],
    [0, 0, 1],
    ]).cuda()

    # world_view_transform, projection_matrix, cam_pos, fovy, fovx = common_camera_properties_from_gsplat(
    #     camera.world_view_transform.T, K, camera.image_height, camera.image_width)
    cam_pos = camera.world_view_transform.T.inverse()[:, 3]

    render_pkg = render_alpha_blend_tiles_slang_raw(indices, vertices, rgbs_fn,
                                                    camera.world_view_transform.T, K, cam_pos,
                                                    camera.fovy, camera.fovx, camera.image_height, camera.image_width, tile_size=tile_size)
    return render_pkg

optim = torch.optim.Adam([
    {"params": net.parameters(), "lr": 5e-3},
    {"params": [vertices], "lr": 2.5e-4},
])
tile_size = 8
images = []

st = time.time()
v = DelaunayTriangulation()
v.init_from_points(vertices.detach().cpu().numpy())
indices_np = v.get_cells()
print(time.time() - st)

# v = Delaunay(vertices.detach().cpu().numpy())
# indices_np = v.simplices

indices_np = indices_np[np.lexsort(indices_np.T)].astype(np.int32)
indices = torch.as_tensor(indices_np).cuda()

old_vertices = vertices.detach().cpu().numpy()
render_pkg = render(camera, indices, vertices)
last_circumcenters = render_pkg['circumcenters']
num_violations = []
num_tets = []
for i in tqdm(range(201)):
    optim.zero_grad()


    ind = random.randint(0, len(train_cameras)-1)
    camera = train_cameras[ind]
    target = camera.original_image.cuda()

    st = time.time()
    render_pkg = render(camera, indices, vertices)
    # print(1/(time.time()-st))
    image = render_pkg['render']
    loss = ((target - image)**2).mean()

    ind = 0
    camera = train_cameras[ind]
    render_pkg = render(camera, indices, vertices)
    image = render_pkg['render']
    image = image.permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    images.append(image)
    loss.backward()
    optim.step()

    new_vertices = vertices.detach().cpu().numpy()
    if i % K in range(K-1):
        old_centers, old_radii = topo_utils.calculate_circumcenters(old_vertices[indices_np])
        new_centers, new_radii = topo_utils.calculate_circumcenters(new_vertices[indices_np])
        violations = topo_utils.find_potential_violations(old_vertices, new_vertices, indices_np,
                                                          old_centers, new_centers,
                                                          old_radii, new_radii)
        mask = np.zeros((vertices.shape[0]), dtype=bool)
        mask[indices_np[violations].reshape(-1)] = True
        print(mask.sum())
        st = time.time()
        torch.cuda.synchronize()
        print("Moving to cpu", time.time() - st)
        st = time.time()
        v.sparse_update_points(mask, new_vertices)
        # v.update_points(vertices_np)
        print("Updating", time.time() - st)

    else:
        st = time.time()
        v = DelaunayTriangulation()
        v.init_from_points(new_vertices)
        print("Reinit: ", time.time()-st)
    old_vertices = new_vertices

    st = time.time()
    indices_np = v.get_cells()
    print("Getting cells", time.time()-st)
    st = time.time()

    # indices_np = indices_np[np.lexsort(indices_np.T)].astype(np.int32)
    indices = torch.as_tensor(indices_np.astype(np.int32)).cuda()
    torch.cuda.synchronize()
    print("Moving cells", time.time()-st)
    old_vertices = vertices.detach().cpu().numpy()

    # new_vertices = vertices.detach().cpu().numpy()
    # indices_np, _ = topo_utils.update_tetrahedralization(old_vertices, new_vertices, indices_np)
    # indices = torch.as_tensor(indices_np).cuda()

    # if i % 50 == 0:
    #     plt.imshow(image)
    #     plt.show()
    #     mediapy.show_video(images)
# mediapy.show_video(images)
# plt.plot(range(len(num_violations)), num_violations)
# plt.plot(range(len(num_tets)), num_tets)
# plt.show()

mediapy.write_video(f"training_{K}.mp4", images)
