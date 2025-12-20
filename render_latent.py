from utils.train_util import render
from delaunay_rasterization.internal.alphablend_tiled_slang import render_constant
# from models.vertex_color import Model
import pickle
import torch
from tqdm import tqdm
from pathlib import Path
import imageio
import numpy as np
from data import loader
from util import test_util
from util.args import Args
from util import cam_util
import mediapy
from icecream import ic
from utils.model_util import *
import time

args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.eval = True
args.use_ply = False
args.render_train = False
args.min_t = 0.2
args.resolution = "test"
args = Args.from_namespace(args.get_parser().parse_args())

device = torch.device('cuda')
if args.use_ply:
    from models.tet_color import Model
    model = Model.load_ply(args.output_path / "ckpt.ply", device)
else:
    from models.ingp_color import Model
    from models.frozen import FrozenTetModel
    try:
        model = Model.load_ckpt(args.output_path, device)
    except:
        model = FrozenTetModel.load_ckpt(args.output_path, device)

# model.light_offset = -1
train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cpu", eval=args.eval)

# ic(model.min_t)
# model.min_t = args.min_t
ic(model.min_t)
if args.render_train:
    splits = zip(['train', 'test'], [train_cameras, test_cameras])
else:
    splits = zip(['test'], [test_cameras])

indices = model.indices
vertices = model.vertices
circumcenters, density, rgb, grd, sh, _ = model.compute_batch_features(
    vertices, indices)

camera = train_cameras[0]
dvrgbs = activate_output(camera.camera_center.to(model.device),
            density, rgb, grd,
            sh.reshape(-1, (model.max_sh_deg+1)**2 - 1, 3),
            indices,
            circumcenters,
            vertices, model.max_sh_deg, model.max_sh_deg)


lw = camera.image_width // 16
lh = camera.image_height // 16
c = 129
latent = torch.zeros((1, c, lh, lw))
latent3D = torch.randn((dvrgbs.shape[0], 128+1), device=dvrgbs.device)
latent3D[:, 128] = dvrgbs[:, 0]

for split, cameras in splits:
    for idx, camera in enumerate(tqdm(cameras[:1], desc=f"Rendering {split} set")):
        camera.image_width = lw
        camera.image_height = lh
        camera.gt_alpha_mask = torch.ones((1, camera.image_height, camera.image_width), device=camera.data_device)
        with torch.no_grad():
            render_pkg = render_constant(camera, model.indices, model.vertices, tile_size=args.tile_size, min_t=args.min_t, cell_values=latent3D)
            image = render_pkg['render']
            print(image.shape, render_pkg['weight_square'].shape, render_pkg['weight_square'])
            latent = image / render_pkg['weight_square']
        # with torch.no_grad():
        #     for i in range(0, c, 3):
        #         # d (1), rgb (3), grd (3)
        #         dvrgbs[:, 1:4] = torch.randn_like(dvrgbs[:, 1:4])
        #         dvrgbs[:, 4:-1] = 0
        #
        #         camera.image_width = lw
        #         camera.image_height = lh
        #         camera.gt_alpha_mask = torch.ones((1, camera.image_height, camera.image_width), device=camera.data_device)
        #
        #         render_pkg = render(camera, model, tile_size=args.tile_size, min_t=args.min_t, cell_values=dvrgbs)
        #         image = render_pkg['render']
        #         print(image.shape, render_pkg['weight_square'].shape, render_pkg['weight_square'])
        #         latent[0, i:i+3] = image / render_pkg['weight_square']
latent = latent[:, :128] / 3.5
ic(torch.std(latent), torch.mean(latent))
torch.save(latent.unsqueeze(0), "latent.pt")
