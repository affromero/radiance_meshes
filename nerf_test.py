import os
# VERSION = 9
# if VERSION is not None:
#     os.environ["CC"] = f"/usr/bin/gcc-{VERSION}"
#     os.environ["CXX"] = f"/usr/bin/g++-{VERSION}"
from pathlib import Path
import sys
sys.path.append(str(Path(os.path.abspath('')).parent))
print(str(Path(os.path.abspath('')).parent))
import math
import torch
import matplotlib.pyplot as plt
import mediapy
from icecream import ic
from data import loader
import random
import time
from tqdm import tqdm
import numpy as np
from utils import cam_util
from utils.train_util import *
# from models.vertex_color import Model, TetOptimizer
from models.ingp_color import Model, TetOptimizer
from fused_ssim import fused_ssim
from pathlib import Path
<<<<<<< HEAD
from utils.args import Args
=======
import pickle
>>>>>>> 466bedbc5546cb1db65bcc6ffd8fe28adb08a93c

args = Args()
args.tile_size = 16
args.sh_deg = 3
args.output_path = Path("output")
args.cloning_interval = 500
args.budget = 1_000_000
args.num_densification_samples = 50
args.num_densify_iter = 3500
args.densify_start = 2500
args.num_iter = args.densify_start + args.num_densify_iter + 1500
args.sh_degree_interval = 500
args.image_folder = "images_4"
args.eval = True
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.s_param_lr = 0.025
args.rgb_param_lr = 0.025
args.sh_param_lr = 0.0025
args.vertices_lr = 5e-5
args.final_vertices_lr = 5e-7
args.vertices_lr_delay_mult = 0.01
args.vertices_lr_max_steps = args.num_iter
args.delaunay_start = 1500

args.log2_hashmap_size = 18
args.per_level_scale = 1.3
args.L = 12
args.scale_multi = 1
args.network_lr = 1e-3
args.encoding_lr = 1e-2
args.lambda_ssim = 0.2

args = Args.from_namespace(args.get_parser().parse_args())

args.output_path.mkdir(exist_ok=True, parents=True)

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cuda", eval=args.eval)

with open('camera.pkl', 'wb') as f:
    pickle.dump(test_cameras[0], f)

camera = train_cameras[0]

device = torch.device('cuda')
model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device, **args.as_dict())
tet_optim = TetOptimizer(model, **args.as_dict())

images = []
psnrs = [[]]
inds = []
# def target_num(x):
#     S = model.vertices.shape[0]
#     N = args.num_densify_iter // args.cloning_interval
#     k = (args.budget - S) // N
#     return (args.budget - S - k * N) // N**2 * x**2 + k * x + S

def target_num(x):
    S = model.vertices.shape[0]
    N = args.num_densify_iter // args.cloning_interval
    k = (args.budget - S) // N
    return k * x + S

print([target_num(i+1) for i in range(args.num_densify_iter // args.cloning_interval)])

progress_bar = tqdm(range(args.num_iter))
for train_ind in progress_bar:
    delaunay_interval = 1 if train_ind < args.delaunay_start else 10
    do_delaunay = train_ind % delaunay_interval == 0 and train_ind > 0
    do_cloning = max(train_ind - args.densify_start, 0) % args.cloning_interval == 0 and (args.num_densify_iter + args.densify_start) > train_ind >= args.densify_start
    do_tracking = False
    do_sh = train_ind % args.sh_degree_interval == 0 and train_ind > 0
    do_sh_step = train_ind % 16 == 0

    # ind = random.randint(0, len(train_cameras)-1)
    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs.append([])
    ind = inds.pop()
    # ind = 1
    camera = train_cameras[ind]
    target = camera.original_image.cuda()

    st = time.time()
    render_pkg = render(camera, model, tile_size=args.tile_size)
    # torch.cuda.synchronize()
    # print(f'render: {(time.time()-st)}')
    image = render_pkg['render']
    l2_loss = ((target - image)**2).mean()
    reg = model.regularizer()
    ssim_loss = 1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
    loss = (1-args.lambda_ssim)*l2_loss + args.lambda_ssim*ssim_loss + reg

    st = time.time()
    loss.backward()
    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_tracking:
        tet_optim.track_gradients()

    if do_sh_step:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()
    # print(f'bw: {(time.time()-st)}')
    tet_optim.update_ema()
    tet_optim.update_learning_rate(train_ind)

    if do_sh:
        model.sh_up()

    if do_cloning:
        # collect data
        tet_optim.optim.zero_grad()
        tet_optim.vertex_optim.zero_grad()
        tet_optim.sh_optim.zero_grad()

        full_inds = list(range(len(train_cameras)))
        random.shuffle(full_inds)

        sampled_cameras = [train_cameras[i] for i in full_inds[:args.num_densification_samples]]
        tet_rgbs_grad = None
        for camera in sampled_cameras:
            render_pkg = render(camera, model, register_tet_hook=True, tile_size=args.tile_size)
            # torch.cuda.synchronize()
            # print(f'render: {(time.time()-st)}')
            image = render_pkg['render']
            l2_loss = ((target - image)**2).mean()
            ssim_loss = 1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))
            loss = (1-args.lambda_ssim)*l2_loss + args.lambda_ssim*ssim_loss
            # loss = l2_loss
        
            loss.backward()
            with torch.no_grad():
                tet_grad = render_pkg['tet_grad'][0]
                scores = tet_grad.abs().sum(dim=-1)# / render_pkg['tet_area'].clip(min=1e-8)
                if tet_rgbs_grad is None:
                    tet_rgbs_grad = scores
                else:
                    tet_rgbs_grad = torch.maximum(scores, tet_rgbs_grad)
            tet_optim.sh_optim.zero_grad()
            tet_optim.optim.zero_grad()
            tet_optim.vertex_optim.zero_grad()
            torch.cuda.empty_cache()

        # tet_optim.track_gradients()

        # tet_optim.optim.step()
        # tet_optim.vertex_optim.step()
        # tet_optim.sh_optim.step()

        target_addition = target_num((train_ind - args.densify_start) // args.cloning_interval + 1) - model.vertices.shape[0]
        ic(target_addition, (train_ind - args.densify_start) // args.cloning_interval + 1)
        rgbs_threshold = torch.sort(tet_rgbs_grad).values[-min(int(target_addition), tet_rgbs_grad.shape[0])]

        clone_mask = tet_rgbs_grad > rgbs_threshold

        # rgbs_grad, vertex_grad = tet_optim.get_tracker_predicates() 
        # reduce_type = "sum"
        # # tet_std = torch.std(model.vertex_rgbs_param[model.indices], dim=1).max(dim=1).values
        # tet_rgbs_grad = rgbs_grad[model.indices].sum(dim=1)
        # tet_vertex_grad = vertex_grad[model.indices].sum(dim=1)
        # clone_mask = (tet_rgbs_grad > rgbs_threshold) | (tet_vertex_grad > vertex_threshold)
        clone_indices = model.indices[clone_mask]

        tet_optim.split(clone_indices)

        # new_vertex_location, radius = topo_utils.calculate_circumcenters_torch(model.vertices[clone_indices])
        # new_feat = model.vertex_rgbs_param[clone_indices].mean(dim=1)
        # perturb = sample_uniform_in_sphere(new_vertex_location.shape[0], 3).to(new_vertex_location.device)
        # tet_optim.add_points(new_vertex_location + radius.reshape(-1, 1) * perturb, new_feat)

        out = f"#RGBS Clone: {(tet_rgbs_grad > rgbs_threshold).sum()} "
        # out += f"#Vertex Clone: {(tet_vertex_grad > vertex_threshold).sum()} "
        out += f"∇RGBS: {tet_rgbs_grad.mean()} "
        out += f"target_addition: {target_addition} "
        # out += f"∇Vertex: {tet_vertex_grad.mean()} "
        # out += f"σ: {tet_std.mean()}"
        print(out)
        torch.cuda.empty_cache()

    psnr = 20 * math.log10(1.0 / math.sqrt(l2_loss.detach().cpu().item()))
    psnrs[-1].append(psnr)

    disp_ind = max(len(psnrs)-2, 0)
    avg_psnr = sum(psnrs[disp_ind]) / max(len(psnrs[disp_ind]), 1)
    # progress_bar.set_postfix({"PSNR": f"{psnr:>8.2f} Mean: {avg_psnr:>8.2f} #V: {len(model)} #T: {model.indices.shape[0]}"})
    progress_bar.set_postfix({
        "PSNR": repr(f"{psnr:>5.2f}"),
        "Mean": repr(f"{avg_psnr:>5.2f}"),
        "#V": len(model),
        "#T": model.indices.shape[0]
    })

    st = time.time()
    with torch.no_grad():
        if train_ind % 1 == 0:
            train_ind = 1
            camera = train_cameras[train_ind]
            render_pkg = render(camera, model, tile_size=args.tile_size)
            image = render_pkg['render']
            image = image.permute(1, 2, 0)
            image = image.detach().cpu().numpy()
            images.append(image)
    # print(f'second: {(time.time()-st)}')

    if do_delaunay:
        st = time.time()
        model.update_triangulation()
        # print(f'update: {(time.time()-st)}')

avged_psnrs = [sum(v)/len(v) for v in psnrs if len(v) == len(train_cameras)]
plt.plot(range(len(avged_psnrs)), avged_psnrs)
plt.show()
mediapy.write_video(args.output_path / "training.mp4", images)

torch.cuda.synchronize()
torch.cuda.empty_cache()

with torch.no_grad():
    epath = cam_util.generate_cam_path(train_cameras, 400)
    eimages = []
    for camera in tqdm(epath):
        render_pkg = render(camera, model, tile_size=args.tile_size)
        image = render_pkg['render']
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        eimages.append(image)

mediapy.write_video(args.output_path / "rotating.mp4", eimages)
model.save2ply(args.output_path / "test" / "ckpt.ply")

