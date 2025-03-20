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
from pathlib import Path, PosixPath
from utils.args import Args
import pickle
import json
from utils import safe_math
from delaunay_rasterization.internal.render_err import render_err
import imageio

args = Args()
args.tile_size = 16
args.sh_deg = 3
args.output_path = Path("output")
args.cloning_interval = 500
args.budget = 1_000_000
args.num_densification_samples = 50
args.num_densify_iter = 3500
args.densify_start = 2000
args.num_iter = 7000 # args.densify_start + args.num_densify_iter + 1500
args.sh_degree_interval = 500
args.image_folder = "images_4"
args.eval = True
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.output_path = Path("output/test/")
args.s_param_lr = 0.025
args.rgb_param_lr = 0.025
args.sh_param_lr = 0.0025
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-4
args.vertices_lr_delay_mult = 0.01
args.vertices_lr_max_steps = args.num_iter
args.delaunay_start = 1500

train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device="cuda", eval=args.eval)

inds = []
for i in tqdm(range(10000)):
    # ind = random.randint(0, len(train_cameras)-1)
    if len(inds) == 0:
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
    ind = inds.pop()
    # ind = 1
    camera = train_cameras[ind]
    target = camera.original_image.cuda()