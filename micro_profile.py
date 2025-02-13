from utils.train_util import render
from models.vertex_color import Model
import pickle
import torch
from tqdm import tqdm

print("1")
tile_size = 16
with open('camera.pkl', 'rb') as f:
    camera = pickle.load(f)

device = torch.device('cuda')
print("Loading")
model = Model.load_ply("ckpt.ply", device)
print(model.indices.shape)
print("Starting")
for i in tqdm(range(100)):
    render_pkg = render(camera, model, tile_size=tile_size)
print("Done")
