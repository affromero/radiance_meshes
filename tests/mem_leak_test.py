from gDel3D.build.gdel3d import Del
from tqdm import tqdm
import torch

vertices = torch.randn((2_000_000, 3))
for i in tqdm(range(10000)):
    v = Del(vertices.shape[0])
    indices_np, prev = v.compute(vertices.detach().cpu())
    indices_np = indices_np.numpy()
    indices_np = indices_np[(indices_np < vertices.shape[0]).all(axis=1)]
    indices = torch.as_tensor(indices_np).cuda()