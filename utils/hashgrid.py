# MIT License

# Copyright (c) 2022 Yash Sanjay Bhalgat

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
# torch.autograd.set_detect_anomaly(True)
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch.autograd import Function

BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')

def ingp_hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0], device=xyz.device)*grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = ingp_hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level
        self.n_output_dims = self.out_dim # backwards compat

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1)#, keep_mask

class HashEmbedderOptimized(nn.Module):
    def __init__(
        self, 
        bounding_box, 
        n_levels=16, 
        n_features_per_level=2,
        log2_hashmap_size=19, 
        base_resolution=16, 
        finest_resolution=512
    ):
        """
        bounding_box: (box_min, box_max), each a tensor of shape (3,)
        """
        super(HashEmbedderOptimized, self).__init__()

        # Store bounding box as buffers (won't be treated as model params)
        box_min, box_max = bounding_box
        self.register_buffer("box_min", box_min)
        self.register_buffer("box_max", box_max)

        # Scalar (ints or floats). We'll keep them in CPU or GPU depending on usage.
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.register_buffer("base_resolution", torch.tensor(base_resolution, dtype=torch.float32))
        self.register_buffer("finest_resolution", torch.tensor(finest_resolution, dtype=torch.float32))

        self.out_dim = n_levels * n_features_per_level
        self.n_output_dims = self.out_dim  # backwards compat

        # b = exp( (log(finest_res) - log(base_res)) / (n_levels - 1) )
        self.register_buffer(
            "b", 
            torch.exp(
                (self.finest_resolution.log() - self.base_resolution.log()) / (n_levels - 1)
            )
        )

        # Precompute the resolution for each level
        level_resolutions = []
        for i in range(n_levels):
            # floor(base_resolution * (b ** i))
            res_i = torch.floor(self.base_resolution * (self.b ** i))
            level_resolutions.append(res_i)
        # Turn into a single buffer (1D)
        self.register_buffer("level_resolutions", torch.stack(level_resolutions))  # shape (n_levels,)

        # Embedding tables
        self.embeddings = nn.ModuleList([
            nn.Embedding(2**log2_hashmap_size, n_features_per_level, dtype=torch.bfloat16)
            for _ in range(n_levels)
        ])

        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, a=-0.0001, b=0.0001)

        # Precompute primes used in hashing; store as buffer
        # Using the same primes you had
        prime_list = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
        self.register_buffer("primes", torch.tensor(prime_list, dtype=torch.long))

        # Precompute BOX_OFFSETS: 8 corners of a voxel
        # shape = (8, 3)
        corners = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    corners.append([i, j, k])
        self.register_buffer("box_offsets", torch.tensor(corners, dtype=torch.long))

        # For use in hashing
        self.hash_mask = (1 << log2_hashmap_size) - 1

    def ingp_hash(self, coords):
        """
        coords: (..., D) up to 7D, but we use 3D in this scenario.
        Returns a long tensor of shape (...).
        """
        # coords[..., i] * primes[i], then XOR them all
        # coords must be long
        # We'll flatten last dim, do prime multiplication, XOR, then mask
        xor_result = coords[..., 0] * self.primes[0]
        xor_result ^= coords[..., 1] * self.primes[1]
        xor_result ^= coords[..., 2] * self.primes[2]
        # for i in range(1, coords.shape[-1]):
        #     xor_result ^= coords[..., i] * self.primes[i]

        return xor_result & self.hash_mask

    def get_voxel_vertices(self, x, level_idx):
        """
        x: (B, 3)
        level_idx: which level we are computing
        Returns:
          voxel_min_vertex, voxel_max_vertex, hashed_indices
        """
        resolution = self.level_resolutions[level_idx]
        grid_size = (self.box_max - self.box_min) / resolution

        # clamp out-of-bounds
        x_clamped = torch.clamp(x, min=self.box_min, max=self.box_max)

        # bottom_left_idx: floor((x - box_min)/grid_size)
        # shape = (B, 3)
        bottom_left_idx = torch.floor((x_clamped - self.box_min) / grid_size).long()

        # Build 8 corners for each bottom-left
        # bottom_left_idx.unsqueeze(1) shape = (B, 1, 3)
        # box_offsets shape = (8, 3)
        # => voxel_indices shape = (B, 8, 3)
        voxel_indices = bottom_left_idx.unsqueeze(1) + self.box_offsets

        hashed_voxel_indices = self.ingp_hash(voxel_indices)

        # Compute the continuous coordinates of the voxel corners
        voxel_min_vertex = bottom_left_idx * grid_size + self.box_min  # (B, 3)
        voxel_max_vertex = voxel_min_vertex + grid_size  # (B, 3)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def trilinear_interp_vectorized(self, x, vmin, vmax, corner_embs):
        """
        x: (B, 3)
        vmin: (B, 3)  min corner
        vmax: (B, 3)  max corner
        corner_embs: (B, 8, D)  D = self.n_features_per_level
        Returns: (B, D)
        """
        # weights in [0,1] across each axis
        weights = (x - vmin) / (vmax - vmin)  # (B, 3)
        w0 = 1.0 - weights
        w1 = weights

        # We want to multiply corner_embs by the trilinear coefficients:
        # corner #0 -> (0,0,0) => w0[:,0]*w0[:,1]*w0[:,2]
        # corner #1 -> (0,0,1) => w0[:,0]*w0[:,1]*w1[:,2]
        # corner #2 -> (0,1,0) => w0[:,0]*w1[:,1]*w0[:,2]
        # corner #3 -> (0,1,1) => w0[:,0]*w1[:,1]*w1[:,2]
        # corner #4 -> (1,0,0) => w1[:,0]*w0[:,1]*w0[:,2]
        # corner #5 -> (1,0,1) => w1[:,0]*w0[:,1]*w1[:,2]
        # corner #6 -> (1,1,0) => w1[:,0]*w1[:,1]*w0[:,2]
        # corner #7 -> (1,1,1) => w1[:,0]*w1[:,1]*w1[:,2]

        # We'll gather these in the same order we stored corner_embs
        # which is 8 corners in the order (0,0,0)->(0,0,1)->(0,1,0)->...->(1,1,1)
        c0 = (w0[:,0] * w0[:,1] * w0[:,2])[:, None]
        c1 = (w0[:,0] * w0[:,1] * w1[:,2])[:, None]
        c2 = (w0[:,0] * w1[:,1] * w0[:,2])[:, None]
        c3 = (w0[:,0] * w1[:,1] * w1[:,2])[:, None]
        c4 = (w1[:,0] * w0[:,1] * w0[:,2])[:, None]
        c5 = (w1[:,0] * w0[:,1] * w1[:,2])[:, None]
        c6 = (w1[:,0] * w1[:,1] * w0[:,2])[:, None]
        c7 = (w1[:,0] * w1[:,1] * w1[:,2])[:, None]

        coeffs = torch.stack([c0,c1,c2,c3,c4,c5,c6,c7], dim=1)  # (B,8,1)
        # corner_embs shape: (B,8,D)
        out = (coeffs * corner_embs).sum(dim=1)  # (B, D)
        return out

    def forward(self, x):
        """
        x: (B, 3)
        returns: (B, n_levels * n_features_per_level)
        """
        # # OPTIONAL: if you want half precision:
        # if x.dtype == torch.float32:
        #     x = x.half()

        B = x.shape[0]
        x_embedded_all = []

        for i in range(self.n_levels):
            vmin, vmax, hashed_idxs = self.get_voxel_vertices(x, i)
            # Gather the corner embeddings: shape (B, 8, F)
            corner_embs = self.embeddings[i](hashed_idxs)  # random gather
            # Interpolate
            x_embedded = self.trilinear_interp_vectorized(x, vmin, vmax, corner_embs)
            x_embedded_all.append(x_embedded)

        # Concatenate along feature dim
        return torch.cat(x_embedded_all, dim=-1)

    def forward_in_chunks(self, x, chunk_size=548576):
    # def forward_in_chunks(self, x, chunk_size=65536):
        """
        Same as forward(), but processes 'x' in chunks to reduce memory usage.
        """
        outputs = []
        start = 0
        while start < x.shape[0]:
            end = min(start + chunk_size, x.shape[0])
            x_chunk = x[start:end]
            outputs.append(self.forward(x_chunk))
            start = end
        return torch.cat(outputs, dim=0)

def test_same_output():
    # Setup
    box_min = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    box_max = torch.tensor([1.0, 1.0, 1.0], device='cuda')
    bounding_box = (box_min, box_max)

    old_model = HashEmbedder(  # your original class
        bounding_box=bounding_box,
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=5,
        base_resolution=16,
        finest_resolution=64
    ).cuda()

    new_model = HashEmbedderOptimized(
        bounding_box=bounding_box,
        n_levels=4,
        n_features_per_level=2,
        log2_hashmap_size=5,
        base_resolution=16,
        finest_resolution=64
    ).cuda()

    # Random input
    x = torch.rand(1000, 3, device='cuda')

    # Evaluate
    with torch.no_grad():
        out_old = old_model(x)
        out_new = new_model(x)

    # Compare
    diff = (out_old - out_new).abs().max().item()
    print(f"Max absolute difference = {diff}")
    if diff < 1e-7:
        print("Outputs match!")
    else:
        print("Outputs differ by more than 1e-7. Check numeric differences or see if half precision is in use.")

if __name__ == "__main__":
    test_same_output()