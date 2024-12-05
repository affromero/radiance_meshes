from absl.testing import absltest
from absl.testing import parameterized
import torch
from pyquaternion import Quaternion
import numpy as np
from utils.compare_quad import test_tetrahedra_rendering
import random
import math

class TetrahedraRenderingTest(parameterized.TestCase):
    def setUp(self):
        torch.manual_seed(189710234)
        self.height = 32
        self.width = 32
        self.indices = torch.tensor([[0, 1, 2, 3]]).int().cuda()
        
    def _create_base_tetrahedra(self, radius):
        vertices = (torch.rand((4, 3)).cuda() * 2 - 1) * radius
        return vertices
        
    def _random_rotation_matrix(self):
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        quat = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
        return torch.tensor(quat.rotation_matrix, device='cuda').float()

    def run_test(self, vertices, viewmat, tile_size):
        rgbs = torch.ones(1, 4).cuda()
        rgbs[:, 3] = 10
        results = test_tetrahedra_rendering(vertices, self.indices, rgbs, viewmat, 
                                         height=self.height, width=self.width, tile_size=tile_size)
        try:
            np.testing.assert_allclose(
                np.array(results['torch_image'][..., 0]),
                np.array(results['jax_image'][..., 0]),
                atol=1e-1, rtol=1e-1)
        except AssertionError as e:
            if vertices is not None:
                print(f"\nvertices = torch.{vertices}")
            if viewmat is not None:
                print(f"\nviewmat = torch.{viewmat}")
            # # Stack and save images
            # torch_img = np.array(results['torch_image'][..., 0])
            # jax_img = np.array(results['jax_image'][..., 0])
            # diff = np.abs(torch_img - jax_img)
            # stacked = np.vstack([torch_img, jax_img, diff])
             
            # import matplotlib.pyplot as plt
            # plt.imsave('debug_output.png', stacked, cmap='viridis')
            raise e

    @parameterized.product(
        tile_size=[4]#, 8, 16],
    )
    def test_center_view(self, tile_size, N=20, radius=100):
        """Test rendering from random center with random rotation."""
        for i in range(N):
            vertices = self._create_base_tetrahedra(radius)
            
            # Generate barycentric point
            barycentric = torch.rand(4).cuda()
            barycentric = barycentric / barycentric.sum()
            origin = vertices[self.indices[0]].T @ barycentric
            
            # Create view matrix with random rotation
            viewmat = torch.eye(4).cuda()
            viewmat[:3, :3] = self._random_rotation_matrix()
            viewmat[:3, 3] = origin
            viewmat = torch.linalg.inv(viewmat)
            
            self.run_test(vertices, viewmat, tile_size)

    @parameterized.product(
        offset_mag=[0.1, 1, 5, 10, 100, 1000],
        tile_size=[4]#, 8, 16],
    )
    def test_face_view(self, offset_mag, tile_size, width=32, height=32, radius=100, N=5):
        """Test rendering from face with inward-pointing rotation."""
        for i in range(N):
            vertices = self._create_base_tetrahedra(radius)
            
            # Select random face
            face_idx = torch.randint(0, 4, (1,)).item()
            face_verts = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]][face_idx]
            
            # Generate point on face
            face_barycentric = torch.rand(3).cuda()
            face_barycentric = face_barycentric / face_barycentric.sum()
            face_vertices = vertices[self.indices[0][face_verts]]
            face_point = face_vertices.T @ face_barycentric
            
            # Calculate normal and offset
            edge1 = face_vertices[1] - face_vertices[0]
            edge2 = face_vertices[2] - face_vertices[0]
            normal = torch.cross(edge1, edge2)
            normal = normal / torch.norm(normal)
            
            # Position camera
            offset = (random.random()*2-1) * offset_mag
            camera_pos = face_point + normal * offset
            
            # Create view matrix looking at face point
            forward = face_point - camera_pos
            forward = forward / torch.norm(forward)
            right = torch.cross(forward, torch.randn(3).cuda())
            right = right / torch.norm(right)
            up = torch.cross(right, forward)
            
            viewmat = torch.eye(4).cuda()
            viewmat[:3, 0] = right
            viewmat[:3, 1] = up
            viewmat[:3, 2] = forward
            viewmat[:3, 3] = camera_pos
            viewmat = torch.linalg.inv(viewmat)
            
            self.run_test(vertices, viewmat, tile_size)

    @parameterized.product(
        depth=[1, 5, 10],
        origin_radius=[1],
        tile_size=[4]#, 8, 16],
    )
    def test_frustum_point(self, depth, origin_radius, tile_size, radius=1, N=5):
        """Test rendering tetrahedra positioned in view frustum."""
        for i in range(N):
            vertices = self._create_base_tetrahedra(radius)
            
            # Sample point in view frustum at identity
            # For 90 degree FOV, frustum forms right triangle
            z = torch.rand(1).cuda().item() * depth / 2 + depth/2  # depth
            xy_range = z  # at z distance, xy bounds are [-z, z]
            x = (torch.rand(1).cuda().item() * 2 - 1) * xy_range
            y = (torch.rand(1).cuda().item() * 2 - 1) * xy_range
            frustum_point = torch.tensor([x, y, z], device='cuda')
            
            # Create random camera transform
            viewmat = torch.eye(4).cuda()
            viewmat[:3, :3] = self._random_rotation_matrix()
            viewmat[:3, 3] = torch.randn(3).cuda() * origin_radius  # random origin
            
            # Transform frustum point to camera space
            world_point = viewmat @ torch.cat([frustum_point, torch.ones(1).cuda()])
            world_point = world_point[:3] / world_point[3]
            
            # Create tetrahedra at transformed point
            vertices = vertices + world_point
            
            self.run_test(vertices, viewmat, tile_size)

if __name__ == '__main__':
    absltest.main()
