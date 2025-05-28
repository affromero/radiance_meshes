import pytest
import torch
import numpy as np
from sh_slang.eval_sh import eval_sh as eval_sh_slang
from sh_slang.eval_sh_py import eval_sh as eval_sh_py
from icecream import ic

def generate_random_unit_vectors(batch_size):
    """Generate random unit vectors on a sphere."""
    # Generate random angles
    theta = torch.rand(batch_size) * 2 * np.pi  # azimuthal angle
    phi = torch.arccos(2 * torch.rand(batch_size) - 1)  # polar angle
    
    # Convert to Cartesian coordinates
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    return torch.stack([x, y, z], dim=-1)

def generate_random_sh_coeffs(batch_size, sh_degree, num_channels=3):
    """Generate random SH coefficients."""
    num_coeffs = (sh_degree + 1) ** 2
    # Generate with shape [batch_size, sh_dim, channels] as expected by Slang implementation
    return torch.randn(batch_size, num_coeffs, num_channels)

def test_sh_implementations():
    """Test both implementations across different SH degrees."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    batch_size = 10
    
    for sh_degree in range(1, 4):  # Test degrees 0 through 3
        print(f"\n=== Testing SH degree {sh_degree} ===")
        
        # Generate random data
        directions = generate_random_unit_vectors(batch_size)
        sh_coeffs = generate_random_sh_coeffs(batch_size, sh_degree)
        means = torch.zeros_like(directions)  # Not used in Python implementation
        sh0 = sh_coeffs[:, 0, :]  # Use the actual base SH coefficient
        
        # Move tensors to cuda if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        directions = directions.to(device)
        sh_coeffs = sh_coeffs.to(device)
        means = means.to(device)
        sh0 = sh0.to(device)
        
        # For Python implementation, we need to transpose the coefficients
        sh_coeffs_py = sh_coeffs.transpose(1, 2)
        
        # Print shapes for debugging
        print(f"Shapes:")
        print(f"directions: {directions.shape}")
        print(f"sh_coeffs (Slang): {sh_coeffs.shape}")
        print(f"sh_coeffs (Python): {sh_coeffs_py.shape}")
        print(f"sh0: {sh0.shape}")
        
        # Print first item coefficients
        print(f"\nFirst item coefficients:")
        print(sh_coeffs_py[0])
        
        # Evaluate using both implementations
        result_py = eval_sh_py(sh_degree, sh_coeffs_py, directions)
        result_slang = eval_sh_slang(directions, sh0, sh_coeffs[:, 1:, :], torch.zeros((3), device=device), sh_degree)
        
        print(f"\nResults for first item:")
        print(f"Python implementation: {result_py[0]}")
        print(f"Slang implementation: {result_slang[0]}")
        
        # Compare results
        try:
            torch.testing.assert_close(
                result_py, 
                result_slang,
                rtol=1e-4,  # Relative tolerance
                atol=1e-4,  # Absolute tolerance
            )
            print(f"✓ Forward pass test passed for degree {sh_degree}!")
        except AssertionError as e:
            print(f"✗ Forward Test failed for degree {sh_degree}:")
            print(e)
        try:
            
            # Test gradients
            sh_coeffs_py.requires_grad_(True)
            sh_coeffs.requires_grad_(True)
            
            ic(sh_degree)
            result_py = eval_sh_py(sh_degree, sh_coeffs_py, directions)
            result_slang = eval_sh_slang(directions, sh_coeffs[:, 0], sh_coeffs[:, 1:, :], torch.zeros((3), device=device), sh_degree)
            
            loss_py = result_py.sum()
            loss_slang = result_slang.sum()
            
            loss_py.backward()
            loss_slang.backward()
            
            grad_py = sh_coeffs_py.grad.transpose(1, 2)
            # grad_sl = torch.cat([sh0.grad.reshape(-1, 1, 3), sh_coeffs.grad], dim=1)
            grad_sl = sh_coeffs.grad
            ic(grad_py[0], grad_sl[0])
            
            torch.testing.assert_close(
                grad_py,
                sh_coeffs.grad,
                rtol=1e-4,
                atol=1e-4
            )
            print(f"✓ Backward pass test passed for degree {sh_degree}!")
            
        except AssertionError as e:
            print(f"✗ Backward Test failed for degree {sh_degree}:")
            print(e)

if __name__ == "__main__":
    print("Testing SH implementations across different degrees...")
    test_sh_implementations() 
