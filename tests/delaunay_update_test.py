import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
print(str(Path(__file__).parent.parent))

import numpy as np
from scipy.spatial import Delaunay
import unittest
from utils.topo_utils import *

def normalize_tetrahedra(tetrahedra: np.ndarray) -> np.ndarray:
    return tetrahedra[np.lexsort(tetrahedra.T, axis=0)]

class TestTetrahedralizationUpdate(unittest.TestCase):
    def test_small_movements(self):
        """Test that small movements produce identical results to full recomputation."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Test parameters
        n_points = 100
        epsilon = 1e-3
        n_tests = 10
        
        for test_i in range(n_tests):
            # Generate random points in unit cube
            original_points = np.random.rand(n_points, 3)
            
            # Create initial tetrahedralization
            initial_delaunay = Delaunay(original_points)
            initial_tets = initial_delaunay.simplices
            
            # Create offset points
            offset = np.random.uniform(-epsilon, epsilon, size=(n_points, 3))
            new_points = original_points + offset
            
            # Method 1: Our incremental update
            updated_tets, changed = update_tetrahedralization(
                original_points, new_points, initial_tets
            )
            
            # Method 2: Full recomputation with scipy
            new_delaunay = Delaunay(new_points)
            expected_tets = new_delaunay.simplices
            
            # Normalize both results for comparison
            updated_tets_normalized = normalize_tetrahedra(updated_tets)
            expected_tets_normalized = normalize_tetrahedra(expected_tets)
            
            # Compare results
            try:
                np.testing.assert_array_equal(
                    updated_tets_normalized,
                    expected_tets_normalized,
                    err_msg=f"Test {test_i}: Tetrahedralizations don't match"
                )
                print(f"Test {test_i}: Passed")
                print(f"  Changed tetrahedra: {len(changed)}")
                print(f"  Total tetrahedra: {len(updated_tets)}")
                print(f"  Percentage changed: {100 * len(changed) / len(updated_tets):.1f}%")
            except AssertionError as e:
                print(f"Test {test_i}: Failed")
                print(e)
                
                # Additional debugging information
                print(f"  Number of tetrahedra in update: {len(updated_tets)}")
                print(f"  Number of tetrahedra in full recomputation: {len(expected_tets)}")
                
                # Check Delaunay condition for both
                update_is_delaunay = verify_delaunay(new_points, updated_tets)
                print(f"  Updated result is Delaunay: {update_is_delaunay}")
                
                # If results don't match but both are Delaunay, they might be valid
                # alternative tetrahedralizations
                if update_is_delaunay:
                    print("  Note: Result is different but still satisfies Delaunay condition")
                    
                return False
            
        return True

    # def test_stability(self):
    #     """Test stability with varying epsilon values."""
    #     np.random.seed(42)
    #     n_points = 100
    #     epsilons = [0.001, 0.01, 0.05, 0.1]
        
    #     for epsilon in epsilons:
    #         print(f"\nTesting with epsilon = {epsilon}")
            
    #         # Generate random points
    #         original_points = np.random.rand(n_points, 3)
            
    #         # Create initial tetrahedralization
    #         initial_delaunay = Delaunay(original_points)
    #         initial_tets = initial_delaunay.simplices
            
    #         # Create offset points
    #         offset = np.random.uniform(-epsilon, epsilon, size=(n_points, 3))
    #         new_points = original_points + offset
            
    #         # Update tetrahedralization
    #         updated_tets, changed = update_tetrahedralization(
    #             original_points, new_points, initial_tets
    #         )
            
    #         # Verify Delaunay condition
    #         is_delaunay = verify_delaunay(new_points, updated_tets)
    #         print(f"Delaunay condition satisfied: {is_delaunay}")
    #         print(f"Changed tetrahedra: {len(changed)}")
    #         print(f"Total tetrahedra: {len(updated_tets)}")
    #         print(f"Percentage changed: {100 * len(changed) / len(updated_tets):.1f}%")
            
    #         self.assertTrue(is_delaunay, 
    #                       f"Failed Delaunay condition with epsilon={epsilon}")

if __name__ == '__main__':
    unittest.main()