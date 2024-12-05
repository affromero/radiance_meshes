import numpy as np
from dtet import DelaunayTriangulation

def test_delaunay():
    # Create initial points
    points = np.random.rand(10, 3)
    
    # Create triangulation
    dt = DelaunayTriangulation()
    dt.init_from_points(points)
    print(f"Initial vertices: {dt.num_vertices()}")
    print(f"Initial cells: {dt.num_cells()}")
    
    # Add more points
    more_points = np.random.rand(5, 3)
    dt.add_points(more_points)
    print(f"After adding points - vertices: {dt.num_vertices()}")
    print(f"After adding points - cells: {dt.num_cells()}")

    # Get and print vertices and cells
    cells = dt.get_cells()
    print(f"Cells shape: {cells.shape}")

    # Remove some points
    indices_to_remove = np.array([0, 2], dtype=np.uint64)
    dt.remove_points(indices_to_remove)
    print(f"After removing points - vertices: {dt.num_vertices()}")
    print(f"After removing points - cells: {dt.num_cells()}")
    inds = dt.get_cells()
    print(inds[np.lexsort(inds.T)])
    to_remove = dt.update_points(np.random.rand(dt.num_vertices(), 3))
    inds = dt.get_cells()
    print(inds[np.lexsort(inds.T)])

if __name__ == "__main__":
    test_delaunay()
