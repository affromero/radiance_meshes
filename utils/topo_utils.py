import numpy as np
from scipy.spatial import KDTree, Delaunay
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import time

def find_potential_violations(
    old_points: np.ndarray,
    new_points: np.ndarray,
    tetrahedra: np.ndarray,
    old_centers: np.ndarray,
    new_centers: np.ndarray,
    old_radii: np.ndarray,
    new_radii: np.ndarray
) -> Set[int]:
    """
    Find tetrahedra that might violate Delaunay condition after point movement.
    
    Args:
        old_points: (n, 3) array of original point positions
        new_points: (n, 3) array of new point positions
        tetrahedra: (m, 4) array of tetrahedra indices
    Returns:
        Set of indices of potentially violating tetrahedra
    """
    tree = KDTree(old_centers)
    distances, neighbors = tree.query(old_centers, k=2)  # k=2 to get nearest non-self neighbor
    delta_c = np.linalg.norm(new_centers - old_centers, axis=-1)  # movement of this center
    delta_n = np.linalg.norm(new_centers[neighbors[:, 1]] - old_centers[neighbors[:, 1]], axis=-1)  # movement of nearest center
    violations, = np.where(distances[:, 1] < (delta_c + delta_n))
    return violations

def calculate_circumcenters(vertices):
    """
    Compute the circumcenter of a tetrahedron.
    
    Args:
        vertices: Tensor of shape (..., 4, 3) containing the vertices of the tetrahedron(a).
                 The first dimension can be batched.
    
    Returns:
        circumcenter: Tensor of shape (..., 3) containing the circumcenter coordinates
    """
    # Compute vectors from v0 to other vertices
    a = vertices[..., 1, :] - vertices[..., 0, :]  # v1 - v0
    b = vertices[..., 2, :] - vertices[..., 0, :]  # v2 - v0
    c = vertices[..., 3, :] - vertices[..., 0, :]  # v3 - v0
    
    # Compute squares of lengths
    aa = np.sum(a * a, axis=-1, keepdims=True)  # |a|^2
    bb = np.sum(b * b, axis=-1, keepdims=True)  # |b|^2
    cc = np.sum(c * c, axis=-1, keepdims=True)  # |c|^2
    
    # Compute cross products
    cross_bc = np.cross(b, c, axis=-1)
    cross_ca = np.cross(c, a, axis=-1)
    cross_ab = np.cross(a, b, axis=-1)
    
    # Compute denominator
    denominator = 2.0 * np.sum(a * cross_bc, axis=-1, keepdims=True)
    
    # Create mask for small denominators
    mask = np.abs(denominator) < 1e-6
    
    # Compute circumcenter relative to verts[0]
    relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / np.where(mask, np.ones_like(denominator), denominator)
    
    # For small denominators, use center of mass instead
    center_of_mass = vertices[..., 0, :] + (a + b + c) / 4.0
    
    # Blend between circumcenter and center of mass based on mask
    relative_circumcenter = np.where(
        # mask.expand_as(relative_circumcenter),
        np.broadcast_to(mask, relative_circumcenter.shape),
        center_of_mass - vertices[..., 0, :],
        relative_circumcenter
    )

    radius = np.linalg.norm(a - relative_circumcenter, axis=-1)
    
    # Return absolute position
    return vertices[..., 0, :] + relative_circumcenter, radius

def compute_circumsphere(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute circumcenter and circumradius of a tetrahedron.
    
    Args:
        points: (4, 3) array of tetrahedron vertex coordinates
    Returns:
        center: (3,) array of circumcenter coordinates
        radius: float circumradius
    """
    # Create the linear system to solve for circumcenter
    a = points[1:] - points[0]
    b = np.sum(a * a, axis=1) / 2
    
    # Solve for circumcenter relative to points[0]
    center = np.linalg.solve(a, b)
    center = center + points[0]
    
    # Compute radius
    radius = np.linalg.norm(points[0] - center)
    return center, radius

def build_adjacency_map(tetrahedra: np.ndarray) -> Dict[int, Set[int]]:
    """
    Build a map of vertex to tetrahedra indices.
    
    Args:
        tetrahedra: (m, 4) array of tetrahedra indices
    Returns:
        Dictionary mapping vertex indices to sets of tetrahedra indices
    """
    vertex_to_tet = defaultdict(set)
    for i, tet in enumerate(tetrahedra):
        for vertex in tet:
            vertex_to_tet[vertex].add(i)
    return vertex_to_tet

def find_affected_region(
    violation_indices: Set[int],
    tetrahedra: np.ndarray,
    vertex_to_tet: Dict[int, Set[int]],
    old_points: np.ndarray,
    new_points: np.ndarray,
    safety_factor: float = 1.0
) -> Tuple[Set[int], Set[int]]:
    """
    Find all tetrahedra and vertices that need to be considered for re-tetrahedralization.
    
    Args:
        violation_indices: Indices of tetrahedra violating Delaunay condition
        tetrahedra: (m, 4) array of tetrahedra indices
        vertex_to_tet: Map of vertex to tetrahedra indices
        old_points: Original point positions
        new_points: New point positions
        safety_factor: Factor to expand search radius
    Returns:
        affected_tets: Set of tetrahedra indices to be re-tetrahedralized
        affected_vertices: Set of vertex indices involved
    """
    affected_tets = set(violation_indices)
    affected_vertices = set()
    
    # First, get all vertices from violating tetrahedra
    for tet_idx in violation_indices:
        affected_vertices.update(tetrahedra[tet_idx])
    
    # Find maximum radius of influence
    max_radius = 0
    for tet_idx in violation_indices:
        tet_vertices = tetrahedra[tet_idx]
        _, old_radius = compute_circumsphere(old_points[tet_vertices])
        _, new_radius = compute_circumsphere(new_points[tet_vertices])
        max_radius = max(max_radius, old_radius, new_radius)
    
    # Expand search radius by safety factor
    search_radius = max_radius * safety_factor
    
    # Add all tetrahedra that share any vertex with affected vertices
    queue = list(affected_vertices)
    while queue:
        vertex = queue.pop()
        # Get all tetrahedra containing this vertex
        for tet_idx in vertex_to_tet[vertex]:
            if tet_idx not in affected_tets:
                tet_vertices = tetrahedra[tet_idx]
                tet_center, tet_radius = compute_circumsphere(new_points[tet_vertices])
                
                # Check if this tetrahedron's circumsphere overlaps with our region
                for violation_idx in violation_indices:
                    violation_vertices = tetrahedra[violation_idx]
                    violation_center, violation_radius = compute_circumsphere(new_points[violation_vertices])
                    
                    if np.linalg.norm(tet_center - violation_center) < (tet_radius + violation_radius + search_radius):
                        affected_tets.add(tet_idx)
                        new_vertices = set(tet_vertices) - affected_vertices
                        affected_vertices.update(new_vertices)
                        queue.extend(new_vertices)
                        break
    
    return affected_tets, affected_vertices

def local_retetrahedralize(
    affected_vertices: Set[int],
    new_points: np.ndarray
) -> np.ndarray:
    """
    Create a new tetrahedralization for the affected region.
    
    Args:
        affected_vertices: Set of vertex indices to retriangulate
        new_points: New point positions
    Returns:
        Array of new tetrahedra indices
    """
    # Convert vertex indices to list and create mapping
    vertex_list = sorted(affected_vertices)
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_list)}
    
    # Extract relevant points and create local tetrahedralization
    local_points = new_points[vertex_list]
    local_delaunay = Delaunay(local_points)
    
    # Map local indices back to global indices
    global_tetrahedra = np.array([[vertex_list[idx] for idx in tet] 
                                 for tet in local_delaunay.simplices])
    
    return global_tetrahedra

def update_tetrahedralization(
    old_points: np.ndarray,
    new_points: np.ndarray,
    tetrahedra: np.ndarray
) -> Tuple[np.ndarray, Set[int]]:
    """
    Update a tetrahedralization after points have moved.
    
    Args:
        old_points: (n, 3) array of original point positions
        new_points: (n, 3) array of new point positions
        tetrahedra: (m, 4) array of tetrahedra indices
    Returns:
        updated_tetrahedra: Updated tetrahedra array
        changed_indices: Indices of tetrahedra that were modified
    """
    # Find initial violations
    old_centers, old_radii = calculate_circumcenters(old_points[tetrahedra])
    new_centers, new_radii = calculate_circumcenters(new_points[tetrahedra])

    violations = find_potential_violations(old_points, new_points, tetrahedra,
                                           old_centers, new_centers)
    print(violations)
    
    if not violations:
        return tetrahedra, set()
    
    # Build vertex to tetrahedra map
    vertex_to_tet = build_adjacency_map(tetrahedra)
    
    # Find affected region
    affected_tets, affected_vertices = find_affected_region(
        violations, tetrahedra, vertex_to_tet, old_points, new_points
    )
    
    # Create new tetrahedralization for affected region
    new_local_tetrahedra = local_retetrahedralize(affected_vertices, new_points)
    
    # Merge new tetrahedra with unaffected ones
    unaffected_mask = np.array([i not in affected_tets for i in range(len(tetrahedra))])
    unaffected_tetrahedra = tetrahedra[unaffected_mask]
    
    updated_tetrahedra = np.vstack([unaffected_tetrahedra, new_local_tetrahedra])
    
    return updated_tetrahedra, affected_tets

def verify_delaunay(points: np.ndarray, tetrahedra: np.ndarray) -> bool:
    """
    Verify that the tetrahedralization satisfies the Delaunay condition.
    
    Args:
        points: Point coordinates
        tetrahedra: Tetrahedra indices
    Returns:
        bool: True if Delaunay condition is satisfied
    """
    for i, tet in enumerate(tetrahedra):
        center, radius = compute_circumsphere(points[tet])
        
        # Check if any other point lies inside this circumsphere
        distances = np.linalg.norm(points - center, axis=1)
        inside_points = np.where(distances < radius - 1e-10)[0]
        
        # Remove the vertices of this tetrahedron
        inside_points = set(inside_points) - set(tet)
        
        if inside_points:
            return False
            
    return True
