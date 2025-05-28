import numpy as np
from collections import defaultdict, deque

def extract_meshes(rgb, verts, tets):
    # ---------- 1.  collect candidate faces ------------------------------------
    # Each tet contributes four faces, always in counter-clockwise order if you
    # keep the vertex you *omit* in first position.
    faces = np.stack([
        tets[:, [1, 2, 3]],
        tets[:, [0, 3, 2]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 2, 1]],
    ], axis=1).reshape(-1, 3)                   # (4*M, 3)

    # Associate RGB values with faces
    face_rgb = rgb.reshape(-1, 3)

    # ---------- 2.  locate boundary faces --------------------------------------
    faces_sorted = np.sort(faces, axis=1)       # canonical key, orientation lost
    faces_key = np.ascontiguousarray(faces_sorted).view(
        np.dtype((np.void, faces_sorted.dtype.itemsize * 3))
    )
    keys, inv, counts = np.unique(faces_key, return_inverse=True, return_counts=True)
    boundary_mask   = counts[inv] == 1
    boundary_faces  = faces[boundary_mask]      # keeps the original winding
    boundary_rgb    = face_rgb[boundary_mask]   # RGB values for boundary faces

    # Build edge→face lookup for shared edge connectivity
    # This maps a canonical edge (sorted tuple of two vertex indices) to a list of face indices
    e2f = defaultdict(list)
    for fi, tri in enumerate(boundary_faces):
        # Extract edges from the current triangle (v0, v1, v2)
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0])))
        ]
        for edge in edges:
            e2f[edge].append(fi)

    # Build face→face lookup based on shared edges
    # This maps a face index to a list of other face indices that share an edge with it
    f2f = defaultdict(list)
    for fi, tri in enumerate(boundary_faces):
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0])))
        ]
        for edge in edges:
            # Get all faces that share this specific edge
            shared_faces = e2f[edge]
            for other_fi in shared_faces:
                if other_fi != fi: # Ensure we don't add the face to its own adjacency list
                    f2f[fi].append(other_fi)

    # Flood-fill over triangles that share at least one edge
    components, seen = [], set()
    for seed in range(len(boundary_faces)):
        if seed in seen:
            continue
        q, comp = deque([seed]), []
        while q:
            f = q.popleft()
            if f in seen:
                continue
            seen.add(f)
            comp.append(f)
            # Extend the queue with all faces that share an edge with the current face 'f'
            q.extend(f2f[f])
        components.append(np.array(comp, dtype=np.int64))

    meshes = []
    for comp in components:
        tris = boundary_faces[comp]             # (F,3)
        tris_rgb = boundary_rgb[comp]           # RGB values for triangles
        unique_vs, new_idx = np.unique(tris, return_inverse=True)
        tris_reindexed = new_idx.reshape(-1, 3)

        # Calculate vertex colors by averaging face colors
        vertex_colors = np.zeros((len(unique_vs), boundary_rgb.shape[1]), dtype=np.float32)
        vertex_face_counts = np.zeros(len(unique_vs), dtype=np.int32)

        # Map original vertex indices to reindexed vertex indices
        orig_to_new_v_map = {orig_v: new_v_idx for new_v_idx, orig_v in enumerate(unique_vs)}

        for face_idx_in_comp, face_orig_indices in enumerate(tris):
            current_face_rgb = tris_rgb[face_idx_in_comp]
            for orig_v_idx in face_orig_indices:
                new_v_idx = orig_to_new_v_map[orig_v_idx]
                vertex_colors[new_v_idx] += current_face_rgb
                vertex_face_counts[new_v_idx] += 1
        
        # Avoid division by zero for isolated vertices if any
        vertex_colors = np.divide(vertex_colors, vertex_face_counts[:, np.newaxis], 
                                  out=np.zeros_like(vertex_colors), 
                                  where=vertex_face_counts[:, np.newaxis] != 0)


        meshes.append(
            dict(
                vertex = dict(
                    x = verts[unique_vs, 0].astype(np.float32),
                    y = verts[unique_vs, 1].astype(np.float32),
                    z = verts[unique_vs, 2].astype(np.float32),
                    r = vertex_colors[:, 0].astype(np.float32), # Separate R channel
                    g = vertex_colors[:, 1].astype(np.float32), # Separate G channel
                    b = vertex_colors[:, 2].astype(np.float32)), # Separate B channel
                face    = dict(
                    vertex_indices=tris_reindexed.astype(np.int32),
                    rgb=tris_rgb.astype(np.float32)) # Include face RGB values
            ))
    return meshes
