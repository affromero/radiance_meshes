# Radiance Meshes: A Deep Dive

## Table of Contents
1. [Background: The Evolution of Neural Scene Representations](#1-background)
2. [What Are Radiance Meshes?](#2-what-are-radiance-meshes)
3. [Core Architecture](#3-core-architecture)
4. [The Rendering Pipeline](#4-the-rendering-pipeline)
5. [Training Process](#5-training-process)
6. [Comparison with NeRF and Gaussian Splatting](#6-comparison)
7. [Mathematical Foundations](#7-mathematical-foundations)
   - 7.5 [Delaunay-Voronoi Duality (Radiant Foam)](#75-delaunay-voronoi-duality)
8. [Key Innovations](#8-key-innovations)

---

## 1. Background: The Evolution of Neural Scene Representations <a name="1-background"></a>

### 1.1 Neural Radiance Fields (NeRF)

NeRF represents a scene as a continuous 5D function:

```
F: (x, y, z, θ, φ) → (r, g, b, σ)
```

Where:
- `(x, y, z)` is the 3D position
- `(θ, φ)` is the viewing direction
- `(r, g, b)` is the emitted color
- `σ` is the volume density

**Rendering Process (Volume Rendering):**
```
         Camera                     Scene
            │
            │  Ray: r(t) = o + t·d
            ▼
    ┌───────────────────────────────────────┐
    │  t₀    t₁    t₂    t₃    ...    tₙ   │
    │   ●─────●─────●─────●─────────────●   │
    │   │     │     │     │             │   │
    │   ▼     ▼     ▼     ▼             ▼   │
    │  MLP   MLP   MLP   MLP          MLP   │
    │   │     │     │     │             │   │
    │   ▼     ▼     ▼     ▼             ▼   │
    │ (σ,c) (σ,c) (σ,c) (σ,c)       (σ,c)  │
    └───────────────────────────────────────┘
                      │
                      ▼
              C = Σ Tᵢ · αᵢ · cᵢ

    Where: Tᵢ = exp(-Σⱼ<ᵢ σⱼδⱼ)  (transmittance)
           αᵢ = 1 - exp(-σᵢδᵢ)   (alpha)
```

**Pros:** Continuous representation, view-dependent effects, high quality
**Cons:** Slow (hundreds of MLP queries per ray), no explicit geometry

### 1.2 3D Gaussian Splatting (3DGS)

Gaussian Splatting represents scenes as a collection of 3D Gaussians:

```
Each Gaussian: G = (μ, Σ, c, α)
- μ: 3D position (mean)
- Σ: 3x3 covariance matrix (shape/orientation)
- c: color (often with spherical harmonics)
- α: opacity
```

**Rendering Process (Splatting):**
```
    3D Gaussians              2D Projection           Final Image

       ⬭  ⬭                      ○  ○                  ┌─────────┐
      ⬭    ⬭        ────►       ○    ○      ────►     │ ░░▒▒▓▓ │
       ⬭  ⬭                      ○  ○                  │ ▓▓▒▒░░ │
                                                       └─────────┘

    1. Project 3D Gaussians    2. Sort by depth      3. Alpha-blend
       to 2D ellipses             (front-to-back)       per pixel
```

**Pros:** Real-time rendering, explicit primitives, fast training
**Cons:** No watertight geometry, floaters, splat artifacts

---

## 2. What Are Radiance Meshes? <a name="2-what-are-radiance-meshes"></a>

Radiance Meshes combine the best of both worlds by representing scenes as a **tetrahedral mesh** with learned radiance properties.

### 2.1 Core Idea

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Point Cloud  ──►  Delaunay        ──►  Tetrahedral  ──►  Rendered│
│   from SfM          Triangulation        Mesh with         Image   │
│                                          Radiance                  │
│                                                                     │
│   ●  ●  ●           ┌─────┐              ┌─────┐                   │
│    ● ●  ●    ──►    │╲   ╱│      ──►     │╲σ,c╱│         ──►  🖼️   │
│   ●  ●  ●           └─────┘              └─────┘                   │
│                                                                     │
│   Sparse points     Tetrahedra           Each tet has:             │
│                     partition space      - density σ               │
│                                          - color c                  │
│                                          - gradient ∇c              │
│                                          - SH coefficients          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Properties

| Property | NeRF | Gaussian Splatting | Radiance Meshes |
|----------|------|-------------------|-----------------|
| Representation | Implicit (MLP) | Point primitives | Tetrahedral mesh |
| Geometry | Implicit | None (splats) | Explicit mesh |
| Rendering | Ray marching | Splatting | Ray-tet intersection |
| Topology | None | None | Delaunay triangulation |
| Watertight | No | No | Yes (can be) |
| Speed | Slow | Real-time | Near real-time |

---

## 3. Core Architecture <a name="3-core-architecture"></a>

### 3.1 Scene Structure

```
                    RADIANCE MESH STRUCTURE
    ════════════════════════════════════════════════════

    Vertices (V)                    Tetrahedra (T)
    ─────────────                   ──────────────

         v₂                              v₂
         ╱╲                              ╱│╲
        ╱  ╲                            ╱ │ ╲
       ╱    ╲                          ╱  │  ╲
      ╱      ╲                        ╱   │   ╲
    v₀────────v₁                    v₀────│────v₁
                                      ╲   │   ╱
         v₃                            ╲  │  ╱
                                        ╲ │ ╱
    Interior vertices:                   ╲│╱
    - Learnable positions                 v₃
    - Optimized during training
                                    Each tetrahedron:
    Exterior vertices:              - 4 vertices (indices)
    - Fixed on bounding sphere      - Circumcenter
    - Prevent boundary issues       - Circumradius
                                    - Per-tet features
```

**Code: Model initialization** (`models/ingp_color.py`)
```python
class Model(BaseModel):
    def __init__(self,
                 vertices: torch.Tensor,      # Interior vertices (learnable)
                 ext_vertices: torch.Tensor,  # Exterior vertices (fixed)
                 center: torch.Tensor,
                 scene_scaling: float,
                 ...):
        super().__init__()

        # Exterior vertices are fixed (on bounding sphere)
        self.register_buffer('ext_vertices', ext_vertices.to(self.device))
        self.register_buffer('center', center.reshape(1, 3))
        self.register_buffer('scene_scaling', torch.tensor(float(scene_scaling)))

        # Interior vertices are learnable parameters
        self.contracted_vertices = nn.Parameter(vertices.detach())

        # Build initial Delaunay triangulation
        self.update_triangulation()

    @property
    def vertices(self):
        """Combined interior + exterior vertices"""
        return torch.cat([self.contracted_vertices, self.ext_vertices])
```

**Code: Frozen model structure** (`models/frozen.py`)
```python
class FrozenTetModel(nn.Module):
    def __init__(self, int_vertices, ext_vertices, indices, density, rgb, gradient, sh, ...):
        super().__init__()

        # Geometry (still learnable for fine-tuning)
        self.interior_vertices = nn.Parameter(int_vertices.cuda(), requires_grad=True)
        self.register_buffer("ext_vertices", ext_vertices.cuda())
        self.register_buffer("indices", indices.int())  # Tetrahedra indices

        # Per-tetrahedron learnable features
        self.density  = nn.Parameter(density, requires_grad=True)   # (T, 1)
        self.gradient = nn.Parameter(gradient, requires_grad=True)  # (T, 3, 3)
        self.rgb      = nn.Parameter(rgb, requires_grad=True)       # (T, 3)
        self.sh       = nn.Parameter(sh.half(), requires_grad=True) # (T, SH, 3)
```

### 3.2 Per-Tetrahedron Features

Each tetrahedron stores learned features computed by an **Instant-NGP** style hash grid:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE COMPUTATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Tetrahedron                                                   │
│       ╱╲              Circumcenter                              │
│      ╱  ╲                  ●  ◄─── Query point for hash grid    │
│     ╱ ●  ╲                 │                                    │
│    ╱      ╲                ▼                                    │
│   ╱________╲        ┌─────────────┐                             │
│                     │  Hash Grid  │                             │
│                     │   (iNGP)    │                             │
│                     └──────┬──────┘                             │
│                            │                                    │
│                            ▼                                    │
│                     ┌─────────────┐                             │
│                     │  Small MLP  │                             │
│                     └──────┬──────┘                             │
│                            │                                    │
│                            ▼                                    │
│            ┌───────────────┴───────────────┐                    │
│            │                               │                    │
│            ▼                               ▼                    │
│     ┌─────────────┐                 ┌─────────────┐             │
│     │   Density   │                 │    Color    │             │
│     │     σ       │                 │   (RGB)     │             │
│     │   (1,)      │                 │   (3,)      │             │
│     └─────────────┘                 └─────────────┘             │
│            │                               │                    │
│            │                    ┌──────────┴──────────┐         │
│            │                    │                     │         │
│            │                    ▼                     ▼         │
│            │             ┌─────────────┐       ┌───────────┐    │
│            │             │  Gradient   │       │    SH     │    │
│            │             │    ∇c       │       │  Coeffs   │    │
│            │             │   (3,3)     │       │ (15, 3)   │    │
│            │             └─────────────┘       └───────────┘    │
│            │                    │                     │         │
│            └────────────────────┴─────────────────────┘         │
│                                 │                               │
│                                 ▼                               │
│                    Per-tet features: (1 + 3 + 9 + 45) dims      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Code: Feature computation for a batch of tetrahedra** (`models/ingp_color.py`)
```python
def compute_batch_features(self, vertices, indices, start, end, circumcenters=None):
    tets = vertices[indices[start:end]]

    # Compute circumcenter and circumradius for each tetrahedron
    if circumcenters is None:
        circumcenter, radius = calculate_circumcenters_torch(tets.double())
    else:
        circumcenter = circumcenters[start:end]
        radius = torch.linalg.norm(circumcenter - vertices[indices[start:end, 0]], dim=-1)

    # Normalize to scene space and apply contraction
    normalized = (circumcenter.detach() - self.center) / self.scene_scaling
    cv, cr = contract_mean_std(normalized, radius / self.scene_scaling)
    x = (cv/2 + 1)/2  # Map to [0,1] for hash grid

    # Query hash grid and MLPs
    output = checkpoint(self.backbone, x, cr, use_reentrant=True)

    return circumcenter, *output  # Returns: circumcenter, density, rgb, gradient, sh, attr
```

**Code: iNGP backbone forward pass** (`models/ingp_color.py`)
```python
def forward(self, x, cr):
    # Encode position through multi-resolution hash grid
    output = self._encode(x, cr)
    h = output.reshape(-1, self.L * self.dim).float()

    # Separate MLPs for each output type
    sigma = self.density_net(h)          # Density
    rgb = self.color_net(h)              # Base color
    field_samples = self.gradient_net(h)  # Color gradient
    sh = self.sh_net(h).half()           # Spherical harmonics

    # Activations
    rgb = rgb.reshape(-1, 3, 1) + 0.5
    density = safe_exp(sigma + self.density_offset)

    # Normalize gradient (constrained magnitude)
    grd = field_samples.reshape(-1, 1, 3)
    grd = grd / ((grd * grd).sum(dim=-1, keepdim=True) + 1).sqrt()

    return density, rgb.reshape(-1, 3), grd, sh, attr
```

### 3.3 Instant-NGP Hash Grid

The hash grid provides fast, multi-resolution spatial encoding:

```
    MULTI-RESOLUTION HASH ENCODING
    ══════════════════════════════

    Query point (x,y,z)
           │
           ▼
    ┌──────────────────────────────────────┐
    │  Level 0: Coarse (64³ resolution)    │
    │  ┌───┬───┬───┐                       │
    │  │ ● │   │   │  Hash: h(x,y,z) → idx │
    │  ├───┼───┼───┤  Features: F₀         │
    │  │   │   │   │                       │
    │  └───┴───┴───┘                       │
    ├──────────────────────────────────────┤
    │  Level 1: Medium                     │
    │  ┌─┬─┬─┬─┬─┬─┐                       │
    │  │●│ │ │ │ │ │  Hash: h(x,y,z) → idx │
    │  ├─┼─┼─┼─┼─┼─┤  Features: F₁         │
    │  │ │ │ │ │ │ │                       │
    │  └─┴─┴─┴─┴─┴─┘                       │
    ├──────────────────────────────────────┤
    │  Level L: Fine                       │
    │  ┌┬┬┬┬┬┬┬┬┬┬┬┐                       │
    │  │●│││││││││││  Hash: h(x,y,z) → idx │
    │  └┴┴┴┴┴┴┴┴┴┴┴┘  Features: Fₗ         │
    └──────────────────────────────────────┘
           │
           ▼
    Concatenate: [F₀, F₁, ..., Fₗ]
           │
           ▼
       Small MLP
           │
           ▼
    Output features
```

**Code: Hash grid initialization and encoding** (`models/ingp_color.py`)
```python
class iNGPDW(nn.Module):
    def __init__(self, sh_dim=0, scale_multi=0.5, log2_hashmap_size=16,
                 base_resolution=16, per_level_scale=2, L=10, hashmap_dim=4, ...):
        super().__init__()
        self.L = L  # Number of resolution levels
        self.dim = hashmap_dim  # Features per level

        # Hash grid configuration
        self.config = dict(
            per_level_scale=per_level_scale,  # Resolution multiplier between levels
            n_levels=L,
            otype="HashGrid",
            n_features_per_level=self.dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,  # Hash table size = 2^16
        )

        # Initialize hash grid (PyTorch or TinyCUDANN)
        self.encoding = hashgrid.HashEmbedderOptimized(
            [torch.zeros((3)), torch.ones((3))],
            self.L, n_features_per_level=self.dim,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=base_resolution * per_level_scale**self.L
        )

    def _encode(self, x: torch.Tensor, cr: torch.Tensor):
        """Encode position with circumradius-based downweighting"""
        output = self.encoding(x)
        output = output.reshape(-1, self.dim, self.L)

        # Downweight high-frequency levels for large tetrahedra
        # (prevents aliasing artifacts)
        if not self.ablate_downweighing:
            cr = cr.detach() * self.scale_multi
            n = torch.arange(self.L, device=x.device).reshape(1, 1, -1)
            erf_x = safe_div(1.0, safe_sqrt(self.per_level_scale * 4*n*cr.reshape(-1,1,1)))
            scaling = approx_erf(erf_x)
            output = output * scaling

        return output
```

### 3.4 Color Representation with Spherical Harmonics

View-dependent color is modeled using **Spherical Harmonics** (same as 3DGS):

```
    SPHERICAL HARMONICS FOR VIEW-DEPENDENT COLOR
    ═════════════════════════════════════════════

    Degree 0 (1 term):      Degree 1 (3 terms):     Degree 2 (5 terms):

         ●                       ◐                       ◑◐◑
     Constant               Directional               Quadratic
     (diffuse)              (simple shading)          (specular)


    Color at view direction d:

    c(d) = Σₗ Σₘ cₗₘ · Yₗₘ(d)

    Where:
    - cₗₘ are learnable SH coefficients (stored per-tet)
    - Yₗₘ are spherical harmonic basis functions
    - l is the degree (0 to max_sh_deg, typically 3)
    - m ranges from -l to l

    Total coefficients per color channel:
    - Degree 0: 1
    - Degree 1: 4  (1 + 3)
    - Degree 2: 9  (1 + 3 + 5)
    - Degree 3: 16 (1 + 3 + 5 + 7)
```

**Code: Spherical harmonics evaluation** (`utils/eval_sh_py.py`)
```python
# SH constants (from PlenOctree)
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, ...]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, ...]

@torch.jit.script
def eval_sh(means, sh0, sh_rest, camera_center, deg: int):
    """
    Evaluate spherical harmonics at unit directions.

    Args:
        means: (N, 3) tetrahedron centers
        sh0: (N, 3) degree-0 coefficients (base color)
        sh_rest: (N, num_coeffs, 3) higher-degree coefficients
        camera_center: (3,) camera position
        deg: maximum SH degree to evaluate

    Returns:
        (N, 3) view-dependent colors
    """
    # Compute viewing direction (normalized)
    dirs = torch.nn.functional.normalize(
        means.reshape(-1, 3) - camera_center.reshape(1, 3), dim=1, eps=1e-8)

    # Degree 0: constant term
    result = C0 * sh0 + 0.5

    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        # Degree 1: linear terms (3 coefficients)
        result = (result -
            C1 * y * sh_rest[..., 0, :] +
            C1 * z * sh_rest[..., 1, :] -
            C1 * x * sh_rest[..., 2, :])

        if deg > 1:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            # Degree 2: quadratic terms (5 coefficients)
            result = (result +
                C2[0] * xy * sh_rest[..., 3, :] +
                C2[1] * yz * sh_rest[..., 4, :] +
                C2[2] * (2*zz - xx - yy) * sh_rest[..., 5, :] +
                ...)  # continues for degree 3, 4...

    return result
```

**Code: Applying SH to tetrahedron colors** (`utils/model_util.py`)
```python
def activate_output(camera_center, density, rgb, grd, sh, attr,
                    indices, circumcenters, vertices, current_sh_deg, max_sh_deg):
    tets = vertices[indices]

    # Apply color gradient normalization
    base_color_v0_raw, normed_grd = offset_normalize(rgb, grd, circumcenters, tets)

    # Evaluate spherical harmonics for view-dependent color
    tet_color_raw = eval_sh(
        tets.mean(dim=1),           # Tetrahedron centers
        RGB2SH(base_color_v0_raw),  # Convert RGB to SH space
        sh,                          # Higher-degree SH coefficients
        camera_center,
        current_sh_deg
    ).float()

    # Final activation (softplus ensures positive colors)
    base_color_v0 = torch.nn.functional.softplus(tet_color_raw.reshape(-1, 3, 1), beta=10)

    # Concatenate all features: [density, color, gradient, extra_attrs]
    features = torch.cat([density, base_color_v0.reshape(-1, 3), normed_grd.reshape(-1, 3), attr], dim=1)
    return features.float()
```

### 3.5 Color Gradient Field

A key innovation is the **per-tetrahedron color gradient**:

```
    COLOR GRADIENT WITHIN TETRAHEDRON
    ═══════════════════════════════════

    Traditional:                    Radiance Meshes:
    Constant color per tet          Linear color field per tet

         v₂                              v₂ (color₂)
         ╱╲                              ╱╲
        ╱  ╲                            ╱  ╲
       ╱    ╲                          ╱ ∇c ╲    ◄── gradient direction
      ╱  c   ╲                        ╱      ╲
    v₀────────v₁                   v₀─────────v₁
                                 (color₀)  (color₁)

    Single color                 Color varies linearly:
    everywhere                   c(p) = c₀ + ∇c · (p - circumcenter)


    Benefit: Smoother color transitions, fewer tetrahedra needed
```

**Code: Computing per-vertex colors from gradient field** (`utils/model_util.py`)
```python
@torch.jit.script
def compute_vertex_colors_from_field(
    element_verts: torch.Tensor,   # (T, 4, 3) - Tetrahedron vertices
    base:          torch.Tensor,   # (T, 3)    - Base color at circumcenter
    gradients:     torch.Tensor,   # (T, 3, 3) - Color gradient (C=3, D=3)
    circumcenters: torch.Tensor    # (T, 3)    - Circumcenter positions
) -> torch.Tensor:
    """
    Compute per-vertex colors for each tetrahedron.

    For each vertex v:
      color(v) = base + gradient · (v - circumcenter)

    This creates a linear color field within each tetrahedron,
    enabling smooth color interpolation during rendering.
    """
    # Offset from circumcenter to each vertex
    offsets = element_verts - circumcenters[:, None, :]  # (T, 4, 3)

    # Apply gradient: einsum computes dot product for each color channel
    # 'tcd,tvd->tvc' means: for each tet t, vertex v, color c:
    #   grad_contrib[t,v,c] = sum_d(gradients[t,c,d] * offsets[t,v,d])
    grad_contrib = torch.einsum('tcd,tvd->tvc', gradients, offsets)  # (T, 4, 3)

    # Final vertex colors
    vertex_colors = base[:, None, :] + grad_contrib  # (T, 4, 3)

    return vertex_colors
```

**Code: Gradient normalization for rendering** (`utils/model_util.py`)
```python
@torch.jit.script
def offset_normalize(rgb, grd, circumcenters, tets):
    """
    Normalize color gradient based on tetrahedron size.

    The gradient is scaled by:
    1. The minimum color channel (prevents oversaturation)
    2. The circumradius (normalizes for tetrahedron size)
    """
    # Scale gradient by minimum color value
    grd = grd.reshape(-1, 1, 3) * rgb.reshape(-1, 3, 1).min(dim=1, keepdim=True).values

    # Normalize by circumradius
    radius = torch.linalg.norm(tets[:, 0] - circumcenters, dim=-1, keepdim=True).reshape(-1, 1, 1)
    normed_grd = safe_div(grd, radius)

    # Compute vertex colors using the normalized gradient
    vcolors = compute_vertex_colors_from_field(
        tets.detach(), rgb.reshape(-1, 3), normed_grd.float(), circumcenters.float().detach())

    # Return base color at v0 and the normalized gradient
    base_color_v0_raw = vcolors[:, 0]
    return base_color_v0_raw, normed_grd
```

---

## 4. The Rendering Pipeline <a name="4-the-rendering-pipeline"></a>

### 4.1 Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RENDERING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. VERTEX SHADER                                                   │
│  ─────────────────                                                  │
│                                                                     │
│     Camera        Tetrahedra                                        │
│        ●          ╱╲  ╱╲  ╱╲                                        │
│       ╱│╲        ╱  ╲╱  ╲╱  ╲                                       │
│      ╱ │ ╲      ────────────────                                    │
│                                                                     │
│                  │                                                  │
│                  ▼                                                  │
│     Project vertices to screen space                                │
│     Compute bounding boxes per tet                                  │
│                                                                     │
│  2. TILE SHADER                                                     │
│  ──────────────                                                     │
│                                                                     │
│     ┌───┬───┬───┬───┐     Assign tetrahedra to screen tiles         │
│     │ 3 │ 2 │ 1 │ 0 │     (similar to 3DGS tile-based rasterizer)   │
│     ├───┼───┼───┼───┤                                               │
│     │ 4 │ 5 │ 2 │ 1 │     Each tile: list of overlapping tets       │
│     ├───┼───┼───┼───┤                                               │
│     │ 2 │ 3 │ 4 │ 3 │     Sort by depth (front-to-back)             │
│     └───┴───┴───┴───┘                                               │
│                                                                     │
│  3. ALPHA BLEND SHADER (per tile)                                   │
│  ───────────────────────────────                                    │
│                                                                     │
│     For each pixel in tile:                                         │
│     ┌─────────────────────────────────────────────┐                 │
│     │  Ray from camera through pixel              │                 │
│     │         │                                   │                 │
│     │         ▼                                   │                 │
│     │  For each tet (front-to-back):              │                 │
│     │    ┌─────────────────────────────────┐      │                 │
│     │    │ 1. Ray-tetrahedron intersection │      │                 │
│     │    │    - Entry point tₑₙₜᵣᵧ          │      │                 │
│     │    │    - Exit point tₑₓᵢₜ            │      │                 │
│     │    │                                 │      │                 │
│     │    │ 2. Compute color at midpoint    │      │                 │
│     │    │    c = c₀ + ∇c·(midpoint - cc)  │      │                 │
│     │    │    + SH(view_direction)         │      │                 │
│     │    │                                 │      │                 │
│     │    │ 3. Compute alpha                │      │                 │
│     │    │    α = 1 - exp(-σ · length)     │      │                 │
│     │    │                                 │      │                 │
│     │    │ 4. Alpha-blend                  │      │                 │
│     │    │    C += T · α · c               │      │                 │
│     │    │    T *= (1 - α)                 │      │                 │
│     │    └─────────────────────────────────┘      │                 │
│     │                                             │                 │
│     │  Early termination if T < threshold        │                 │
│     └─────────────────────────────────────────────┘                 │
│                                                                     │
│  4. OUTPUT                                                          │
│  ─────────                                                          │
│     ┌─────────────────────┐                                         │
│     │                     │                                         │
│     │   Final RGB Image   │                                         │
│     │                     │                                         │
│     └─────────────────────┘                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Code: Main render function** (`utils/train_util.py`)
```python
def render(camera: Camera, model, cell_values=None, tile_size=16, min_t=0.1,
           scene_scaling=1, clip_multi=0, ray_jitter=None, **kwargs):
    """
    Render a single image from the given camera viewpoint.

    Args:
        camera: Camera object with intrinsics/extrinsics
        model: Radiance mesh model
        tile_size: Size of screen tiles for rasterization
        min_t: Minimum ray distance (near plane)

    Returns:
        Dictionary with rendered image and auxiliary outputs
    """
    device = model.device
    vertices = model.vertices

    # 1. Set up tile-based render grid (like 3DGS)
    render_grid = RenderGrid(camera.image_height, camera.image_width,
                             tile_height=tile_size, tile_width=tile_size)

    # 2. VERTEX SHADER: Project tetrahedra, compute per-tile assignments
    tcam = dict(
        tile_height=tile_size, tile_width=tile_size,
        grid_height=render_grid.grid_height, grid_width=render_grid.grid_width,
        min_t=min_t, **camera.to_dict(device)
    )
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        model.indices, vertices, tcam, render_grid)

    # 3. Compute per-tetrahedron features (if not precomputed)
    if cell_values is None:
        cell_values = torch.zeros((mask.shape[0], model.feature_dim), device=device)
        if mask.sum() > 0 and model.mask_values:
            shs, values = model.get_cell_values(camera, mask, circumcenter[mask])
            cell_values[mask] = values
        else:
            shs, cell_values = model.get_cell_values(camera, all_circumcenters=circumcenter)

    # 4. PIXEL SHADER: Ray-tet intersection + alpha blending
    image_rgb, xyzd_img, distortion_img, tet_alive = AlphaBlendTiledRender.apply(
        sorted_tetra_idx, tile_ranges, model.indices, vertices,
        cell_values, render_grid, tcam, ray_jitter)

    # 5. Post-processing
    alpha = image_rgb.permute(2,0,1)[3, ...]
    render_pkg = {
        'render': image_rgb.permute(2,0,1)[:3, ...] * camera.gt_alpha_mask.to(device),
        'alpha': alpha,
        'distortion_loss': distortion_loss.mean(),
        'mask': mask,
        'xyzd': rxyzd_img,
        # ...
    }
    return render_pkg
```

### 4.2 Ray-Tetrahedron Intersection

```
    RAY-TETRAHEDRON INTERSECTION
    ════════════════════════════

    Ray: r(t) = o + t·d

    Tetrahedron: 4 triangular faces

              v₂
              ╱╲
             ╱  ╲
            ╱    ╲
           ╱  ●───╲────────► ray
          ╱   │    ╲
        v₀────│─────v₁
          ╲   │    ╱
           ╲  │   ╱
            ╲ │  ╱
             ╲│ ╱
              v₃

    Algorithm:
    1. Test ray against each of the 4 faces
    2. Find entry point (first intersection)
    3. Find exit point (second intersection)
    4. Segment length = |exit - entry|

    For rendering:
    - Compute barycentric coordinates at entry/exit
    - Interpolate color using gradient field
```

### 4.3 Volume Rendering Equation (Discretized)

```
    VOLUME RENDERING
    ════════════════

    Continuous:

    C(r) = ∫₀^∞ T(t) · σ(r(t)) · c(r(t), d) dt

    Where: T(t) = exp(-∫₀^t σ(r(s)) ds)


    Discretized (per tetrahedron):

    C = Σᵢ Tᵢ · αᵢ · cᵢ

    Where:
    - Tᵢ = Πⱼ<ᵢ (1 - αⱼ)     (accumulated transmittance)
    - αᵢ = 1 - exp(-σᵢ · δᵢ)  (alpha for tet i)
    - δᵢ = ray segment length through tet i
    - cᵢ = color at ray midpoint in tet i


    Visual:

    Ray ──────●──────●──────●──────●──────►
              │      │      │      │
            tet₀   tet₁   tet₂   tet₃
              │      │      │      │
              ▼      ▼      ▼      ▼
             T₀α₀c₀ T₁α₁c₁ T₂α₂c₂ T₃α₃c₃
              │      │      │      │
              └──────┴──────┴──────┘
                        │
                        ▼
                   Final Color C
```

---

## 5. Training Process <a name="5-training-process"></a>

### 5.1 Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PHASE 1: Dynamic Mesh (iterations 0 - 18,000)                      │
│  ═════════════════════════════════════════════                      │
│                                                                     │
│    ┌─────────────┐                                                  │
│    │ Point Cloud │                                                  │
│    │  (from SfM) │                                                  │
│    └──────┬──────┘                                                  │
│           │                                                         │
│           ▼                                                         │
│    ┌─────────────────────────────────────────┐                      │
│    │         TRAINING LOOP                   │                      │
│    │  ┌───────────────────────────────────┐  │                      │
│    │  │  1. Delaunay Triangulation        │  │  Every 10 iters     │
│    │  │     (vertices → tetrahedra)       │  │                      │
│    │  └───────────────────────────────────┘  │                      │
│    │                  │                      │                      │
│    │                  ▼                      │                      │
│    │  ┌───────────────────────────────────┐  │                      │
│    │  │  2. Render Image                  │  │                      │
│    │  │     - Compute features (iNGP)     │  │                      │
│    │  │     - Ray-tet intersection        │  │                      │
│    │  │     - Alpha blending              │  │                      │
│    │  └───────────────────────────────────┘  │                      │
│    │                  │                      │                      │
│    │                  ▼                      │                      │
│    │  ┌───────────────────────────────────┐  │                      │
│    │  │  3. Compute Loss                  │  │                      │
│    │  │     L = λ₁·L₁ + λ₂·LSSIM         │  │                      │
│    │  └───────────────────────────────────┘  │                      │
│    │                  │                      │                      │
│    │                  ▼                      │                      │
│    │  ┌───────────────────────────────────┐  │                      │
│    │  │  4. Backprop & Update             │  │                      │
│    │  │     - Hash grid parameters        │  │                      │
│    │  │     - MLP weights                 │  │                      │
│    │  │     - Vertex positions            │  │                      │
│    │  └───────────────────────────────────┘  │                      │
│    │                  │                      │                      │
│    │                  ▼                      │                      │
│    │  ┌───────────────────────────────────┐  │                      │
│    │  │  5. Densification (every 500)     │  │  Iterations         │
│    │  │     - Clone under-reconstructed   │  │  2000 - 16000       │
│    │  │     - Split over-reconstructed    │  │                      │
│    │  │     - Prune low-contribution      │  │                      │
│    │  └───────────────────────────────────┘  │                      │
│    └─────────────────────────────────────────┘                      │
│                                                                     │
│  PHASE 2: Frozen Mesh (iterations 18,000 - 30,000)                  │
│  ════════════════════════════════════════════════                   │
│                                                                     │
│    ┌─────────────────────────────────────────┐                      │
│    │  - Freeze mesh topology                 │                      │
│    │  - Continue optimizing:                 │                      │
│    │    • Vertex positions (fine-tuning)     │                      │
│    │    • Per-tet density                    │                      │
│    │    • Per-tet color/gradient/SH          │                      │
│    └─────────────────────────────────────────┘                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Delaunay Triangulation

```
    DELAUNAY TRIANGULATION
    ══════════════════════

    Input: Set of 3D points (vertices)
    Output: Tetrahedral mesh with Delaunay property

    Delaunay Property:
    ─────────────────
    No vertex lies inside the circumsphere of any tetrahedron


           Bad (non-Delaunay):              Good (Delaunay):

              ╱╲                                 ╱╲
             ╱  ╲                               ╱  ╲
            ╱ ●  ╲  ◄── vertex inside         ╱    ╲
           ╱  ○   ╲     circumsphere         ╱  ○   ╲  ◄── no vertex
          ╱________╲                        ╱________╲     inside


    Why Delaunay?
    ─────────────
    1. Maximizes minimum angle (avoids skinny tetrahedra)
    2. Unique for points in general position
    3. Efficient algorithms exist (O(n log n))
    4. Good for interpolation and rendering


    Implementation (gDel3D - GPU Delaunay):
    ──────────────────────────────────────
    - Parallel GPU algorithm
    - Incremental insertion with flipping
    - ~10ms for 100k points
```

**Code: Delaunay triangulation update** (`models/ingp_color.py`)
```python
@torch.no_grad()
def update_triangulation(self, high_precision=False, density_threshold=0.0, alpha_threshold=0.0):
    """
    Recompute Delaunay triangulation from current vertices.
    Called every ~10 iterations during training.
    """
    torch.cuda.empty_cache()
    verts = self.vertices  # Interior + exterior vertices

    if high_precision:
        # Use SciPy for high precision (slower)
        indices_np = Delaunay(verts.detach().cpu().numpy()).simplices.astype(np.int32)
    else:
        # Use GPU Delaunay (gDel3D) - fast parallel algorithm
        v = Del(verts.shape[0])
        indices_np, prev = v.compute(verts.detach().cpu().double())
        indices_np = indices_np.numpy()
        # Filter invalid tetrahedra (may reference dummy vertices)
        indices_np = indices_np[(indices_np < verts.shape[0]).all(axis=1)]

    # Ensure positive volume (consistent winding order)
    indices = torch.as_tensor(indices_np).cuda()
    vols = topo_utils.tet_volumes(verts[indices])
    reverse_mask = vols < 0
    if reverse_mask.sum() > 0:
        # Swap two vertices to flip orientation
        indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

    self.indices = indices

    # Optional: cull low-density tetrahedra
    if density_threshold > 0 or alpha_threshold > 0:
        tet_density = self.calc_tet_density()
        tet_alpha = self.calc_tet_alpha(mode="min", density=tet_density)
        mask = (tet_density > density_threshold) | (tet_alpha > alpha_threshold)
        self.empty_indices = self.indices[~mask]  # Store culled tets
        self.indices = self.indices[mask]
```

### 5.3 Densification Strategy

```
    ADAPTIVE DENSIFICATION
    ══════════════════════

    Goal: Adaptively add/remove vertices where needed


    1. CLONE (under-reconstruction)
    ───────────────────────────────

    Problem: Area not well covered

    Before:                      After:
         ╱╲                           ╱╲
        ╱  ╲                         ╱  ╲
       ╱    ╲  Large tet,           ╱ ╲╱ ╲  New vertex
      ╱      ╲  poor detail        ╱__╲╱__╲  added
     ╱________╲

    Trigger: High error, low contribution


    2. SPLIT (over-reconstruction)
    ──────────────────────────────

    Problem: Trying to represent too much detail

    Before:                      After:
         ●                            ●     ●
         │                            │     │
    ─────●─────  Single vertex   ─────●─────●───── Split into
                 with high             multiple
                 gradient

    Trigger: High gradient, high contribution


    3. PRUNE (low contribution)
    ───────────────────────────

    Problem: Vertices not visible from any view

    Before:                      After:
     ● ● ● ●                      ● ●   ●
      ●   ●                        ●   ●
       ● ●                           ●
        ●  ◄── rarely visible       (removed)

    Trigger: Low peak transmittance across all views


    Contribution Metric:
    ────────────────────
    For each tetrahedron, track:
    - Peak transmittance T when tet is hit
    - Number of rays passing through
    - Error contribution to final image
```

**Code: Collecting densification statistics** (`utils/densification.py`)
```python
@torch.no_grad()
def collect_render_stats(sampled_cameras, model, args, device):
    """
    Accumulate densification statistics across multiple camera views.
    Used to decide which tetrahedra to clone/split.
    """
    n_tets = model.indices.shape[0]

    # Accumulators for statistics
    peak_contrib = torch.zeros((n_tets), device=device)      # Max transmittance
    total_var_moments = torch.zeros((n_tets, 3), device=device)  # Error variance
    top_ssim = torch.zeros((n_tets, 2), device=device)       # Top-2 SSIM errors
    within_var_rays = torch.zeros((n_tets, 2, 6), device=device)  # Ray segments

    for cam in sampled_cameras:
        target = cam.original_image.cuda()

        # Render with error tracking
        image_votes, extras = render_err(target, cam, model, tile_size=args.tile_size)

        # Track peak contribution (max transmittance when tet is visible)
        tc = extras["tet_count"][..., 0]  # Number of rays hitting this tet
        max_T = extras["tet_count"][..., 1].float() / 65535
        peak_contrib = torch.maximum(max_T, peak_contrib)

        # Track error variance within each tet
        image_T, image_err, image_err2 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
        N = tc
        within_var_mu = safe_div(image_err, N)
        within_var_std = (safe_div(image_err2, N) - within_var_mu**2).clip(min=0)

        # Accumulate across views
        total_var_moments[:, 0] += image_T
        total_var_moments[:, 1] += image_err
        total_var_moments[:, 2] += image_err2

    return RenderStats(
        within_var_rays=within_var_rays,
        total_var_moments=total_var_moments,
        peak_contrib=peak_contrib,
        top_ssim=top_ssim,
        ...
    )
```

**Code: Applying densification (clone/split)** (`utils/densification.py`)
```python
@torch.no_grad()
def apply_densification(stats, model, tet_optim, args, iteration, device, ...):
    """
    Convert accumulated statistics into actual vertex cloning/splitting.
    """
    # Compute variance scores
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    total_var_mu = safe_div(s1_t, s0_t)
    total_var_std = (safe_div(s2_t, s0_t) - total_var_mu**2).clip(min=0)

    # Compute within-image variance (from top-2 SSIM errors)
    within_var = (stats.top_ssim).sum(dim=1)
    total_var = s0_t * total_var_std

    # Mask out tets with low contribution (not visible)
    total_var[stats.peak_contrib < args.clone_min_contrib] = 0
    within_var[stats.peak_contrib < args.split_min_contrib] = 0

    # Decide which tets to clone (high total variance) or split (high within variance)
    within_mask = (within_var > args.within_thresh)
    total_mask = (total_var > args.total_thresh)
    clone_mask = within_mask | total_mask

    # Limit number of new vertices per iteration
    if clone_mask.sum() > target_addition:
        true_indices = clone_mask.nonzero().squeeze(-1)
        perm = torch.randperm(true_indices.size(0))
        selected_indices = true_indices[perm[:target_addition]]
        clone_mask = torch.zeros_like(clone_mask, dtype=torch.bool)
        clone_mask[selected_indices] = True

    # Compute split points (intersection of top-2 rays through each tet)
    split_point, bad = get_approx_ray_intersections(stats.within_var_rays)
    split_point = split_point[clone_mask]

    # Add new vertices
    tet_optim.split(split_point)

    print(f"#Grow: {total_mask.sum():4d} #Split: {within_mask.sum():4d}")
```

### 5.4 Loss Functions

```
    LOSS FUNCTIONS
    ══════════════

    Total Loss:

    L = (1 - λ_ssim) · L₁ + λ_ssim · L_SSIM + λ_dist · L_dist + λ_sh · L_sh


    1. L1 Loss (Photometric):

       L₁ = |I_pred - I_gt| · mask

       Simple pixel-wise difference


    2. SSIM Loss (Structural Similarity):

       L_SSIM = 1 - SSIM(I_pred, I_gt)

       Captures structural patterns, luminance, contrast


    3. Distortion Loss (Ray regularization):

                  Σᵢ Σⱼ wᵢwⱼ|tᵢ - tⱼ|
       L_dist = ─────────────────────
                      (Σᵢ wᵢ)²

       Encourages weights to concentrate (reduces floaters)


    4. SH Regularization:

       L_sh = ||higher_degree_SH_coeffs||²

       Penalizes high-frequency view-dependent effects
```

---

## 6. Comparison with NeRF and Gaussian Splatting <a name="6-comparison"></a>

### 6.1 Representation Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REPRESENTATION COMPARISON                        │
├─────────────────┬─────────────────┬─────────────────┬───────────────┤
│                 │      NeRF       │    3D Gaussians │Radiance Meshes│
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│ Primitive       │ Continuous      │ 3D Ellipsoids   │ Tetrahedra    │
│                 │ function        │                 │               │
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│ Storage         │ MLP weights     │ Per-Gaussian:   │ Per-tet:      │
│                 │ (~1-5 MB)       │ μ,Σ,c,α,SH      │ σ,c,∇c,SH     │
│                 │                 │ (~100-500 MB)   │ (~50-200 MB)  │
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│ Geometry        │ Implicit        │ None            │ Explicit mesh │
│                 │ (via density)   │ (point cloud)   │ (watertight)  │
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│ Topology        │ None            │ None            │ Delaunay      │
│                 │                 │                 │ triangulation │
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│ Rendering       │ Ray marching    │ Splatting       │ Ray-tet       │
│                 │ (many samples)  │ (rasterization) │ intersection  │
├─────────────────┼─────────────────┼─────────────────┼───────────────┤
│ View-dependent  │ MLP input       │ Spherical       │ Spherical     │
│ effects         │                 │ Harmonics       │ Harmonics     │
└─────────────────┴─────────────────┴─────────────────┴───────────────┘
```

### 6.2 Rendering Pipeline Comparison

```
    NeRF:
    ─────
    Ray ──●──●──●──●──●──●──●──●──●──●──►
          │  │  │  │  │  │  │  │  │  │
         MLP MLP MLP MLP MLP MLP MLP MLP MLP MLP
          │  │  │  │  │  │  │  │  │  │
          ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
         σ,c σ,c σ,c ...              (many samples, slow)


    3D Gaussian Splatting:
    ──────────────────────
    Gaussians    Project     Sort      Blend
       ⬭         ──►  ○      ──►  ○    ──►  pixel
       ⬭              ○           ○
       ⬭              ○           ○         (fast, but no geometry)


    Radiance Meshes:
    ────────────────
    Ray ─────────────────────────────────►
           │         │         │
          tet₀      tet₁      tet₂
           │         │         │
         entry     entry     entry
         exit      exit      exit
           │         │         │
           ▼         ▼         ▼
          α₀c₀     α₁c₁     α₂c₂     (sparse intersections, explicit geometry)
```

### 6.3 Quality vs Speed Trade-offs

```
                    Quality
                       ▲
                       │
             NeRF ●    │
                       │
                       │    ● Radiance Meshes
                       │
                       │         ● 3DGS
                       │
                       └──────────────────────► Speed

    Training Time:
    - NeRF: Hours
    - 3DGS: Minutes
    - Radiance Meshes: ~30 minutes

    Rendering Speed:
    - NeRF: < 1 FPS
    - 3DGS: 100+ FPS
    - Radiance Meshes: 30-60 FPS

    Geometry Quality:
    - NeRF: Implicit (noisy meshes via marching cubes)
    - 3DGS: None (point cloud only)
    - Radiance Meshes: Explicit tetrahedral mesh
```

---

## 7. Mathematical Foundations <a name="7-mathematical-foundations"></a>

### 7.1 Circumcenter and Circumsphere

For a tetrahedron with vertices v₀, v₁, v₂, v₃:

```
    CIRCUMSPHERE COMPUTATION
    ════════════════════════

    The circumcenter is equidistant from all 4 vertices.

    Given vectors from v₀:
    - a = v₁ - v₀
    - b = v₂ - v₀
    - c = v₃ - v₀

    Circumcenter:

              |a|²(b × c) + |b|²(c × a) + |c|²(a × b)
    cc = v₀ + ─────────────────────────────────────────
                        2 · a · (b × c)


    Circumradius:

    r = |cc - v₀| = |cc - v₁| = |cc - v₂| = |cc - v₃|


    Why circumcenter?
    ─────────────────
    - Natural "center" of tetrahedron
    - Query point for hash grid (more stable than centroid)
    - Defines circumsphere for Delaunay property
```

**Code: Circumcenter computation** (`utils/topo_utils.py`)
```python
def calculate_circumcenters_torch(vertices: torch.Tensor):
    """
    Compute the circumcenter and circumradius of tetrahedra.

    Args:
        vertices: Tensor of shape (..., 4, 3) containing tetrahedron vertices

    Returns:
        circumcenter: (..., 3) circumcenter coordinates
        radius: (...,) circumradius
    """
    # Compute vectors from v0 to other vertices
    a = vertices[..., 1, :] - vertices[..., 0, :]  # v1 - v0
    b = vertices[..., 2, :] - vertices[..., 0, :]  # v2 - v0
    c = vertices[..., 3, :] - vertices[..., 0, :]  # v3 - v0

    # Compute squared lengths
    aa = torch.sum(a * a, dim=-1, keepdim=True)  # |a|²
    bb = torch.sum(b * b, dim=-1, keepdim=True)  # |b|²
    cc = torch.sum(c * c, dim=-1, keepdim=True)  # |c|²

    # Compute cross products
    cross_bc = torch.cross(b, c, dim=-1)
    cross_ca = torch.cross(c, a, dim=-1)
    cross_ab = torch.cross(a, b, dim=-1)

    # Denominator: 2 * a · (b × c) = 2 * scalar triple product = 6 * volume
    denominator = 2.0 * torch.sum(a * cross_bc, dim=-1, keepdim=True)

    # Circumcenter formula: cc = v0 + (|a|²(b×c) + |b|²(c×a) + |c|²(a×b)) / (2·a·(b×c))
    relative_circumcenter = safe_div(
        aa * cross_bc + bb * cross_ca + cc * cross_ab,
        denominator
    )

    # Circumradius
    radius = torch.norm(a - relative_circumcenter, dim=-1)

    return vertices[..., 0, :] + relative_circumcenter, radius
```

### 7.2 Barycentric Coordinates

```
    BARYCENTRIC COORDINATES
    ═══════════════════════

    Any point p inside tetrahedron can be written as:

    p = λ₀v₀ + λ₁v₁ + λ₂v₂ + λ₃v₃

    Where: λ₀ + λ₁ + λ₂ + λ₃ = 1
           λᵢ ≥ 0 for all i


    Computation:
    ────────────
    Let T = [v₁-v₀, v₂-v₀, v₃-v₀]ᵀ  (3x3 matrix)

    Solve: T · [λ₁, λ₂, λ₃]ᵀ = p - v₀

    Then: λ₀ = 1 - λ₁ - λ₂ - λ₃


    Use in rendering:
    ─────────────────
    - Interpolate vertex colors: c(p) = Σᵢ λᵢcᵢ
    - Check if point inside tet: all λᵢ ≥ 0
    - Smooth color gradients across tet
```

**Code: Barycentric coordinate computation** (`utils/topo_utils.py`)
```python
@torch.jit.script
def calc_barycentric(points, tets):
    """
    Compute barycentric coordinates for points inside tetrahedra.

    Args:
        points: (N, 3) query points
        tets: (N, 4, 3) tetrahedron vertices

    Returns:
        bary: (N, 4) barycentric coordinates [λ₀, λ₁, λ₂, λ₃]
    """
    v0 = tets[:, 0, :]                           # (N, 3)
    T = tets[:, 1:, :] - v0.unsqueeze(1)         # (N, 3, 3)
    T = T.permute(0, 2, 1)                        # Transpose for solving

    # Solve T · x = (p - v0) for barycentric coords of v1, v2, v3
    p_minus_v0 = points - v0                     # (N, 3)
    x = torch.linalg.solve(T, p_minus_v0.unsqueeze(2)).squeeze(2)  # (N, 3)

    # Compute weight for v0 (must sum to 1)
    w0 = 1 - x.sum(dim=1, keepdim=True)          # (N, 1)
    bary = torch.cat([w0, x], dim=1)              # (N, 4)

    # Clamp to ensure non-negative (for points outside tet)
    bary = bary.clip(min=0)
    return bary
```

**Code: Tetrahedron volume computation** (`utils/topo_utils.py`)
```python
def tet_volumes(tets):
    """
    Compute signed volumes of tetrahedra.

    Volume = (1/6) * det([v1-v0, v2-v0, v3-v0])

    Positive volume indicates consistent vertex ordering.
    Negative volume means vertices need to be swapped.

    Args:
        tets: (T, 4, 3) tetrahedron vertices

    Returns:
        vol: (T,) signed volumes
    """
    v0, v1, v2, v3 = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]

    # Edge vectors from v0
    a = v1 - v0
    b = v2 - v0
    c = v3 - v0

    # Stack into 3x3 matrices and compute determinant
    mat = torch.stack((a, b, c), dim=1)  # (T, 3, 3)
    det = torch.det(mat)

    vol = det / 6.0
    return vol
```

### 7.3 Volume Rendering in Tetrahedra

```
    VOLUME RENDERING MATH
    ═════════════════════

    For ray segment through tetrahedron with:
    - Entry point: p_entry
    - Exit point: p_exit
    - Constant density: σ
    - Linear color field: c(p) = c₀ + ∇c · (p - cc)


    Segment length:

    δ = |p_exit - p_entry|


    Alpha (opacity):

    α = 1 - exp(-σ · δ)


    Color at midpoint:

    p_mid = (p_entry + p_exit) / 2
    c = c₀ + ∇c · (p_mid - cc) + SH(view_dir)


    Contribution to pixel:

    C += T · α · c
    T *= (1 - α)


    Gradient flow:
    ──────────────
    ∂L/∂σ = ∂L/∂α · ∂α/∂σ
          = ∂L/∂α · δ · exp(-σδ)

    ∂L/∂c = T · α · ∂L/∂C

    ∂L/∂v = ∂L/∂δ · ∂δ/∂v + ∂L/∂c · ∂c/∂v
```

### 7.4 Hash Grid Encoding

```
    INSTANT-NGP HASH ENCODING
    ═════════════════════════

    Multi-resolution hash tables:

    For level l with resolution Nₗ:

    Nₗ = floor(N_min · b^l)

    Where:
    - N_min = base_resolution (e.g., 64)
    - b = per_level_scale (e.g., 2)
    - l = 0, 1, ..., L-1


    For query point x ∈ [0,1]³:

    1. Scale to grid: x_grid = x · Nₗ

    2. Find surrounding vertices (8 corners of voxel)

    3. Hash each corner:
       h(v) = (⊕ᵢ vᵢ · πᵢ) mod T

       Where πᵢ are large primes, T is table size

    4. Lookup features F[h(v)] for each corner

    5. Trilinear interpolation

    6. Concatenate all levels: [F₀, F₁, ..., F_L]


    Benefits:
    ─────────
    - O(1) lookup (vs O(n) for octree)
    - Compact storage (hash collisions are OK)
    - Multi-resolution captures both coarse and fine detail
```

**Code: Hash grid implementation** (`utils/hashgrid.py`)
```python
class HashEmbedderOptimized(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level

        # Compute per-level resolutions (geometric progression)
        self.b = torch.exp(
            (finest_resolution.log() - base_resolution.log()) / (n_levels - 1)
        )
        level_resolutions = [torch.floor(base_resolution * (self.b ** i)) for i in range(n_levels)]

        # Per-level embedding tables (hash tables)
        # Size: min(2^log2_hashmap_size, resolution^3) to avoid waste
        embedding_sizes = [
            min(2**log2_hashmap_size, int(res.item())**3)
            for res in level_resolutions
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, n_features_per_level)
            for size in embedding_sizes
        ])

        # Large primes for spatial hashing
        self.primes = torch.tensor([1, 2654435761, 805459861, 3674653429, ...])
        self.hash_mask = (1 << log2_hashmap_size) - 1

    def ingp_hash(self, coords):
        """Spatial hash function: h(x,y,z) = (x*p1 ^ y*p2 ^ z*p3) & mask"""
        xor_result = torch.zeros_like(coords[..., 0])
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i] * self.primes[i]
        return xor_result & self.hash_mask

    def forward(self, x):
        """Query hash grid at position x ∈ [0,1]³"""
        x_embedded_all = []
        for i in range(self.n_levels):
            # 1. Get voxel corners and interpolation weights
            fracs, hashed_idxs = self.get_voxel_vertices(x, i)

            # 2. Lookup embeddings for 8 corners of voxel
            corner_embs = self.embeddings[i](hashed_idxs)

            # 3. Trilinear interpolation
            x_embedded = self.trilinear_interp_direct(fracs, corner_embs)
            x_embedded_all.append(x_embedded)

        # 4. Concatenate all levels
        return torch.cat(x_embedded_all, dim=-1)
```

### 7.5 Delaunay-Voronoi Duality

Radiance Meshes operates on the **Delaunay triangulation**, while the related work **Radiant Foam** (Govindarajan et al., ICCV 2025) uses the **Voronoi diagram**. These are dual structures of each other:

```
    DELAUNAY-VORONOI DUALITY
    ════════════════════════

    Given a set of points (sites) in 3D space:

         ●₁    ●₂
           ●₃
         ●₄    ●₅


    DELAUNAY TRIANGULATION              VORONOI DIAGRAM
    (Used by Radiance Meshes)           (Used by Radiant Foam)
    ─────────────────────────           ───────────────────────

         ●₁────●₂                       ┌─────┬─────┐
          ╲  ╱  ╲                       │  1  │  2  │
           ●₃────                       ├──┬──┼──┬──┤
          ╱  ╲  ╱                       │  │ 3│  │  │
         ●₄────●₅                       ├──┴──┼──┴──┤
                                        │  4  │  5  │
    Tetrahedra connect                  └─────┴─────┘
    neighboring points
                                        Each cell contains all
    Edge exists iff points              points closer to site i
    share a Voronoi face                than any other site


    THE DUALITY RELATIONSHIP
    ════════════════════════

    ┌────────────────────┬────────────────────┐
    │     Delaunay       │      Voronoi       │
    ├────────────────────┼────────────────────┤
    │  Vertex (point)    │   Cell (region)    │
    │  Edge              │   Face             │
    │  Triangle (2D)     │   Edge             │
    │  Tetrahedron (3D)  │   Vertex           │
    └────────────────────┴────────────────────┘

    Key insight: Delaunay tetrahedron ↔ Voronoi vertex
                 (circumcenter of tet = Voronoi vertex!)


    WHY BOTH REPRESENTATIONS WORK
    ═════════════════════════════

    1. CONTINUOUS DIFFERENTIABILITY
       ────────────────────────────
       When a site (point) moves continuously:
       - Voronoi cell boundaries move continuously
       - Delaunay tetrahedra vertices move continuously

       → Gradients flow smoothly during optimization
       → No discrete jumps (except at topology changes)

    2. EFFICIENT RAY TRAVERSAL
       ───────────────────────

       Delaunay (Radiance Meshes):
       Ray ────●────●────●────►
              tet₀  tet₁  tet₂

       Voronoi (Radiant Foam):
       Ray ────●────●────●────►
             cell₀ cell₁ cell₂

       Both allow O(1) neighbor lookup:
       - Delaunay: tetrahedra share faces
       - Voronoi: cells share faces (= Delaunay edges)

    3. CIRCUMCENTER CONNECTION
       ───────────────────────
       The circumcenter of a Delaunay tetrahedron
       IS the corresponding Voronoi vertex.

       This is why Radiance Meshes queries the hash grid
       at circumcenters - it's querying at Voronoi vertices!

              Delaunay tet              Voronoi vertex
                  ╱╲
                 ╱  ╲                        ●  ← circumcenter
                ╱ ●  ╲                          = Voronoi vertex
               ╱      ╲
              ╱________╲


    COMPARISON: RADIANCE MESHES vs RADIANT FOAM
    ════════════════════════════════════════════

    ┌──────────────────┬─────────────────────┬─────────────────────┐
    │     Aspect       │   Radiance Meshes   │    Radiant Foam     │
    ├──────────────────┼─────────────────────┼─────────────────────┤
    │ Representation   │ Delaunay tetrahedra │ Voronoi cells       │
    │ Primitives       │ ~500k-2M tets       │ ~500k-2M cells      │
    │ Feature location │ Circumcenter (=Vor) │ Site (=Del vertex)  │
    │ Ray traversal    │ Tet-to-tet via face │ Cell-to-cell        │
    │ Rendering        │ Ray-tet intersection│ Volumetric foam     │
    │ Light transport  │ Basic               │ Reflection/refract  │
    │ GPU Delaunay     │ gDel3D              │ Custom impl         │
    │ Speed            │ 30-60 FPS           │ Real-time           │
    │ Venue            │ -                   │ ICCV 2025 Highlight │
    └──────────────────┴─────────────────────┴─────────────────────┘

    Both methods leverage the same underlying mathematical structure!
```

The key insight is that **Radiance Meshes and Radiant Foam are two views of the same coin**:
- Radiance Meshes: "Tetrahedra partition space, store features at circumcenters"
- Radiant Foam: "Voronoi cells partition space, store features at sites"

The circumcenter-based feature query in Radiance Meshes is mathematically equivalent to storing features at Voronoi vertices in Radiant Foam.

---

## 8. Key Innovations <a name="8-key-innovations"></a>

### 8.1 Innovation Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                      KEY INNOVATIONS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. TETRAHEDRAL MESH AS SCENE REPRESENTATION                        │
│     ─────────────────────────────────────────                       │
│     • Combines explicit geometry with volumetric rendering          │
│     • Delaunay triangulation ensures good mesh quality              │
│     • Watertight geometry (vs floaters in 3DGS)                     │
│                                                                     │
│  2. CIRCUMCENTER-BASED FEATURE QUERIES                              │
│     ──────────────────────────────────────                          │
│     • Use circumcenter (not centroid) for hash grid queries         │
│     • More stable for skinny tetrahedra                             │
│     • Natural integration with Delaunay property                    │
│                                                                     │
│  3. PER-TETRAHEDRON COLOR GRADIENTS                                 │
│     ───────────────────────────────────                             │
│     • Linear color field within each tet                            │
│     • c(p) = c₀ + ∇c · (p - cc)                                     │
│     • Smoother rendering with fewer primitives                      │
│                                                                     │
│  4. ADAPTIVE MESH REFINEMENT                                        │
│     ─────────────────────────────                                   │
│     • Clone/split/prune based on rendering statistics               │
│     • Vertices move to where detail is needed                       │
│     • Budget control for different platforms                        │
│                                                                     │
│  5. TWO-PHASE TRAINING                                              │
│     ─────────────────────                                           │
│     • Phase 1: Dynamic mesh (topology changes)                      │
│     • Phase 2: Frozen mesh (feature optimization)                   │
│     • Balances flexibility and stability                            │
│                                                                     │
│  6. EFFICIENT GPU RASTERIZATION                                     │
│     ─────────────────────────────                                   │
│     • Tile-based rendering (like 3DGS)                              │
│     • Slang shaders for ray-tet intersection                        │
│     • Parallel alpha blending                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Practical Considerations

```
    WHEN TO USE RADIANCE MESHES
    ═══════════════════════════

    ✓ Good for:
    ───────────
    • Scenes where geometry matters (AR/VR)
    • Export to game engines (mesh output)
    • Moderate real-time requirements (30-60 FPS)
    • Scenes with clear surfaces

    ✗ Less ideal for:
    ─────────────────
    • Highly view-dependent effects (pure reflections)
    • Extremely fine detail (hair, fur)
    • Maximum speed requirements (use 3DGS)
    • Purely implicit applications (use NeRF)


    TUNING PARAMETERS
    ═════════════════

    For quality:
    • ↑ budget (more vertices)
    • ↑ iterations
    • ↓ voxel_size (finer initial sampling)

    For speed/mobile:
    • ↓ budget (250k vertices)
    • ↑ within_thresh, total_thresh
    • ↓ iterations (10k)
    • ↑ voxel_size
```

---

## Appendix: Code Structure

```
radiance_meshes/
├── train.py                 # Main training script
├── models/
│   ├── ingp_color.py       # Model with iNGP backbone
│   ├── frozen.py           # Frozen model (post phase 1)
│   └── base_model.py       # Base class with save2ply
├── utils/
│   ├── train_util.py       # Render function, LR schedulers
│   ├── topo_utils.py       # Circumcenter, barycentric, volumes
│   ├── densification.py    # Clone/split/prune logic
│   ├── model_util.py       # Feature activation, SH conversion
│   └── hashgrid.py         # iNGP hash grid implementation
├── delaunay_rasterization/
│   └── internal/
│       ├── alphablend_*.py # Alpha blending shaders
│       ├── tile_shader_*.py# Tile binning
│       └── slang/          # Slang GPU kernels
├── gDel3D/                  # GPU Delaunay triangulation
└── data/
    ├── loader.py           # Dataset loading
    └── camera.py           # Camera model
```

---

## References

1. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020
2. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
3. Müller et al., "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding", SIGGRAPH 2022
4. Delaunay, "Sur la sphère vide", 1934 (Delaunay triangulation)
5. Radiance Meshes project: https://half-potato.gitlab.io/rm/
6. Govindarajan et al., "Radiant Foam: Real-Time Differentiable Ray Tracing", ICCV 2025 (Highlight)
   - Project: https://radfoam.github.io/
   - Code: https://github.com/theialab/radfoam
   - arXiv: https://arxiv.org/abs/2502.01157
