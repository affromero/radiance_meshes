# Computer Graphics Background for Radiance Meshes

This document provides foundational computer graphics concepts needed to understand Radiance Meshes and related neural scene representations.

## Table of Contents
1. [Basic 3D Geometry](#1-basic-3d-geometry)
2. [Triangles and Meshes](#2-triangles-and-meshes)
3. [Tetrahedra: The 3D Simplex](#3-tetrahedra)
4. [Triangulation](#4-triangulation)
5. [Delaunay Triangulation](#5-delaunay-triangulation)
6. [Voronoi Diagrams](#6-voronoi-diagrams)
7. [Delaunay-Voronoi Duality](#7-delaunay-voronoi-duality)
8. [Barycentric Coordinates](#8-barycentric-coordinates)
9. [Circumcenters and Circumspheres](#9-circumcenters)
10. [Ray Tracing Fundamentals](#10-ray-tracing)
11. [Volume Rendering](#11-volume-rendering)
12. [Spatial Data Structures](#12-spatial-data-structures)

---

## 1. Basic 3D Geometry <a name="1-basic-3d-geometry"></a>

### 1.1 Points and Vectors

```
    COORDINATE SYSTEM
    ═════════════════

           Y
           ▲
           │
           │
           │
           └──────────► X
          ╱
         ╱
        ▼
        Z

    Point P = (x, y, z)

    A point represents a location in 3D space.
    A vector represents a direction and magnitude.

    Vector from A to B:

    v = B - A = (Bx - Ax, By - Ay, Bz - Az)
```

### 1.2 Basic Operations

```
    VECTOR OPERATIONS
    ═════════════════

    1. DOT PRODUCT (scalar result)
       ──────────────────────────
       a · b = ax*bx + ay*by + az*bz

       Geometric meaning: a · b = |a| |b| cos(θ)

       Uses:
       - Find angle between vectors
       - Project one vector onto another
       - Check if vectors are perpendicular (dot = 0)


    2. CROSS PRODUCT (vector result)
       ─────────────────────────────
       a × b = (ay*bz - az*by,
                az*bx - ax*bz,
                ax*by - ay*bx)

       Result is perpendicular to both a and b

            a × b
              ▲
              │
              │
         a ───┼───► b
              │
              │

       Uses:
       - Find surface normal
       - Calculate area of parallelogram/triangle
       - Determine winding order


    3. MAGNITUDE (length)
       ──────────────────
       |v| = sqrt(vx² + vy² + vz²)


    4. NORMALIZATION (unit vector)
       ───────────────────────────
       v̂ = v / |v|

       Makes vector length = 1, preserves direction
```

### 1.3 Geometric Primitives

```
    PRIMITIVES IN INCREASING DIMENSION
    ═══════════════════════════════════

    0D: Point      ●

    1D: Edge       ●─────────●
                   v₀        v₁

    2D: Triangle       v₂
                       ╱╲
                      ╱  ╲
                     ╱    ╲
                   v₀──────v₁

    3D: Tetrahedron    v₃
                       ╱│╲
                      ╱ │ ╲
                     ╱  │  ╲
                   v₀───│───v₁
                     ╲  │  ╱
                      ╲ │ ╱
                       ╲│╱
                        v₂

    These are called "simplices" (singular: simplex)
    - 0-simplex: point
    - 1-simplex: line segment
    - 2-simplex: triangle
    - 3-simplex: tetrahedron
```

---

## 2. Triangles and Meshes <a name="2-triangles-and-meshes"></a>

### 2.1 Why Triangles?

```
    WHY TRIANGLES ARE FUNDAMENTAL
    ═════════════════════════════

    1. ALWAYS PLANAR
       Any 3 points define a unique plane
       (4+ points might not be coplanar)

           ●
          ╱ ╲
         ╱   ╲     ← Always flat!
        ●─────●


    2. ALWAYS CONVEX
       No "dents" possible with only 3 vertices

       Triangle:        Quad (can be concave):
           ●                  ●───●
          ╱ ╲                  ╲ ╱
         ╱   ╲                  ●
        ●─────●                ╱ ╲
                              ●───●
       Always convex      Can have concavity


    3. SIMPLE INTERPOLATION
       Barycentric coordinates give unique weights
       for any point inside the triangle


    4. HARDWARE OPTIMIZED
       GPUs are designed to rasterize triangles
       extremely efficiently (billions per second)
```

### 2.2 Triangle Mesh

```
    TRIANGLE MESH REPRESENTATION
    ════════════════════════════

    A mesh consists of:

    1. VERTEX LIST (positions)
       ────────────────────────
       V = [(x₀,y₀,z₀), (x₁,y₁,z₁), (x₂,y₂,z₂), ...]

       Index:  0          1          2


    2. FACE LIST (triangles as vertex indices)
       ────────────────────────────────────────
       F = [(0,1,2), (1,3,2), (0,2,4), ...]

       Each triplet defines one triangle


    Example: A simple pyramid

         v4 (apex)
          ╱│╲
         ╱ │ ╲
        ╱  │  ╲
      v0───┼───v1
        ╲  │  ╱
         ╲ │ ╱
          ╲│╱
        v2─┴─v3

    Vertices: [(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0.5,0.5,1)]
    Faces:    [(0,1,4), (1,3,4), (3,2,4), (2,0,4), (0,2,3,1)]
                 ↑        ↑        ↑        ↑        ↑
              front    right     back     left     bottom


    WHY INDEX-BASED?
    ─────────────────
    - Vertices are shared between triangles
    - Saves memory (store position once)
    - Enables smooth shading (shared normals)
    - Easier to edit (move vertex, all faces update)
```

### 2.3 Surface Normal

```
    COMPUTING TRIANGLE NORMAL
    ═════════════════════════

    Given triangle with vertices v₀, v₁, v₂:

         v₂
         ╱╲
        ╱  ╲
       ╱  n ╲   ← normal points "out"
      ╱   ▲  ╲
     ╱    │   ╲
   v₀─────┴────v₁

    Edge vectors:
    e₁ = v₁ - v₀
    e₂ = v₂ - v₀

    Normal (unnormalized):
    n = e₁ × e₂

    Unit normal:
    n̂ = n / |n|

    WINDING ORDER MATTERS!
    ──────────────────────
    Counter-clockwise (CCW) → normal points toward viewer
    Clockwise (CW) → normal points away

    v₀ → v₁ → v₂ (CCW)     v₀ → v₂ → v₁ (CW)
         n ▲                    │ n
           │                    ▼
```

---

## 3. Tetrahedra: The 3D Simplex <a name="3-tetrahedra"></a>

### 3.1 What is a Tetrahedron?

```
    TETRAHEDRON (plural: tetrahedra)
    ════════════════════════════════

    The 3D equivalent of a triangle.
    - 4 vertices
    - 6 edges
    - 4 triangular faces
    - 1 volume

              v₃
              ╱│╲
             ╱ │ ╲
            ╱  │  ╲
           ╱   │   ╲
         v₀────│────v₁
           ╲   │   ╱
            ╲  │  ╱
             ╲ │ ╱
              ╲│╱
               v₂

    Vertices: v₀, v₁, v₂, v₃

    Edges (6):
    - v₀-v₁, v₀-v₂, v₀-v₃
    - v₁-v₂, v₁-v₃
    - v₂-v₃

    Faces (4 triangles):
    - (v₀, v₁, v₂) - base
    - (v₀, v₁, v₃)
    - (v₀, v₂, v₃)
    - (v₁, v₂, v₃)


    WHY TETRAHEDRA?
    ───────────────
    Just as triangles can tile any 2D surface,
    tetrahedra can fill any 3D volume.

    - Simplest 3D volume element
    - Always convex
    - Unique interpolation (barycentric)
    - Can represent any shape
```

### 3.2 Tetrahedron Properties

```
    TETRAHEDRON VOLUME
    ══════════════════

    Given vertices v₀, v₁, v₂, v₃:

    Edge vectors from v₀:
    a = v₁ - v₀
    b = v₂ - v₀
    c = v₃ - v₀

    Volume = (1/6) |a · (b × c)|

    The expression a · (b × c) is called the
    "scalar triple product" or "determinant"

              │ ax  ay  az │
    det(M) =  │ bx  by  bz │ = a · (b × c)
              │ cx  cy  cz │


    SIGNED VOLUME
    ─────────────
    Volume can be positive or negative depending
    on vertex ordering (like winding order for triangles)

    Positive: vertices in "right-hand" order
    Negative: vertices in "left-hand" order

    This is used to check/fix orientation consistency.


    CENTROID (center of mass)
    ─────────────────────────

    centroid = (v₀ + v₁ + v₂ + v₃) / 4

    Simple average of all vertices.
```

### 3.3 Tetrahedral Mesh

```
    TETRAHEDRAL MESH
    ════════════════

    Like a triangle mesh, but for volumes:

    1. VERTEX LIST
       V = [(x₀,y₀,z₀), (x₁,y₁,z₁), ...]

    2. TETRAHEDRA LIST (4 indices each)
       T = [(0,1,2,3), (1,4,2,3), ...]


    PROPERTIES
    ──────────

    1. SPACE-FILLING
       Tetrahedra completely fill the volume
       (no gaps, no overlaps)

    2. CONFORMAL
       Adjacent tetrahedra share entire faces
       (not partial faces)

       Good (conformal):       Bad (non-conformal):

        ╱╲    ╱╲                ╱╲    ╱╲
       ╱  ╲  ╱  ╲              ╱  ╲  ╱  ╲
      ╱────╲╱────╲            ╱────╲╱────╲
       Shared face             ╱  Gap!  ╲

    3. NEIGHBORING
       Each tet knows its 4 neighbors (one per face)
       Enables efficient traversal
```

---

## 4. Triangulation <a name="4-triangulation"></a>

### 4.1 What is Triangulation?

```
    TRIANGULATION
    ═════════════

    Given: A set of points
    Goal:  Connect them with triangles (2D) or tetrahedra (3D)

    Input:                    Output (one possibility):

      ●     ●                    ●─────●
                                 │╲   ╱│
        ●                        │ ╲ ╱ │
                          →      │  ●  │
      ●     ●                    │ ╱ ╲ │
                                 │╱   ╲│
        ●                        ●─────●


    MANY TRIANGULATIONS EXIST
    ─────────────────────────

    Same points, different triangulations:

      ●─────●           ●─────●
      │╲    │           │    ╱│
      │ ╲   │           │   ╱ │
      │  ╲  │    vs     │  ╱  │
      │   ╲ │           │ ╱   │
      │    ╲│           │╱    │
      ●─────●           ●─────●

    Which is "better"? → Delaunay triangulation!
```

### 4.2 Why Triangulate?

```
    APPLICATIONS OF TRIANGULATION
    ═════════════════════════════

    1. SURFACE RECONSTRUCTION
       Point cloud → Triangle mesh

       ●  ●  ●           ╱╲  ╱╲
        ●  ●      →     ╱  ╲╱  ╲
       ●  ●  ●         ╱────────╲

    2. FINITE ELEMENT ANALYSIS
       Physics simulation on irregular domains
       (stress, heat, fluid flow)

    3. INTERPOLATION
       Given values at scattered points,
       estimate value anywhere

    4. VOLUME FILLING
       Partition space into cells for
       - Rendering (radiance meshes!)
       - Collision detection
       - Path planning
```

---

## 5. Delaunay Triangulation <a name="5-delaunay-triangulation"></a>

### 5.1 The Delaunay Property

```
    DELAUNAY TRIANGULATION
    ══════════════════════

    Named after Boris Delaunay (1934)

    KEY PROPERTY: Empty Circumcircle (2D) / Circumsphere (3D)
    ─────────────────────────────────────────────────────────

    No point lies inside the circumcircle/circumsphere
    of any triangle/tetrahedron.


    2D EXAMPLE:

    Non-Delaunay:                 Delaunay:

         ●                            ●
        ╱│╲                          ╱ ╲
       ╱ │ ╲                        ╱   ╲
      ●──┼──●                      ●─────●
       ╲ │ ╱    ← point inside      ╲   ╱
        ╲│╱       circumcircle       ╲ ╱
         ●                            ●

         ╭───╮                      ╭───╮
        ╱  ●  ╲   Bad!             ╱     ╲  Good!
       │   │   │                  │       │
        ╲     ╱                    ╲     ╱
         ╰───╯                      ╰───╯


    3D: Same idea with circumsphere around tetrahedron
```

### 5.2 Why Delaunay is Special

```
    ADVANTAGES OF DELAUNAY
    ══════════════════════

    1. MAXIMIZES MINIMUM ANGLE
       ────────────────────────
       Avoids "skinny" triangles/tetrahedra

       Bad (skinny):          Good (Delaunay):

            ●                      ●
           ╱│                     ╱ ╲
          ╱ │                    ╱   ╲
         ╱  │                   ╱     ╲
        ●───●                  ●───────●

       Skinny elements cause:
       - Numerical instability
       - Poor interpolation
       - Rendering artifacts


    2. UNIQUE (for points in general position)
       ───────────────────────────────────────
       Given a point set, there's exactly one
       Delaunay triangulation (almost always).


    3. LOCAL OPTIMIZATION = GLOBAL OPTIMIZATION
       ─────────────────────────────────────────
       If every triangle/tet satisfies Delaunay
       locally, the whole mesh is optimal.


    4. EFFICIENT ALGORITHMS
       ─────────────────────
       O(n log n) in 2D
       O(n²) worst case in 3D, but fast in practice


    5. DUAL OF VORONOI
       ────────────────
       Delaunay and Voronoi are dual structures
       (more on this later!)
```

### 5.3 Computing Delaunay (Conceptual)

```
    DELAUNAY ALGORITHMS
    ═══════════════════

    1. INCREMENTAL INSERTION
       ──────────────────────
       Add points one at a time, maintain Delaunay property

       Start:        Add point:      Fix (flip edges):

       ●─────●       ●─────●         ●─────●
       │    ╱│       │╲   ╱│         │╲   ╱│
       │   ╱ │  →    │ ╲ ╱ │    →    │ ╲ ╱ │
       │  ╱  │       │  ●  │         │  ●  │
       │ ╱   │       │ ╱ ╲ │         │ ╱ ╲ │
       │╱    │       │╱   ╲│         │╱   ╲│
       ●─────●       ●─────●         ●─────●
                        ↑
                    New point


    2. DIVIDE AND CONQUER
       ───────────────────
       Split points, triangulate each half, merge

       ●  ●  │  ●  ●
        ●    │    ●
       ●  ●  │  ●  ●
          ↓     ↓
       Delaunay │ Delaunay
          ↓     ↓
           ╲   ╱
            ╲ ╱
           Merge


    3. EDGE FLIPPING
       ──────────────
       Start with any triangulation, flip non-Delaunay edges

       Before flip:        After flip:

       ●─────●            ●─────●
       │╲    │            │    ╱│
       │ ╲   │     →      │   ╱ │
       │  ╲  │            │  ╱  │
       │   ╲ │            │ ╱   │
       │    ╲│            │╱    │
       ●─────●            ●─────●

       Repeat until no illegal edges remain
```

### 5.4 GPU Delaunay (gDel3D)

```
    GPU-ACCELERATED DELAUNAY
    ════════════════════════

    Traditional algorithms are sequential.
    gDel3D parallelizes on GPU:

    1. Parallel point insertion
    2. Parallel conflict detection
    3. Parallel cavity re-triangulation

    Performance:
    - ~10ms for 100k points
    - Essential for real-time applications
    - Used by Radiance Meshes

    Challenges:
    - Race conditions
    - Numerical robustness
    - Memory management
```

---

## 6. Voronoi Diagrams <a name="6-voronoi-diagrams"></a>

### 6.1 What is a Voronoi Diagram?

```
    VORONOI DIAGRAM
    ═══════════════

    Named after Georgy Voronoi (1908)

    Given: A set of points (called "sites")
    Output: Partition of space into cells

    Each cell contains all points closer to its site
    than to any other site.


    2D EXAMPLE
    ──────────

    Sites:                    Voronoi diagram:

      ●₁    ●₂                 ┌────┬────┐
                               │ 1  │ 2  │
        ●₃                     ├──┬─┴─┬──┤
                        →      │  │ 3 │  │
      ●₄    ●₅                 ├──┴───┴──┤
                               │ 4  │ 5  │
                               └────┴────┘

    Cell i = {x : dist(x, site_i) < dist(x, site_j) for all j ≠ i}


    INTUITION: "Nearest Post Office"
    ────────────────────────────────
    Sites are post offices.
    Voronoi cell shows which houses are closest
    to each post office.
```

### 6.2 Properties of Voronoi Cells

```
    VORONOI CELL PROPERTIES
    ═══════════════════════

    1. CONVEX POLYGONS (2D) / POLYHEDRA (3D)
       ─────────────────────────────────────
       Each cell is always convex (no dents)

    2. EDGES ARE PERPENDICULAR BISECTORS
       ──────────────────────────────────
       Edge between cells i and j is equidistant
       from sites i and j

           site_i                site_j
              ●──────────│──────────●
                         │
                    Voronoi edge
              (perpendicular bisector)

    3. VERTICES ARE EQUIDISTANT
       ────────────────────────
       Voronoi vertex is equidistant from 3+ sites (2D)
       or 4+ sites (3D)

              ●
             ╱│╲
            ╱ │ ╲
           ╱  ●  ╲  ← Voronoi vertex
          ╱   │   ╲    equidistant from all 3 sites
         ●────┴────●


    4. SPACE-FILLING
       ─────────────
       Cells completely partition space
       (no gaps, no overlaps)
```

### 6.3 Computing Voronoi

```
    VORONOI COMPUTATION
    ═══════════════════

    Direct approach: O(n²) to O(n³)
    - For each point, intersect half-spaces

    Better approach: USE DELAUNAY DUALITY
    - Compute Delaunay triangulation: O(n log n)
    - Convert to Voronoi: O(n)


    CONVERSION FROM DELAUNAY
    ────────────────────────

    1. Delaunay vertex → Voronoi cell
    2. Delaunay edge → Voronoi face
    3. Delaunay triangle → Voronoi edge (2D)
    4. Delaunay tetrahedron → Voronoi vertex (3D)

    The circumcenter of a Delaunay element
    is the corresponding Voronoi element!
```

---

## 7. Delaunay-Voronoi Duality <a name="7-delaunay-voronoi-duality"></a>

### 7.1 The Duality Relationship

```
    DELAUNAY ↔ VORONOI DUALITY
    ══════════════════════════

    These are "dual" structures of each other.
    Same information, different perspective.


    DUALITY TABLE
    ─────────────

    ┌─────────────────────┬─────────────────────┐
    │      Delaunay       │       Voronoi       │
    ├─────────────────────┼─────────────────────┤
    │  Vertex (site)      │   Cell (region)     │
    │  Edge               │   Face              │
    │  Triangle (2D)      │   Vertex (2D)       │
    │  Tetrahedron (3D)   │   Vertex (3D)       │
    └─────────────────────┴─────────────────────┘


    VISUAL (2D):

    Delaunay:                 Voronoi:

         ●─────●              ┌─────┬─────┐
        ╱ ╲   ╱ ╲             │     │     │
       ╱   ╲ ╱   ╲            │  ●──┼──●  │
      ●─────●─────●           ├──┼──●──┼──┤
       ╲   ╱ ╲   ╱            │  ●──┼──●  │
        ╲ ╱   ╲ ╱             │     │     │
         ●─────●              └─────┴─────┘

    - Delaunay vertices ARE Voronoi sites
    - Delaunay triangle centers ARE Voronoi vertices
    - Delaunay edges CROSS Voronoi edges
```

### 7.2 The Circumcenter Connection

```
    CIRCUMCENTER = VORONOI VERTEX
    ═════════════════════════════

    This is the KEY insight!

    The circumcenter of a Delaunay triangle (2D)
    or tetrahedron (3D) IS the corresponding
    Voronoi vertex.


    WHY?
    ────
    Circumcenter is equidistant from all vertices.
    Voronoi vertex is equidistant from all nearby sites.
    Delaunay vertices ARE the sites.
    Therefore: circumcenter = Voronoi vertex!


    VISUAL PROOF (2D):

              Voronoi vertex
                   ●  ← circumcenter of triangle
                  ╱│╲
                 ╱ │ ╲
                ╱  │  ╲   All 3 distances equal!
               ╱   │   ╲
              ●────┴────●
          Delaunay vertices
          (= Voronoi sites)


    THIS IS WHY RADIANCE MESHES QUERIES AT CIRCUMCENTERS!
    ─────────────────────────────────────────────────────
    Querying at circumcenter = querying at Voronoi vertex
    = natural "center" of the region owned by that tetrahedron
```

### 7.3 Why Duality Matters for Rendering

```
    DUALITY IN NEURAL RENDERING
    ═══════════════════════════

    Both representations enable the same key properties:


    1. SPACE PARTITIONING
       ──────────────────
       Delaunay: Tetrahedra fill space
       Voronoi:  Cells fill space

       → Every point in space belongs to exactly one element


    2. EFFICIENT NEIGHBOR LOOKUP
       ─────────────────────────
       Delaunay: Tets share faces → O(1) neighbor access
       Voronoi:  Cells share faces → O(1) neighbor access

       → Rays can traverse space by stepping through neighbors


    3. CONTINUOUS UNDER POINT MOTION
       ──────────────────────────────
       When a site moves smoothly:
       - Voronoi cell boundaries move smoothly
       - Delaunay edges/faces move smoothly

       → Gradients can flow during optimization
       → Neural networks can learn by gradient descent


    RADIANCE MESHES vs RADIANT FOAM
    ───────────────────────────────

    Radiance Meshes: Delaunay view
    - Store features at tetrahedron circumcenters
    - Ray traces through tetrahedra

    Radiant Foam: Voronoi view
    - Store features at cell sites (= Delaunay vertices)
    - Ray traces through cells

    SAME MATH, DIFFERENT PERSPECTIVE!
```

---

## 8. Barycentric Coordinates <a name="8-barycentric-coordinates"></a>

### 8.1 What are Barycentric Coordinates?

```
    BARYCENTRIC COORDINATES
    ═══════════════════════

    A way to express a point's position relative to
    a simplex (triangle or tetrahedron) using weights.


    2D (Triangle):
    ──────────────

         v₂
         ╱╲
        ╱  ╲
       ╱ ●p ╲     p = λ₀v₀ + λ₁v₁ + λ₂v₂
      ╱      ╲
    v₀────────v₁   where λ₀ + λ₁ + λ₂ = 1

    The λᵢ are barycentric coordinates.


    3D (Tetrahedron):
    ─────────────────

    p = λ₀v₀ + λ₁v₁ + λ₂v₂ + λ₃v₃

    where λ₀ + λ₁ + λ₂ + λ₃ = 1


    PROPERTIES
    ──────────
    - Point inside simplex: all λᵢ ≥ 0
    - Point on face: one λᵢ = 0
    - Point on edge: two λᵢ = 0 (tet) or one λᵢ = 0 (tri)
    - Point at vertex: one λᵢ = 1, others = 0
```

### 8.2 Computing Barycentric Coordinates

```
    COMPUTING BARYCENTRIC COORDS (2D)
    ══════════════════════════════════

    Given triangle v₀, v₁, v₂ and point p:

    Method: Area ratios

         v₂
         ╱╲
        ╱A₀╲
       ╱────╲
      ╱ A₂ ●p╲ A₁
     ╱────────╲
   v₀──────────v₁

    λ₀ = Area(p, v₁, v₂) / Area(v₀, v₁, v₂)
    λ₁ = Area(v₀, p, v₂) / Area(v₀, v₁, v₂)
    λ₂ = Area(v₀, v₁, p) / Area(v₀, v₁, v₂)

    Area can be computed with cross product:
    Area(a,b,c) = 0.5 * |(b-a) × (c-a)|


    COMPUTING BARYCENTRIC COORDS (3D)
    ══════════════════════════════════

    Given tetrahedron v₀, v₁, v₂, v₃ and point p:

    Method: Volume ratios (same idea!)

    λ₀ = Volume(p, v₁, v₂, v₃) / Volume(v₀, v₁, v₂, v₃)
    λ₁ = Volume(v₀, p, v₂, v₃) / Volume(v₀, v₁, v₂, v₃)
    λ₂ = Volume(v₀, v₁, p, v₃) / Volume(v₀, v₁, v₂, v₃)
    λ₃ = Volume(v₀, v₁, v₂, p) / Volume(v₀, v₁, v₂, v₃)


    MATRIX METHOD (more efficient):
    ───────────────────────────────

    Solve: T · [λ₁, λ₂, λ₃]ᵀ = p - v₀

    Where T = [v₁-v₀, v₂-v₀, v₃-v₀]ᵀ

    Then: λ₀ = 1 - λ₁ - λ₂ - λ₃
```

### 8.3 Uses of Barycentric Coordinates

```
    APPLICATIONS OF BARYCENTRIC COORDS
    ═══════════════════════════════════

    1. INTERPOLATION
       ─────────────
       Given values at vertices, interpolate anywhere:

       value(p) = λ₀·value₀ + λ₁·value₁ + λ₂·value₂ + λ₃·value₃

       Used for:
       - Color interpolation
       - Normal interpolation
       - Texture coordinates
       - ANY per-vertex attribute


    2. INSIDE/OUTSIDE TEST
       ────────────────────
       Point p is inside simplex iff all λᵢ ≥ 0

       λᵢ < 0 → p is outside, on the opposite side of face i


    3. RAY-TRIANGLE INTERSECTION
       ──────────────────────────
       Möller-Trumbore algorithm computes both:
       - Intersection point (as t along ray)
       - Barycentric coordinates at intersection


    4. SMOOTH GRADIENTS IN RENDERING
       ──────────────────────────────
       Radiance Meshes: interpolate color across tetrahedron
       using barycentric weights for smooth appearance
```

---

## 9. Circumcenters and Circumspheres <a name="9-circumcenters"></a>

### 9.1 Definitions

```
    CIRCUMCIRCLE (2D) / CIRCUMSPHERE (3D)
    ═════════════════════════════════════

    The unique circle/sphere that passes through
    all vertices of a triangle/tetrahedron.


    2D: Circumcircle

              ╭─────────────╮
             ╱               ╲
            ╱    v₂           ╲
           │      ●            │
           │     ╱ ╲           │
           │    ╱   ╲          │
           │   ╱  ●  ╲         │   ← circumcircle
           │  ╱   cc  ╲        │      passes through
           │ ╱         ╲       │      v₀, v₁, v₂
           │●───────────●      │
            v₀           v₁
             ╲               ╱
              ╰─────────────╯

    cc = circumcenter (center of circle)
    r = circumradius (radius of circle)


    3D: Circumsphere

    Same concept: sphere through all 4 vertices
    of a tetrahedron
```

### 9.2 Computing the Circumcenter

```
    CIRCUMCENTER COMPUTATION
    ════════════════════════

    2D (Triangle):
    ──────────────

    Given vertices v₀, v₁, v₂:

    The circumcenter is equidistant from all three.
    It lies at the intersection of perpendicular bisectors.

    Formula (using determinants):

         │ |v₀|²  v₀.y  1 │       │ v₀.x  |v₀|²  1 │
    Cx = │ |v₁|²  v₁.y  1 │   Cy = │ v₁.x  |v₁|²  1 │
         │ |v₂|²  v₂.y  1 │       │ v₂.x  |v₂|²  1 │
         ─────────────────        ─────────────────
              2·D                      2·D

         │ v₀.x  v₀.y  1 │
    D =  │ v₁.x  v₁.y  1 │
         │ v₂.x  v₂.y  1 │


    3D (Tetrahedron):
    ─────────────────

    Given vertices v₀, v₁, v₂, v₃:

    Edge vectors from v₀:
    a = v₁ - v₀
    b = v₂ - v₀
    c = v₃ - v₀

    Circumcenter:

              |a|²(b × c) + |b|²(c × a) + |c|²(a × b)
    cc = v₀ + ───────────────────────────────────────
                        2 · a · (b × c)

    Circumradius:
    r = |cc - v₀|
```

### 9.3 Why Circumcenters Matter

```
    IMPORTANCE OF CIRCUMCENTERS
    ═══════════════════════════

    1. DELAUNAY PROPERTY
       ──────────────────
       A triangulation is Delaunay iff no vertex
       lies inside any circumsphere.

       → Circumsphere is the test for Delaunay-ness


    2. VORONOI DUAL
       ────────────
       Circumcenter of Delaunay element =
       Voronoi vertex

       → Converting between representations


    3. NATURAL QUERY POINT
       ────────────────────
       For per-tetrahedron features, the circumcenter
       is a natural "center" to query:

       - Equidistant from all vertices
       - Stable (doesn't jump with small vertex changes)
       - Geometrically meaningful

       → Radiance Meshes queries hash grid at circumcenters


    4. QUALITY METRIC
       ───────────────
       Circumradius / shortest edge = quality measure
       Large ratio → bad (skinny) element
       Small ratio → good (well-shaped) element
```

---

## 10. Ray Tracing Fundamentals <a name="10-ray-tracing"></a>

### 10.1 What is Ray Tracing?

```
    RAY TRACING
    ═══════════

    Simulate light by tracing rays through a scene.


    BASIC IDEA
    ──────────

    For each pixel:
    1. Shoot a ray from camera through pixel
    2. Find what the ray hits
    3. Compute color based on hit point

         Camera              Scene
            ●─────────────────────────────────
           ╱│                    ╱╲
          ╱ │                   ╱  ╲
         ╱  │  Ray            ╱    ╲
        ╱   │ ─────────●──►  ╱ Hit! ╲
       ╱    │               ╱________╲
      Image │
      Plane │


    RAY DEFINITION
    ──────────────

    r(t) = o + t·d

    Where:
    - o = ray origin (camera position)
    - d = ray direction (normalized)
    - t = distance along ray (t ≥ 0)

    Finding "what ray hits" = finding smallest t
    where r(t) intersects geometry
```

### 10.2 Ray-Triangle Intersection

```
    RAY-TRIANGLE INTERSECTION
    ═════════════════════════

    Most fundamental operation in ray tracing.


    MÖLLER-TRUMBORE ALGORITHM
    ─────────────────────────

    Given:
    - Ray: r(t) = o + t·d
    - Triangle: v₀, v₁, v₂

    Find: t, u, v such that
          o + t·d = (1-u-v)·v₀ + u·v₁ + v·v₂

    This gives both:
    - t: distance along ray
    - (u, v): barycentric coordinates (with w = 1-u-v)


    ALGORITHM:

    e₁ = v₁ - v₀
    e₂ = v₂ - v₀
    h = d × e₂
    a = e₁ · h

    if |a| < ε: return NO_HIT  (ray parallel to triangle)

    f = 1/a
    s = o - v₀
    u = f · (s · h)

    if u < 0 or u > 1: return NO_HIT

    q = s × e₁
    v = f · (d · q)

    if v < 0 or u + v > 1: return NO_HIT

    t = f · (e₂ · q)

    if t > ε: return HIT at t with barycentric (1-u-v, u, v)
    else: return NO_HIT
```

### 10.3 Ray-Tetrahedron Intersection

```
    RAY-TETRAHEDRON INTERSECTION
    ════════════════════════════

    Find where ray enters and exits a tetrahedron.


    METHOD: Test ray against all 4 faces
    ────────────────────────────────────

    A tetrahedron has 4 triangular faces.
    Test ray against each face.

              v₃
              ╱│╲
             ╱ │ ╲
            ╱  │  ╲
     Ray ─────●│────●──────►
          ╱    │      ╲
        v₀─────│───────v₁
          ╲    │      ╱
           ╲   │     ╱
            ╲  │    ╱
             ╲ │   ╱
              ╲│  ╱
               v₂
            entry    exit


    CASES
    ─────
    - 0 hits: ray misses tetrahedron
    - 2 hits: ray passes through (entry + exit)
    - 1 hit: ray starts inside (only exit)
             or ends inside (only entry)


    RESULT
    ──────
    - t_entry: where ray enters
    - t_exit: where ray exits
    - length = t_exit - t_entry (path length through tet)

    Used for volume rendering!
```

### 10.4 Acceleration Structures

```
    WHY ACCELERATION?
    ═════════════════

    Naive ray tracing: O(n) per ray (test every primitive)
    With 1M triangles and 1M pixels = 10¹² intersection tests!


    ACCELERATION STRUCTURES
    ───────────────────────

    1. BOUNDING VOLUME HIERARCHY (BVH)
       ────────────────────────────────
       Tree of bounding boxes

                    ┌─────────────────┐
                    │   Root bbox     │
                    └────────┬────────┘
                       ╱           ╲
              ┌───────┴───┐   ┌────┴──────┐
              │  Left     │   │   Right   │
              └─────┬─────┘   └─────┬─────┘
                 ╱     ╲         ╱     ╲
               ...     ...     ...     ...

       Test: If ray misses bbox, skip entire subtree
       Reduces O(n) → O(log n) average


    2. SPATIAL GRIDS
       ──────────────
       Divide space into uniform cells
       Only test primitives in cells ray passes through


    3. NEIGHBOR TRAVERSAL (used by Radiance Meshes!)
       ──────────────────────────────────────────────
       For Delaunay/Voronoi: elements share faces
       → Step from element to element through shared faces
       → No tree structure needed
       → O(k) where k = elements ray passes through
```

---

## 11. Volume Rendering <a name="11-volume-rendering"></a>

### 11.1 What is Volume Rendering?

```
    VOLUME RENDERING
    ════════════════

    Render semi-transparent volumetric data
    (fog, clouds, smoke, medical scans, neural radiance fields)


    DIFFERENCE FROM SURFACE RENDERING
    ─────────────────────────────────

    Surface rendering:          Volume rendering:
    - Hit surface, done         - Accumulate along ray
    - Binary (hit/miss)         - Continuous (density)
    - Single color at hit       - Integrated color

        Surface:                    Volume:

           ╱╲                      ░░▒▒▓▓██▓▓▒▒░░
          ╱  ╲                    ░░▒▒▓▓████▓▓▒▒░░
    Ray ─●────────►          Ray ─●──────────────►
          Hit!                    Accumulate through


    KEY QUANTITIES
    ──────────────

    At each point in the volume:
    - σ(x): density (how much light is absorbed/scattered)
    - c(x): color (what color is emitted/scattered)
```

### 11.2 The Volume Rendering Equation

```
    VOLUME RENDERING INTEGRAL
    ═════════════════════════

    For a ray r(t) = o + t·d from t=0 to t=∞:


                    ∞
    C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt
                    0

    Where T(t) is the transmittance:

                      t
    T(t) = exp( -∫ σ(r(s)) ds )
                      0


    INTUITION
    ─────────

    T(t) = probability that ray reaches distance t
           without being absorbed

    σ(t) = density at distance t (absorption rate)

    c(t) = color emitted at distance t

    The integral sums up:
    "color emitted at each point, weighted by
     probability of reaching that point"


    VISUAL
    ──────

    Ray ──────────────────────────────────────►
              │         │         │
              t₁        t₂        t₃
              │         │         │
              ▼         ▼         ▼
         T(t₁)·σ·c  T(t₂)·σ·c  T(t₃)·σ·c
              │         │         │
              └─────────┴─────────┘
                        │
                        ▼
                   Final color C
```

### 11.3 Discretized Volume Rendering

```
    DISCRETIZED (PRACTICAL) VERSION
    ════════════════════════════════

    Sample the ray at discrete points (or primitives):


    C = Σᵢ Tᵢ · αᵢ · cᵢ

    Where:
    - Tᵢ = Πⱼ<ᵢ (1 - αⱼ)    (accumulated transmittance)
    - αᵢ = 1 - exp(-σᵢ · δᵢ) (opacity of segment i)
    - δᵢ = length of segment i
    - cᵢ = color of segment i


    FRONT-TO-BACK COMPOSITING
    ─────────────────────────

    Initialize: C = 0, T = 1

    For each segment i (front to back):
        α = 1 - exp(-σᵢ · δᵢ)
        C = C + T · α · cᵢ
        T = T · (1 - α)

        if T < threshold: break  (early termination)


    RADIANCE MESHES VERSION
    ───────────────────────

    Each tetrahedron is one "segment":
    - σᵢ = density of tetrahedron i
    - δᵢ = ray path length through tetrahedron i
    - cᵢ = color at ray midpoint in tetrahedron i
           (using color gradient field)
```

### 11.4 Alpha Blending

```
    ALPHA BLENDING
    ══════════════

    The "over" operator for compositing:

    C_out = α·C_front + (1-α)·C_back


    FRONT-TO-BACK ORDER
    ───────────────────

    C = 0, T = 1

    For each layer (front to back):
        C += T · α · c
        T *= (1 - α)

         Layer 0    Layer 1    Layer 2
         α₀, c₀     α₁, c₁     α₂, c₂
            │          │          │
            ▼          ▼          ▼
         T₀·α₀·c₀  T₁·α₁·c₁  T₂·α₂·c₂
            │          │          │
            └──────────┴──────────┘
                       │
                       ▼
                  Final color


    BACK-TO-FRONT ORDER (simpler but less efficient)
    ────────────────────────────────────────────────

    For each layer (back to front):
        C = α·c + (1-α)·C

    No early termination possible (must process all layers)
```

---

## 12. Spatial Data Structures <a name="12-spatial-data-structures"></a>

### 12.1 Hash Grids

```
    SPATIAL HASHING
    ═══════════════

    Map 3D positions to array indices using a hash function.


    BASIC IDEA
    ──────────

    1. Divide space into grid cells
    2. Hash cell coordinates to array index
    3. Store/lookup data by position

       3D Position         Grid Cell         Hash Table
       (x, y, z)    →    (i, j, k)     →    index h
                                              │
                                              ▼
                                          ┌───────┐
                                          │ Data  │
                                          └───────┘


    HASH FUNCTION
    ─────────────

    h(i, j, k) = (i·p₁ ⊕ j·p₂ ⊕ k·p₃) mod T

    Where:
    - p₁, p₂, p₃ are large prime numbers
    - ⊕ is XOR
    - T is table size
    - mod ensures index is in valid range


    WHY HASHING?
    ────────────
    - O(1) lookup (constant time)
    - No tree traversal
    - GPU-friendly (parallel access)
    - Memory efficient (only store occupied cells)
```

### 12.2 Multi-Resolution Hash Encoding (Instant-NGP)

```
    INSTANT-NGP HASH ENCODING
    ═════════════════════════

    Key innovation from Müller et al., 2022
    Used by Radiance Meshes for feature encoding.


    MULTI-RESOLUTION GRIDS
    ──────────────────────

    Multiple hash grids at different resolutions:

    Level 0 (coarse):     Level 1:          Level L (fine):
    ┌───┬───┐            ┌─┬─┬─┬─┐         ┌┬┬┬┬┬┬┬┬┐
    │   │   │            ├─┼─┼─┼─┤         ├┼┼┼┼┼┼┼┼┤
    ├───┼───┤            ├─┼─┼─┼─┤         ├┼┼┼┼┼┼┼┼┤
    │   │   │            ├─┼─┼─┼─┤         ...
    └───┴───┘            └─┴─┴─┴─┘

    Resolution at level l: Nₗ = N₀ · bˡ
    (geometric progression)


    ENCODING PROCESS
    ────────────────

    For query point x:

    1. For each level l:
       a. Scale x to grid resolution Nₗ
       b. Find 8 surrounding voxel corners
       c. Hash each corner to get table indices
       d. Look up learned features at each corner
       e. Trilinear interpolation

    2. Concatenate features from all levels:
       output = [F₀, F₁, F₂, ..., Fₗ]


    WHY IT WORKS
    ────────────
    - Coarse levels: capture large-scale structure
    - Fine levels: capture small details
    - Hash collisions: handled by learning
      (network learns to disambiguate)
    - Very compact: millions of parameters in small tables
```

### 12.3 Octrees and KD-Trees (for comparison)

```
    OTHER SPATIAL STRUCTURES
    ════════════════════════

    OCTREE
    ──────
    Recursive subdivision into 8 children

              ┌───────────────┐
              │               │
              │   ┌───┬───┐   │
              │   │   │   │   │
              │   ├───┼───┤   │
              │   │   │   │   │
              │   └───┴───┘   │
              │               │
              └───────────────┘

    Pros: Adaptive resolution, exact spatial queries
    Cons: Tree traversal overhead, complex GPU implementation


    KD-TREE
    ───────
    Binary space partitioning (alternating axes)

         ┌─────────────────┐
         │        │        │
         │   A    │   B    │
         │        │────────│
         │        │ C │ D  │
         └────────┴───┴────┘

    Pros: Good for nearest-neighbor queries
    Cons: Not ideal for GPU, build time


    WHY HASH GRIDS WIN FOR NEURAL RENDERING
    ───────────────────────────────────────
    - O(1) vs O(log n) lookup
    - Trivially parallel on GPU
    - Learnable (features in table, not structure)
    - No pointer chasing
```

---

## Quick Reference

```
    CONCEPT CHEAT SHEET
    ═══════════════════

    SIMPLEX (n-dimensional "triangle")
    ──────────────────────────────────
    0D: point
    1D: line segment (2 vertices)
    2D: triangle (3 vertices)
    3D: tetrahedron (4 vertices)

    TRIANGULATION
    ─────────────
    Connect points with simplices (no gaps/overlaps)

    DELAUNAY TRIANGULATION
    ──────────────────────
    Triangulation where no point is inside any circumsphere
    → Maximizes minimum angle
    → Unique for points in general position

    VORONOI DIAGRAM
    ───────────────
    Partition space by "nearest site"
    Cell i = all points closest to site i

    DUALITY
    ───────
    Delaunay vertex ↔ Voronoi cell
    Delaunay edge ↔ Voronoi face
    Delaunay tet ↔ Voronoi vertex
    Circumcenter of Delaunay tet = Voronoi vertex

    BARYCENTRIC COORDINATES
    ───────────────────────
    p = λ₀v₀ + λ₁v₁ + ... (weights sum to 1)
    Inside simplex iff all λᵢ ≥ 0

    CIRCUMCENTER
    ────────────
    Center of circumscribed circle/sphere
    Equidistant from all vertices

    VOLUME RENDERING
    ────────────────
    C = Σᵢ Tᵢ · αᵢ · cᵢ
    αᵢ = 1 - exp(-σᵢ · δᵢ)
    Tᵢ = Πⱼ<ᵢ (1 - αⱼ)
```

---

## Further Reading

1. **Computational Geometry**: de Berg et al., "Computational Geometry: Algorithms and Applications"
2. **Delaunay/Voronoi**: Aurenhammer, "Voronoi Diagrams: A Survey of a Fundamental Geometric Data Structure"
3. **Ray Tracing**: Pharr et al., "Physically Based Rendering: From Theory to Implementation"
4. **Volume Rendering**: Levoy, "Display of Surfaces from Volume Data"
5. **Instant-NGP**: Müller et al., "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding", SIGGRAPH 2022
6. **NeRF**: Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields", ECCV 2020
7. **3D Gaussian Splatting**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
