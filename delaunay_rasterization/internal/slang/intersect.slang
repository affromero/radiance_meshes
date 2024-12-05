#define epsilon 1e-6
#define FLT_MAX 1e20
#define FLT_MIN -1e20

// Helper function to compute the determinant of a 3x3 matrix
[Differentiable]
float det_3x3(float3 a, float3 b, float3 c) {
    return dot(a, cross(b, c));
}

[Differentiable]
bool ray_triangle_intersect(in float3 ray_origin, 
                             in float3 ray_vector, 
                             in float3 tri_a,
                             in float3 tri_b,
                             in float3 tri_c,
                             out float t)
{
    t = -1;
    float3 edge1 = tri_b - tri_a;
    float3 edge2 = tri_c - tri_a;
    float3 ray_cross_e2 = cross(ray_vector, edge2);
    float det = dot(edge1, ray_cross_e2);

    if (det > -epsilon && det < epsilon)
        return false;    // This ray is parallel to this triangle.

    float inv_det = 1.0 / det;
    float3 s = ray_origin - tri_a;
    float u = inv_det * dot(s, ray_cross_e2);

    if (u < 0 || u > 1)
        return false;

    float3 s_cross_e1 = cross(s, edge1);
    float v = inv_det * dot(ray_vector, s_cross_e1);

    if (v < 0 || u + v > 1)
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    t = inv_det * dot(edge2, s_cross_e1);

    return true;
}

// Ray-tetrahedron intersection function
[Differentiable]
float ray_tetrahedron_intersect(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2, float3 v3) {
    bool hit = false;
    float t_enter = FLT_MAX;
    float t_exit = -FLT_MAX;

    // Check intersection with each face of the tetrahedron
    float t;
    if (ray_triangle_intersect(orig, dir, v0, v1, v2, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(orig, dir, v0, v1, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(orig, dir, v0, v2, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    if (ray_triangle_intersect(orig, dir, v1, v2, v3, t)) {
        hit = true;
        t_enter = min(t_enter, t);
        t_exit = max(t_exit, t);
    }
    /*
    if (hit) {
        printf("t: [%f, %f], r_o: (%f, %f, %f), r_d: (%f, %f, %f)\n", t_enter, t_exit, 
            orig.x, orig.y, orig.z, 
            dir.x, dir.y, dir.z);
    }
        //*/

    if (hit) {
        return max((t_exit - max(t_enter, 0)) * length(dir), 0);
    } else {
        return -1;
    }
}
