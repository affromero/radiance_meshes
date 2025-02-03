import utils;
import intersect;

// float2 clip(float2 x, float2 

float compute_screen_area(float2 v0, float2 v1, float2 v2, float height, float width)
{
    float2 bounds = {width, height};
    float2 lower_bound = {0.f, 0.f};
    // Clip each vertex to screen bounds
    float2 c0 = clamp(v0, lower_bound, bounds);
    float2 c1 = clamp(v1, lower_bound, bounds);
    float2 c2 = clamp(v2, lower_bound, bounds);

    // Compute the area using the determinant method
    float area = abs(0.5 * ((c1.x - c0.x) * (c2.y - c0.y) - (c2.x - c0.x) * (c1.y - c0.y)));
    // printf("v0: %f, %f, v1: %f, %f, v2: %f, %f, area: %f\n", v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, area);
    return area;
}

// Main function to iterate over tetrahedron faces
float tet_area(Tetrahedra image_tet, float3 view_dir, Camera cam)
{
    float area = 0;
    float3 centroid = (image_tet.verts[0] + image_tet.verts[1] + image_tet.verts[2] + image_tet.verts[3]) * 0.25f;
    int faceIndices[4][3] = {
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3}
    };

    [unroll]
    for (int i = 0; i < 4; i++)
    {
        // Get triangle vertices
        float3 v0 = image_tet.verts[faceIndices[i][0]];
        float3 v1 = image_tet.verts[faceIndices[i][1]];
        float3 v2 = image_tet.verts[faceIndices[i][2]];

        // Compute face normal
        float3 normal = normalize(cross(v1 - v0, v2 - v0));
        float d = dot(normal, (centroid - v0));

        // Check if back-facing
        if (dot(normal, view_dir) * d < 0) {
            continue;
        }

        // Compute screen-space 2D coordinates
        float2 s0 = v0.xy;
        float2 s1 = v1.xy;
        float2 s2 = v2.xy;

        // Compute and store area
        area += compute_screen_area(s0, s1, s2, cam.H, cam.W);
    }
    return area;
}
