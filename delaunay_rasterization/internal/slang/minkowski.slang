#define MAX_POLY_POINTS 16

import utils;

struct Polygon2D {
    float2 points[MAX_POLY_POINTS];
    int count;
};

float cross2D(float2 a, float2 b)
{
    return a.x * b.y - a.y * b.x;
}

// Compare two float2 by x, then y. Return < 0, = 0, or > 0
// as usual "compare" semantics. HLSL doesn't have a built-in comparator, so:
int compareFloat2XY(float2 a, float2 b)
{
    if (a.x < b.x) return -1;
    if (a.x > b.x) return  1;
    // x's are equal, compare y
    if (a.y < b.y) return -1;
    if (a.y > b.y) return  1;
    return 0;
}

// Simple insertion sort by (x,y) since we only have up to ~12 points.
void sortPointsInPlaceXY(inout float2 pts[MAX_POLY_POINTS], int n)
{
    for (int i = 1; i < n; i++)
    {
        float2 val = pts[i];
        int j = i - 1;
        while (j >= 0 && compareFloat2XY(pts[j], val) > 0)
        {
            pts[j+1] = pts[j];
            j--;
        }
        pts[j+1] = val;
    }
}

// Monotone chain convex hull for up to 'inputCount' points in 'inputPts'.
// Fills out 'outHull' and returns hullCount.
int buildConvexHull(
    in float2  inputPts[MAX_POLY_POINTS], 
    int        inputCount,
    out float2 outHull[MAX_POLY_POINTS])
{
    // 1) Sort points by x, then y
    sortPointsInPlaceXY(inputPts, inputCount);

    // We'll build the hull in outHull, first the lower hull, then the upper hull.
    int hullCount = 0;

    // 2) Build lower hull
    for (int i = 0; i < inputCount; i++)
    {
        while (hullCount >= 2)
        {
            float2 A = outHull[hullCount - 2];
            float2 B = outHull[hullCount - 1];
            float2 C = inputPts[i];
            // Check the cross product (B-A) x (C-B); 
            // if <= 0, we pop the last hull point to keep hull convex
            if (cross2D(B - A, C - B) <= 0)
            {
                hullCount--;
            }
            else
                break;
        }
        outHull[hullCount++] = inputPts[i];
    }

    int lowerCount = hullCount;

    // 3) Build upper hull
    //    (We skip the last point because it's the same as the first of upper hull.)
    for (int i = inputCount - 2; i >= 0; i--)
    {
        while (hullCount > lowerCount)
        {
            float2 A = outHull[hullCount - 2];
            float2 B = outHull[hullCount - 1];
            float2 C = inputPts[i];
            if (cross2D(B - A, C - B) <= 0)
            {
                hullCount--;
            }
            else
                break;
        }
        outHull[hullCount++] = inputPts[i];
    }

    // 4) The last point is a duplicate of the first
    hullCount--;

    return hullCount;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 3) Minkowski sum of a Triangle with an Axis-Aligned Square of 'amount'
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Polygon2D grow_triangle_linf(Triangle2D tri, float amount)
{
    // We'll collect up to 12 candidate points: 3 triangle vertices x 4 corners.
    float2 corners[4] = {
        float2(-amount, -amount),
        float2(-amount,  amount),
        float2( amount, -amount),
        float2( amount,  amount)
    };
    
    float2 tverts[3] = { tri.a, tri.b, tri.c };

    float2 sums[MAX_POLY_POINTS];
    int sumCount = 0;
    
    // Generate all combinations of tverts + corners
    [ForceUnroll]
    for (int i = 0; i < 3; i++)
    {
        [ForceUnroll]
        for (int j = 0; j < 4; j++)
        {
            sums[sumCount++] = tverts[i] + corners[j];
        }
    }

    // Now compute the convex hull of these points
    float2 hullPoints[MAX_POLY_POINTS];
    int hullCount = buildConvexHull(sums, sumCount, hullPoints);

    // Package into Polygon2D
    Polygon2D result;
    result.count = hullCount;
    
    for (int i = 0; i < MAX_POLY_POINTS; i++)
    {
        if (i < hullCount)
            result.points[i] = hullPoints[i];
    }

    return result;
}

bool in_polygon(float2 p, Polygon2D poly)
{
    int n = poly.count;
    if (n < 3)
        return false;  // Not a valid polygon

    for (int i = 0; i < n; i++)
    {
        // Current vertex
        float2 vA = poly.points[i];
        // Next vertex (wrap around with modulo)
        float2 vB = poly.points[(i + 1) % n];

        // Edge direction
        float2 edge  = vB - vA;
        // Vector from A to point p
        float2 toP   = p - vA;

        // Cross product
        float c = cross2D(edge, toP);

        // If c < 0, the point is to the right of edge (when traveling CCW),
        // which means it's outside the polygon.
        if (c < 0.0f)
        {
            return false;
        }
    }

    // If we never found a negative cross, the point is inside or on an edge
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Helper: 2D line intersection
//    Solves the system:
//      A1*x + B1*y = C1
//      A2*x + B2*y = C2
//    Returns (x,y).
//    Assumes lines are not parallel (the user triangle must be non-degenerate).
////////////////////////////////////////////////////////////////////////////////
float2 IntersectLines(float A1, float B1, float C1, 
                      float A2, float B2, float C2)
{
    float det = A1*B2 - B1*A2;
    // If det == 0, lines are parallel or degenerate; 
    // for a well-formed triangle it shouldn't happen unless degenerate input.

    float x = (C1*B2 - B1*C2) / det;
    float y = (A1*C2 - C1*A2) / det;
    return float2(x, y);
}

////////////////////////////////////////////////////////////////////////////////
// ExpandTriangleBySquare_Linf
//   Returns the *minimal bounding triangle* that encloses
//       T ⊕ [-amount, amount]^2
//   by offsetting each edge outward in the L∞ sense.
//
//   Assumes the input triangle is in CCW order (v0->v1->v2).
//   If it's not, either reorder it or flip the direction of the normals.
//
//   Steps:
//   (1) For each edge e_i, compute outward normal n_i = rotate -90° (because T is CCW).
//   (2) line_i: n_i dot (x, y) = c_i
//         c_i = n_i dot v_i
//   (3) offset c_i by a*(|n_i.x| + |n_i.y|)
//   (4) Intersect the 3 offset lines pairwise => 3 corners => bounding triangle
////////////////////////////////////////////////////////////////////////////////
Triangle2D grow_triangle(Triangle2D tri, float amount)
{
    // Extract the 3 vertices (CCW order).
    float2 v0 = tri.a;
    float2 v1 = tri.b;
    float2 v2 = tri.c;

    // Edges: e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2
    float2 e0 = v1 - v0;
    float2 e1 = v2 - v1;
    float2 e2 = v0 - v2;

    // Outward normals for a CCW triangle:
    //   "right side" normal is rotate by -90: n = (ey, -ex).
    // e0 => n0
    float2 n0 = float2(e0.y, -e0.x); // outward normal to edge (v0->v1)
    float2 n1 = float2(e1.y, -e1.x); // outward normal to edge (v1->v2)
    float2 n2 = float2(e2.y, -e2.x); // outward normal to edge (v2->v0)

    // For each line: A*x + B*y = C
    // We'll store them in arrays for convenience
    float2 normals[3] = { n0, n1, n2 };
    float2 points[3]  = { v0, v1, v2 };

    float A[3];
    float B[3];
    float C[3];

    // Build the offset lines
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        float2 n = normals[i];
        float2 p = points[i];  // the "anchor" vertex for that edge

        // A, B = n.x, n.y
        A[i] = n.x;
        B[i] = n.y;

        // Base line constant: n dot p
        float baseC = n.x*p.x + n.y*p.y;

        // Offset in L∞: a*(|n.x| + |n.y|)
        float offset = amount * (abs(n.x) + abs(n.y));

        // Final line constant
        C[i] = baseC + offset;
    }

    // Now intersect lines pairwise: (0,1), (1,2), (2,0)
    float2 P0 = IntersectLines(A[0], B[0], C[0], A[1], B[1], C[1]); 
    float2 P1 = IntersectLines(A[1], B[1], C[1], A[2], B[2], C[2]); 
    float2 P2 = IntersectLines(A[2], B[2], C[2], A[0], B[0], C[0]); 

    // Package into a Triangle2D
    Triangle2D outTri;
    outTri.a = P0;
    outTri.b = P1;
    outTri.c = P2;

    // OPTIONAL: Make sure it's in CCW order
    // We can do a quick cross test and swap if needed:
    {
        float2 eA = outTri.b - outTri.a;
        float2 eB = outTri.c - outTri.a;
        float crossZ = eA.x*eB.y - eA.y*eB.x;  // cross2D(eA, eB)
        if (crossZ < 0.0f)
        {
            // swap v1 <-> v2
            float2 tmp = outTri.b;
            outTri.b = outTri.c;
            outTri.c = tmp;
        }
    }

    return outTri;
}