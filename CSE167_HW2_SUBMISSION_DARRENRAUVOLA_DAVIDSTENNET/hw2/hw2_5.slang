[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

bool is_within_triangle(int2 pixel, float2 sp0, float2 sp1, float2 sp2) {
    // Compute Edges
    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    // Compute Normals
    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    // Variables for computing edge dot prodcut for loop
    float2 p = float2(pixel.x + 0.5, pixel.y + 0.5);
    float2 points[] = { sp0, sp1, sp2 };
    float2 normals[] = { n01, n12, n20 };
    float edges[3];
    for (int i = 0; i < 3; i++) {
        edges[i] = dot(p - points[i], normals[i]);
    }

    // If all non-negative or non-positive
    if ((edges[0] >= 0 && edges[1] >= 0 && edges[2] >= 0) ||
        (edges[0] <= 0 && edges[1] <= 0 && edges[2] <= 0)) {
        return true;
    }

    return false;
}

// Helper function for computing barycentric coordinates
float compute_area(float2 p1, float2 p2, float2 p3) {
    float3 e12 = float3(p2 - p1, 0.0);
    float3 e13 = float3(p3 - p1, 0.0);

    float area = 0.5 * length(cross(e12, e13));

    return area;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {

    // Makes sure we just run as many threads as we have triangle
    if (tid >= num_triangles)
        return;

    // Get face in parallel
    uint3 face = faces[tid];
    float2 sp0 = float2(projected_vertices[face.x].x, projected_vertices[face.x].y);
    float2 sp1 = float2(projected_vertices[face.y].x, projected_vertices[face.y].y);
    float2 sp2 = float2(projected_vertices[face.z].x, projected_vertices[face.z].y);
    
    // Calculate the triangle's screen-space bounding box.
    float2 tri_min = float2(fmin3(sp0.x, sp1.x, sp2.x), fmin3(sp0.y, sp1.y, sp2.y));
    float2 tri_max = float2(fmax3(sp0.x, sp1.x, sp2.x), fmax3(sp0.y, sp1.y, sp2.y));

    // Clamp in floating-point space first.
    tri_min = max(tri_min, float2(0.0, 0.0));
    tri_max = min(tri_max, float2(float(width - 1), float(height - 1)));

    // Convert to integer pixel bounds.
    int2 aabb_min = int2(floor(tri_min));
    int2 aabb_max = int2(ceil(tri_max));

    // Skip triangles entirely outside the viewport.
    if (any(tri_min > tri_max) || aabb_min.x > aabb_max.x || aabb_min.y > aabb_max.y) {
        return;
    }

    // for x, y in screen
    for (int x = aabb_min.x; x <= aabb_max.x; x++) {
        for (int y = aabb_min.y; y <= aabb_max.y; y++) {
            
            int2 pixel = int2(x, y);

            if (is_within_triangle(pixel, sp0, sp1, sp2)) {
                float inv_depth0 = projected_vertices[face.x].z;
                float inv_depth1 = projected_vertices[face.y].z;
                float inv_depth2 = projected_vertices[face.z].z;
                float2 p_screen = float2(pixel.x + 0.5, pixel.y + 0.5);

                // a) Getting camera space coordinates and then original coordinates
                float2 cp0 = float2(((2 * s * a * sp0.x) / (width)) - s * a, s - ((2 * s * sp0.y) / (height)));
                float2 cp1 = float2(((2 * s * a * sp1.x) / (width)) - s * a, s - ((2 * s * sp1.y) / (height)));
                float2 cp2 = float2(((2 * s * a * sp2.x) / (width)) - s * a, s - ((2 * s * sp2.y) / (height)));
                float2 cp = float2(((2 * s * a * p_screen.x) / (width)) - s * a, s - ((2 * s * p_screen.y) / (height)));

                // b) Getting the barycentric coordinate coefficeints in camera space
                float cp0cp1cp2_area = compute_area(cp0, cp1, cp2);
                if (cp0cp1cp2_area == 0) {
                    continue;
                }
                float cb0 = compute_area(cp, cp1, cp2) / cp0cp1cp2_area;
                float cb1 = compute_area(cp0, cp, cp2) / cp0cp1cp2_area;
                float cb2 = compute_area(cp0, cp1, cp) / cp0cp1cp2_area;

                // c) Getting the original barycentric coordinate coefficients
                float denom = cb0 / (-inv_depth0) + cb1 / (-inv_depth1) + cb2 / (-inv_depth2);
                float b0 = (cb0 / -inv_depth0) / denom;
                float b1 = (cb1 / -inv_depth1) / denom;
                float b2 = (cb2 / -inv_depth2) / denom;

                // d) and e) Get the depth and compare with the depth image of the pixel and update
                float p_depth = b0 * inv_depth0 + b1 * inv_depth1 + b2 * inv_depth2;

                uint depth_bits = asuint(p_depth);
                uint index = y * width + x;
                InterlockedMin(depth_buffer[index], depth_bits);
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) {

    // Makes sure we just run as many threads as we have triangle
    if (tid >= num_triangles)
        return;

    // Get face in parallel
    uint3 face = faces[tid];
    float2 sp0 = float2(projected_vertices[face.x].x, projected_vertices[face.x].y);
    float2 sp1 = float2(projected_vertices[face.y].x, projected_vertices[face.y].y);
    float2 sp2 = float2(projected_vertices[face.z].x, projected_vertices[face.z].y);

    // Calculate the triangle's screen-space bounding box.
    float2 tri_min = float2(fmin3(sp0.x, sp1.x, sp2.x), fmin3(sp0.y, sp1.y, sp2.y));
    float2 tri_max = float2(fmax3(sp0.x, sp1.x, sp2.x), fmax3(sp0.y, sp1.y, sp2.y));

    // Clamp in floating-point space first.
    tri_min = max(tri_min, float2(0.0, 0.0));
    tri_max = min(tri_max, float2(float(width - 1), float(height - 1)));

    // Convert to integer pixel bounds.
    int2 aabb_min = int2(floor(tri_min));
    int2 aabb_max = int2(ceil(tri_max));

    // Skip triangles entirely outside the viewport.
    if (any(tri_min > tri_max) || aabb_min.x > aabb_max.x || aabb_min.y > aabb_max.y) {
        return;
    }

    // for x, y in screen
    for (int x = aabb_min.x; x <= aabb_max.x; x++) {
        for (int y = aabb_min.y; y <= aabb_max.y; y++) {
            int2 pixel = int2(x, y);

            if (is_within_triangle(pixel, sp0, sp1, sp2)) {
                float inv_depth0 = projected_vertices[face.x].z;
                float inv_depth1 = projected_vertices[face.y].z;
                float inv_depth2 = projected_vertices[face.z].z;
                float2 p_screen = float2(pixel.x + 0.5, pixel.y + 0.5);

                // a) Getting camera space coordinates and then original coordinates
                float2 cp0 = float2(((2 * s * a * sp0.x) / (width)) - s * a, s - ((2 * s * sp0.y) / (height)));
                float2 cp1 = float2(((2 * s * a * sp1.x) / (width)) - s * a, s - ((2 * s * sp1.y) / (height)));
                float2 cp2 = float2(((2 * s * a * sp2.x) / (width)) - s * a, s - ((2 * s * sp2.y) / (height)));
                float2 cp = float2(((2 * s * a * p_screen.x) / (width)) - s * a, s - ((2 * s * p_screen.y) / (height)));

                // b) Getting the barycentric coordinate coefficeints in camera space
                float cp0cp1cp2_area = compute_area(cp0, cp1, cp2);
                if (cp0cp1cp2_area == 0) {
                    continue;
                }
                float cb0 = compute_area(cp, cp1, cp2) / cp0cp1cp2_area;
                float cb1 = compute_area(cp0, cp, cp2) / cp0cp1cp2_area;
                float cb2 = compute_area(cp0, cp1, cp) / cp0cp1cp2_area;

                // c) Getting the original barycentric coordinate coefficients
                float denom = cb0 / (-inv_depth0) + cb1 / (-inv_depth1) + cb2 / (-inv_depth2);
                float b0 = (cb0 / -inv_depth0) / denom;
                float b1 = (cb1 / -inv_depth1) / denom;
                float b2 = (cb2 / -inv_depth2) / denom;

                // d) and e) Get the depth and compare with the depth image of the pixel and update
                float p_depth = b0 * inv_depth0 + b1 * inv_depth1 + b2 * inv_depth2;

                uint depth_bits = asuint(p_depth);
                uint index = y * width + x;
                if (depth_bits == depth_buffer[index]) {
                    InterlockedMin(triangle_id_buffer[index], tid);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    uint id = tid.y * width + tid.x;
    uint face_id = triangle_id_buffer[id];

    if (face_id != 0xffffffff) {
        uint3 face = faces[face_id];

        float2 sp0 = float2(projected_vertices[face.x].x, projected_vertices[face.x].y);
        float2 sp1 = float2(projected_vertices[face.y].x, projected_vertices[face.y].y);
        float2 sp2 = float2(projected_vertices[face.z].x, projected_vertices[face.z].y);
        int2 pixel = int2(tid.x, tid.y);

        float inv_depth0 = projected_vertices[face.x].z;
        float inv_depth1 = projected_vertices[face.y].z;
        float inv_depth2 = projected_vertices[face.z].z;
        float2 p_screen = float2(tid.x + 0.5, tid.y + 0.5);

        // a) Getting camera space coordinates and then original coordinates
        float2 cp0 = float2(((2 * s * a * sp0.x) / (width)) - s * a, s - ((2 * s * sp0.y) / (height)));
        float2 cp1 = float2(((2 * s * a * sp1.x) / (width)) - s * a, s - ((2 * s * sp1.y) / (height)));
        float2 cp2 = float2(((2 * s * a * sp2.x) / (width)) - s * a, s - ((2 * s * sp2.y) / (height)));
        float2 cp = float2(((2 * s * a * p_screen.x) / (width)) - s * a, s - ((2 * s * p_screen.y) / (height)));

        // b) Getting the barycentric coordinate coefficeints in camera space
        float cp0cp1cp2_area = compute_area(cp0, cp1, cp2);
        if (cp0cp1cp2_area == 0) {
            return;
        }
        float cb0 = compute_area(cp, cp1, cp2) / cp0cp1cp2_area;
        float cb1 = compute_area(cp0, cp, cp2) / cp0cp1cp2_area;
        float cb2 = compute_area(cp0, cp1, cp) / cp0cp1cp2_area;

        // c) Getting the original barycentric coordinate coefficients
        float denom = cb0 / (-inv_depth0) + cb1 / (-inv_depth1) + cb2 / (-inv_depth2);
        float b0 = (cb0 / -inv_depth0) / denom;
        float b1 = (cb1 / -inv_depth1) / denom;
        float b2 = (cb2 / -inv_depth2) / denom;

        // d) and e) Get the depth and compare with the depth image of the pixel and update
        float p_depth = b0 * inv_depth0 + b1 * inv_depth1 + b2 * inv_depth2;
        float3 color = b0 * vertex_colors[face.x] + b1 * vertex_colors[face.y] + b2 * vertex_colors[face.z];
        image[tid] = float4(color, 1.0);
    }
}
