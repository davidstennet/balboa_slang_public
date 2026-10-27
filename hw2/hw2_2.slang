// HW 2.1 Function
bool is_within_triangle(int2 ipixel, float2 sp0, float2 sp1, float2 sp2) {
    // Compute Edges
    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    // Compute Normals
    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    // Variables for computing edge dot prodcut for loop
    float2 p = float2(ipixel.x + 0.5, ipixel.y + 0.5);
    float2 points[]  = { sp0, sp1, sp2 };
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
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    if (is_within_triangle(ipixel, sp0, sp1, sp2)) {

        float2 p_screen = float2(ipixel.x + 0.5, ipixel.y + 0.5);

        // a) Getting camera space coordinates and then original coordinates
        float2 cp0 = float2(((2 * s * a * sp0.x) / (width)) - s * a, s - ((2 * s * sp0.y) / (height)));
        float2 cp1 = float2(((2 * s * a * sp1.x) / (width)) - s * a, s - ((2 * s * sp1.y) / (height)));
        float2 cp2 = float2(((2 * s * a * sp2.x) / (width)) - s * a, s - ((2 * s * sp2.y) / (height)));
        float2 cp = float2(((2 * s * a * p_screen.x) / (width)) - s * a, s - ((2 * s * p_screen.y) / (height)));

        // b) Getting the barycentric coordinate coefficeints in camera space
        float cp0cp1cp2_area = compute_area(cp0, cp1, cp2);
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
        if (p_depth < depth_image[ipixel]) {
            depth_image[ipixel] = p_depth;

            float3 color = b0 * c0 + b1 * c1 + b2 * c2;
            color_image[ipixel] = float4(color, 1.0);
        }
    }

}
