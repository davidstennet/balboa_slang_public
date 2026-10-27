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

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1
    if (is_within_triangle(ipixel, sp0, sp1, sp2)) {
        image[ipixel] = float4(float3(color), 1.0);
    }
    else {
        image[ipixel] = float4(0.5, 0.5, 0.5, 1.0);
    }
}
