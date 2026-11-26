import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // Converts the screen space of tid to camera space
    float a = float(width) / float(height);
    float x = ((2*a*(tid.x + 0.5))/width) - a;
    float y = ((-2*(tid.y + 0.5))/height) + 1;

    // Calculates direction of the ray
    float3 d = float3(x, y, -1) / length(float3(x, y, -1));

    // Calculates whether there is an intersect with a ray and a sphere and finds its min_t for that ray along with the normal
    float3 n;
    float min_t = 1.0 / 0.0; // represents infinity
    for (int i = 0; i < num_spheres; i++) {

        float2 t;
        float3 x_c = spheres[i].center;
        float r = spheres[i].radius;
        
        bool hit_sphere = solve_quadratic(dot(d, d), -2 * dot(d, x_c), dot(x_c, x_c) - r*r, t);
        if (hit_sphere) {

            // If the first t value is a min else if the second t value is a min
            if (t.x > 0 && t.x < min_t && t.x <= t.y) {
                min_t = t.x;
                
                float3 p = min_t * d;
                n = (p - x_c)/r;
            }
            else if (t.y > 0 && t.y < min_t) {
                min_t = t.y;

                float3 p = min_t * d;
                n = (p - x_c)/r;
            }

        }
    }

    if (min_t != 1.0 / 0.0) {

        image[tid] = float4((n+1)/2, 1.0);
    }
    else {
        image[tid] = float4(background, 1.0);
    }

      
}
