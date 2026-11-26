import hw3_common;

bool ray_sphere_intersection(Sphere sphere, float3 o, float3 d, out float2 t) {
    float3 x_c = sphere.center;
    float r = sphere.radius;

    float a = dot(d, d);
    float b = 2 * dot(d, o - x_c);
    float c = dot(o - x_c, o - x_c) - r*r;

    return solve_quadratic(a, b, c, t);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
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

    // Calculates direction of the camera ray
    float3 cam_o = float3(0.0, 0.0, 0.0);
    float3 cam_d = float3(x, y, -1) / length(float3(x, y, -1));

    
    float3 C_total = 0.0;
    float3 throughput = 1.0;
    for (int b = 0; b < max_depth; b++) {
        // Calculates whether there is an intersect with a ray and a sphere and finds its min_t for that ray along with the normal
        float3 n;
        float3 p;
        int shape_i;
        float min_t = 1.0 / 0.0; // represents infinity
        for (int i = 0; i < num_shapes; i++) {

            float2 t;
            Sphere sphere = spheres[shapes[i].index];
            bool hit_sphere = ray_sphere_intersection(sphere, cam_o, cam_d, t);
            if (hit_sphere) {

                // If the first t value is a min else if the second t value is a min
                if (t.x > 0 && t.x < min_t && t.x <= t.y) {
                    min_t = t.x;
                    
                    p = min_t * cam_d;
                    n = (p - sphere.center)/sphere.radius;
                    shape_i = i;
                }
                else if (t.y > 0 && t.y < min_t) {
                    min_t = t.y;

                    p = min_t * cam_d;
                    n = (p - sphere.center)/sphere.radius;
                    shape_i = i;
                }

            }
        }


        // If statements for calculating what color the pixel needs to be after finding min_t
        if (min_t != 1.0 / 0.0) {

            // Now that we have found a min_t for this pixel we need to see if there needs to be a shadow
            // image[tid] = float4(shapes[shape_i].Kd, 1.0);
            
            float3 contribution = 0.0;
            for (int i = 0; i < num_lights; i++) {

                // Origin of point light
                float3 o_light = lights[i].position;
                
                // Gets the t (upper range) for where the light will be
                float t_light = length(o_light - p);

                // Ray origin and direction from point to point light
                float3 o_shadow = p + EPSILON*n;
                float3 d_shadow = (o_light - p) / length(o_light - p);

                // Figures out if intersects a shape on the way to the point light
                bool intersects_shape = false;
                for (int j = 0; j < num_shapes; j++) {
                    
                    float2 t;
                    Sphere sphere = spheres[shapes[j].index];
                    bool hit_sphere = ray_sphere_intersection(sphere, o_shadow, d_shadow, t);

                    if (hit_sphere) {
                        if ((t.x > 0 && t.x < t_light) || (t.y > 0 && t.y < t_light)) {
                            intersects_shape = true;
                            break;
                        }
                    }
                }

                // Adds color contribution from point light
                if (!intersects_shape) {
                    float3 I = lights[i].intensity;
                    float3 Kd = shapes[shape_i].Kd;
                    float d = length(o_light - p);
                    float3 contribution_i = I * (1/(d*d)) * Kd * max(0, dot(n, d_shadow));

                    contribution += contribution_i;
                }
            }

            C_total += throughput * contribution;
        }
        else {
            // This runs if the ray doesn't hit any shapes and thus we assign the background color
            C_total += throughput * background;
            break;
        }

        throughput *= shapes[shape_i].Kr;
        cam_d = reflect(cam_d, n);
        cam_o = p + EPSILON*n;
    }

    
    
    image[tid] = float4(C_total, 1.0);
}
