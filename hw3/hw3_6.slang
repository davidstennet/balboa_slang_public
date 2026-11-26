import hw3_common;

bool ray_sphere_intersection(Sphere sphere, float3 o, float3 d, out float2 t) {
    float3 x_c = sphere.center;
    float r = sphere.radius;

    float a = dot(d, d);
    float b = 2 * dot(d, o - x_c);
    float c = dot(o - x_c, o - x_c) - r * r;

    return solve_quadratic(a, b, c, t);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
                  StructuredBuffer<BVHNode> bvh_nodes,
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
    float W = 2.0 * tan(radians(camera.fov_x) / 2.0);
    float x = (((2 * a * (tid.x + 0.5)) / width) - a) * (W / (2 * a));
    float y = (((-2 * (tid.y + 0.5)) / height) + 1) * (W / (2 * a));

    // Calculates direction of the camera ray
    float3 cam_o = float3(0.0, 0.0, 0.0);
    float3 cam_d = float3(x, y, -1) / length(float3(x, y, -1));

    // Transform camera to world space
    float3 world_o = xform_pt(camera.cam_to_world, cam_o);
    float3 world_d = xform_vec(camera.cam_to_world, cam_d);

    // These two variables come from 3.3 for bouncing mirror rays
    float3 C_total = 0.0;
    float3 throughput = 1.0;
    for (int b = 0; b < max_depth; b++) {

        // Calculates whether there is an intersect with a ray and a sphere and finds its min_t for that ray along with the normal
        float3 n;
        float3 p;
        int shape_i;
        float min_t = 1.0 / 0.0; // represents infinity
        bool hit_anything = false;
        for (int i = 0; i < num_shapes; i++) {

            Shape curr_shape = shapes[i];

            // Converts the world ray to object space
            float3 object_o = xform_pt(curr_shape.world_to_obj, world_o);
            float3 object_d = xform_vec(curr_shape.world_to_obj, world_d);

            float2 t;
            if (curr_shape.type == ShapeType::Sphere) {
                Sphere sphere = spheres[curr_shape.index];
                bool hit_sphere = ray_sphere_intersection(sphere, object_o, object_d, t);
                if (hit_sphere) {

                    // If the first t value is a min else if the second t value is a min
                    if (t.x > 0 && t.x < min_t && t.x <= t.y) {
                        min_t = t.x;

                        p = object_o + min_t * object_d;
                        n = (p - sphere.center) / sphere.radius;
                        shape_i = i;
                        hit_anything = true;
                    }
                    else if (t.y > 0 && t.y < min_t) {
                        min_t = t.y;

                        p = object_o + min_t * object_d;
                        n = (p - sphere.center) / sphere.radius;
                        shape_i = i;
                        hit_anything = true;
                    }
                }
            }
            else if (curr_shape.type == ShapeType::TriangleMesh) {
                TriangleMesh mesh = meshes[curr_shape.index];

                int node_index = mesh.bvh_root_index;
                int end_index = bvh_nodes[node_index].next_index;
                while (node_index < end_index) {

                    BVHNode node = bvh_nodes[node_index];

                    bool intersects_aabb = intersect_aabb(node.bbox_min, node.bbox_max, object_o, object_d, 0.0, min_t);
                    if (!intersects_aabb) {
                        node_index = node.next_index;
                    }   
                    else if (node.is_leaf) {
                        for (int j = node.triangle_begin; j < node.triangle_end; j++) {

                            float3 b;
                            uint3 triangle = triangles[j] + mesh.vertex_offset;

                            float3 v0 = vertices[triangle.x];
                            float3 v1 = vertices[triangle.y];
                            float3 v2 = vertices[triangle.z];

                            float3 n0 = vertex_normals[triangle.x];
                            float3 n1 = vertex_normals[triangle.y];
                            float3 n2 = vertex_normals[triangle.z];

                            bool intersects_triangle = intersect_tri(v0, v1, v2, object_o, object_d, 0.0, min_t, b, t.x);
                            if (intersects_triangle && t.x < min_t) {
                                n = normalize(b.x * n0 + b.y * n1 + b.z * n2);
                                p = b.x * v0 + b.y * v1 + b.z * v2;
                                min_t = t.x;
                                shape_i = i;
                                hit_anything = true;
                            }
                        }

                        node_index++;
                    }
                    else { // this runs if we just keep traversing the tree and we hit the AABB
                        node_index++;
                    }

                }

            }
        }

        // If statements for calculating what color the pixel needs to be after finding min_t
        if (hit_anything) {

            // Convert p and n back into world space
            p = xform_pt(shapes[shape_i].obj_to_world, p);
            n = xform_vec(transpose(shapes[shape_i].world_to_obj), n);
            n = normalize(n);

            float3 contribution = 0.0;
            for (int i = 0; i < num_lights; i++) {

                // Origin of point light
                float3 o_light = lights[i].position;

                // Ray origin and direction from point to point light
                float3 o_shadow = p + EPSILON * n;
                float3 d_shadow = (o_light - p) / length(o_light - p);

                // Gets the t (upper range) for where the light will be
                float t_light = length(o_light - o_shadow);

                // Figures out if intersects a shape on the way to the point light
                bool intersects_shape = false;
                for (int j = 0; j < num_shapes; j++) {

                    Shape curr_shape = shapes[j];

                    // Converts the world ray to object space
                    float3 object_o_shadow = xform_pt(curr_shape.world_to_obj, o_shadow);
                    float3 object_d_shadow = xform_vec(curr_shape.world_to_obj, d_shadow);

                    float2 t;
                    if (curr_shape.type == ShapeType::Sphere) {
                        Sphere sphere = spheres[curr_shape.index];
                        bool hit_sphere = ray_sphere_intersection(sphere, object_o_shadow, object_d_shadow, t);
                        if (hit_sphere) {
                            if ((t.x > 0 && t.x < t_light) || (t.y > 0 && t.y < t_light)) {
                                intersects_shape = true;
                                break;
                            }
                        }
                    }
                    else if (curr_shape.type == ShapeType::TriangleMesh) {
                        TriangleMesh mesh = meshes[curr_shape.index];

                        int node_index = mesh.bvh_root_index;
                        int end_index = bvh_nodes[node_index].next_index;
                        while (node_index < end_index && !intersects_shape) {
                            BVHNode node = bvh_nodes[node_index];

                            bool intersects_aabb = intersect_aabb(node.bbox_min, node.bbox_max, object_o_shadow, object_d_shadow, 0.0, t_light);
                            if (!intersects_aabb) {
                                node_index = node.next_index;
                            }
                            else if (node.is_leaf) {
                                for (int j = node.triangle_begin; j < node.triangle_end; j++) {

                                    float3 b;
                                    uint3 triangle = triangles[j] + mesh.vertex_offset;

                                    float3 v0 = vertices[triangle.x];
                                    float3 v1 = vertices[triangle.y];
                                    float3 v2 = vertices[triangle.z];

                                    bool intersects_triangle = intersect_tri(v0, v1, v2, object_o_shadow, object_d_shadow, 0.0, t_light, b, t.x);
                                    if (intersects_triangle) {
                                        intersects_shape = true;
                                        break;
                                    }
                                }

                                node_index++;
                            }
                            else { // this runs if we just keep traversing the tree and we hit the AABB
                                node_index++;
                            }     
                        }    
                    }
                }

                // Adds color contribution from point light
                if (!intersects_shape) {
                    float3 I = lights[i].intensity;
                    float3 Kd = shapes[shape_i].Kd;
                    float d = length(o_light - p);
                    float3 contribution_i = I * (1 / (d * d)) * Kd * max(0, dot(n, d_shadow));

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
        world_d = reflect(world_d, n);
        world_o = p + EPSILON * n;
    }

    image[tid] = float4(C_total, 1.0);
}
