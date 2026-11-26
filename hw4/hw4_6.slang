struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;

    // which BVH node we should skip
    // to if this node is totally not
    // intersecting the ray
    // if next_index < 0, then we don't
    // have the next node to skip to
    int next_index;

    uint triangle_begin;
    uint triangle_end;

    bool is_leaf;
};

float point_aabb_distance(float3 x, float3 bmin, float3 bmax) {
    return length(x - clamp(x, bmin, bmax));
}

struct ClosestPoint {
    float3 position;
    float3 bary;
};

// From https://iquilezles.org/articles/distfunctions/
ClosestPoint closest_point_triangle(
        float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a, pa = p - a;
    float3 cb = c - b, pb = p - b;
    float3 ac = a - c, pc = p - c;
    float3 nor = cross(ba, ac);

    float t0 = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
    float t1 = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
    float t2 = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);

    float3 q0 = a + ba * t0;
    float3 q1 = b + cb * t1;
    float3 q2 = c + ac * t2;

    if (sign(dot(cross(ba, nor), pa)) +
        sign(dot(cross(cb, nor), pb)) +
        sign(dot(cross(ac, nor), pc)) < 2.0) {

        float d0 = dot(p - q0, p - q0);
        float d1 = dot(p - q1, p - q1);
        float d2 = dot(p - q2, p - q2);

        if (d0 < d1 && d0 < d2)
            return {q0, float3(1.0 - t0, t0, 0.0)};
        if (d1 < d2)
            return {q1, float3(0.0, 1.0 - t1, t1)};
        return {q2, float3(t2, 0.0, 1.0 - t2)};
    }

    float inv_nor2 = 1.0 / dot(nor, nor);
    float3 q = p - nor * (dot(nor, pa) * inv_nor2);

    float wb = dot(cross(pa, ac), nor) * inv_nor2;
    float wc = dot(cross(pa, ba), nor) * inv_nor2;
    return {q, float3(1.0 - wb - wc, wb, wc)};
}


[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      StructuredBuffer<float3> mesh_vertices,
                      StructuredBuffer<float3> mesh_vertex_normals,
                      StructuredBuffer<uint3> mesh_triangles,
                      StructuredBuffer<BVHNode> bvh_nodes,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform uint num_bvh_nodes,
                      uniform float collision_distance,
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }

    // If it is infinity mass this particle does not move 
    if (inv_masses[tid] == 0) {
        contact_planes[tid] = float4(0, 0, 0, 0);
        return;
    }

    float3 x = positions[tid];
    float4 curr_contact_plane;
    float smallest_d = 1e+20;


    // Iterate through the planes
    for (int i = 0; i < num_planes; i++) {
        float3 n = planes[i].xyz;
        float d = dot(n, x) - planes[i].w;
        if (d >= smallest_d) 
            continue;

        smallest_d = d;
        curr_contact_plane = planes[i];
    }

    // Iterate through the spheres
    for (int i = 0; i < num_spheres; i++) {
        float3 c = spheres[i].xyz;
        float r = spheres[i].w;
        float d = length(x - c) - r; // d = ||x-c|| - r
        if (d >= smallest_d) 
            continue;
        

        // Update d and plane value that would be returned
        smallest_d = d;
        float3 n = normalize(x - c);
        float3 contact_p = c + r * n;
        curr_contact_plane = float4(n.x, n.y, n.z, dot(n, contact_p));
    }
    

    // Iterate through triangles with their BVH
    int node_index = 0;
    float closest_unsigned_d = 1e+20;
    float selected_signed_d = 1e+20;
    while (node_index >= 0 && node_index < num_bvh_nodes) {

        BVHNode node = bvh_nodes[node_index];

        float in_bounding_box = point_aabb_distance(x, node.bbox_min, node.bbox_max);
        if (in_bounding_box > contact_margin + collision_distance) {
            node_index = node.next_index;
        }
        else if (node.is_leaf) {
            for (int i = node.triangle_begin; i < node.triangle_end; i++) {
                uint3 triangle = mesh_triangles[i];
                float3 v0 = mesh_vertices[triangle.x];
                float3 v1 = mesh_vertices[triangle.y];
                float3 v2 = mesh_vertices[triangle.z];

                ClosestPoint cp = closest_point_triangle(x, v0, v1, v2);

                float unsigned_d = distance(x, cp.position);
                if (unsigned_d < contact_margin + collision_distance) {
                    // Get normals
                    float3 n0 = mesh_vertex_normals[triangle.x];
                    float3 n1 = mesh_vertex_normals[triangle.y];
                    float3 n2 = mesh_vertex_normals[triangle.z];

                    float3 ng = normalize(cross(v1 - v0, v2 - v0));
                    float3 ns = normalize(cp.bary.x * n0 + cp.bary.y * n1 + cp.bary.z * n2);
                    if (dot(ng, ns) < 0.0)
                        ng = -ng;

                    float signed_d = dot(x - cp.position, ng);
                    if (unsigned_d < closest_unsigned_d) {
                        closest_unsigned_d = unsigned_d;
                        selected_signed_d = signed_d;

                        curr_contact_plane = float4(ng.x, ng.y, ng.z, dot(ng, cp.position));
                    }
                }
            }

            node_index++;
        }
        else {
            node_index++;
        }

    }
    // Checks to see if we are using the triangle as the shape
    if (closest_unsigned_d < contact_margin + collision_distance && selected_signed_d < smallest_d) {
        smallest_d = selected_signed_d;
    }


    if (smallest_d < contact_margin + collision_distance) {
        contact_planes[tid] = float4(curr_contact_plane.xyz, curr_contact_plane.w + collision_distance);
    }
    else {
        contact_planes[tid] = float4(0, 0, 0, 0);
    }
}
