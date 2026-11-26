[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      // Plane stores (a, b, c, d) where ax+by+cz=d
                      // a, b, c are normalized to a unit 3D vector
                      StructuredBuffer<float4> planes,
                      // Sphere stores center (xyz) + radius (w)
                      StructuredBuffer<float4> spheres,
                      // Contact plane: dot(n, x) = d, with unit normal n
                      // Set to (0, 0, 0, 0) if there is no contact
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      // How much should we shift the generated 
                      // contact plane in the normal direction
                      // (similar to how we shifted the floor by 0.03
                      //  earlier)
                      uniform float collision_distance,
                      // When the signed distance function
                      // evaluates to a value below 
                      // contact_margin + collision_distance
                      // (even when it's positive), we count the
                      // particle to be inside the collider and
                      // a collision event has happened.
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


    if (smallest_d < contact_margin + collision_distance) {
        contact_planes[tid] = float4(curr_contact_plane.xyz, curr_contact_plane.w + collision_distance);
    }
    else {
        contact_planes[tid] = float4(0, 0, 0, 0);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         StructuredBuffer<float3> old_positions,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         StructuredBuffer<float> neighbor_rest,
                         // Contact plane: dot(n, x) = d, with unit normal n
                         StructuredBuffer<float4> contact_planes,
                         uniform uint num_particles,
                         uniform float static_friction,
                         uniform float kinetic_friction) {
    if (tid >= num_particles) {
        return;
    }

    float3 collision_dx = 0;

    // Finds teh collision dx
    float4 plane = contact_planes[tid];
    if (!all(plane == float4(0, 0, 0, 0))) {
        // Initialize variables
        float3 x = positions_in[tid];
        float3 old_position = old_positions[tid];

        // Calculate the normal correction
        float3 n = contact_planes[tid].xyz;
        float d = dot(plane.xyz, x) - plane.w;
        float penetration_d = max(-d, 0.0);
        collision_dx += penetration_d * plane.xyz;

        // Calculate the tangential correction
        float3 tangent = x - old_position - dot(x - old_position, n) * n;
        if (length(tangent) <= static_friction * penetration_d) {
            collision_dx -= tangent;
        }
        else {
            collision_dx -= kinetic_friction * penetration_d * (tangent / length(tangent));
        }
    }


    // Same code from HW4_4 to find the constraint dx
    float3 constraint_dx = 0;
    int num_neighbor = 0;

    uint neigh_start = row_ptr[tid];
    uint neigh_end = row_ptr[tid + 1];
    float3 xi = positions_in[tid];
    for (int i = neigh_start; i < neigh_end; i++) {
        uint neigh_tid = neighbors[i];

        float3 xj = positions_in[neigh_tid];
        float wi = inv_masses[tid];
        float wj = inv_masses[neigh_tid];
        float lij = neighbor_rest[i];
        float3 d = xj - xi;

        constraint_dx += (wi / (wi + wj)) * (length(d) - lij) * (d / length(d));
        num_neighbor++;
    }

    positions_out[tid] = xi + constraint_dx / num_neighbor + collision_dx;
 
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    
    // Make sure its not infinity mass
    if (inv_masses[tid] == 0) {
        velocities[tid] = float3(0, 0, 0);
        return;
    }

    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;
}
