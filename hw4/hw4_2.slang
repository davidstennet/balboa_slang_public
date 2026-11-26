[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             // Updated position
             RWStructuredBuffer<float3> positions,
             // Store the position before update
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             // time step size
             uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    // If the mass is infinity of this particle then don't let it move
    if (inv_masses[tid] == 0) {
        old_positions[tid] = positions[tid];
        velocities[tid] = 0;
        return;
    }

    // Initial condition variables
    float3 grav_accel = float3(0, -9.8, 0);
    float3 posit = positions[tid];
    float3 veloc = velocities[tid];

    // Computing symplectic Euler
    float3 next_veloc = veloc + dt * grav_accel;
    float3 next_posit = posit + dt * next_veloc;

    // Update the position and velocities with final value
    positions[tid] = next_posit;
    old_positions[tid] = posit;
    velocities[tid] = next_veloc;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }

    bool has_left_neighbor = (tid == 0) ? false : true;
    bool has_right_neighbor = (tid == num_particles -1) ? false : true;
    float3 xi = positions_in[tid];
    float3 dx_0;
    float3 dx_1;

    // Iterates through either 1 or 2 neighbors if the mass is not infinity to get dx
    if (inv_masses[tid] == 0) {
        dx_0 = 0;
        dx_1 = 0;
    }
    else {
        float3 wi = inv_masses[tid];

        // Gets the dx for the left neighbor
        if (has_left_neighbor) {
            float3 xj = positions_in[tid-1];
            float wj = inv_masses[tid-1];
            float lij = rest_lengths[tid-1];
            float3 d = xj - xi;

            dx_0 = (wi/(wi+wj)) * (length(d) - lij) * (d/length(d));
        }

        // Gets the dx for the right neighbor
        if (has_right_neighbor) {
            float3 xj = positions_in[tid+1];
            float wj = inv_masses[tid+1];
            float lij = rest_lengths[tid];
            float3 d = xj - xi;

            dx_1 = (wi/(wi+wj)) * (length(d) - lij) * (d/length(d));
        }
    }

    // Updates the output vector depending on what number of neighbors there are
    if (!has_right_neighbor) {
        positions_out[tid] = xi + dx_0;
    }
    else if (!has_left_neighbor) {
        positions_out[tid] = xi + dx_1;
    }
    else {
        positions_out[tid] = xi + (dx_0 + dx_1)/2;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     // time step size
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.

    // Get the positions to be correct value
    float3 posit = positions[tid];
    if (posit.y <= -0.97) {
        posit.y = -0.97;
    }

    // Update position and velocity to be correct
    positions[tid] = posit;
    velocities[tid] = (posit - old_positions[tid]) / dt; // this is dx/dt
}
