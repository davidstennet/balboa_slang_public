[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    // Initial condition variables
    float3 grav_accel = float3(0, -9.8, 0);
    float3 posit = positions[tid];
    float3 veloc = velocities[tid];

    // Computing symplectic Euler
    float3 next_veloc = veloc + dt * grav_accel;
    float3 next_posit = posit + dt * next_veloc;
    if (next_posit.y <= -0.97) {
        next_posit.y = -0.97;
        next_veloc.y = -next_veloc.y;
    }

    // Update the position and velocities with final value
    positions[tid] = next_posit;
    velocities[tid] = next_veloc;
}

