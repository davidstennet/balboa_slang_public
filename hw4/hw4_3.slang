[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }

    int2 xy = int2(tid % grid_size.x, tid / grid_size.x);
    float3 dx_sum = 0;
    int num_neighbor = 0;

    if (inv_masses[tid] == 0) {
        positions_out[tid] = positions_in[tid];
        return;
    }

    // Loop that adds all the forces from the surrounding particles
    int2 directions[12] = {int2(-1,0),  int2(0, 1), int2(0,-1), int2(1, 0),
                           int2(-1,-1), int2(-1,1), int2(1, 1), int2(1,-1),
                           int2(-2,0),  int2(0, 2), int2(0,-2), int2(2, 0)};
    float3 xi = positions_in[tid];
    for (int i = 0; i < 12; i++) {
        int2 neigh_xy = xy + directions[i];
        uint neigh_tid = neigh_xy.y * grid_size.x + neigh_xy.x;

        // Makes sure we have valid xy for our neighbor particle
        if (neigh_xy.x < 0 || neigh_xy.x >= grid_size.x || neigh_xy.y < 0 || neigh_xy.y >= grid_size.y) 
            continue;

        
        float3 xj = positions_in[neigh_tid];
        float wi = inv_masses[tid];
        float wj = inv_masses[neigh_tid];
        float lij = sqrt(pow(directions[i].x * spacing.x, 2) + pow(directions[i].y * spacing.y, 2));
        float3 d = xj - xi;

        dx_sum += (wi/(wi+wj)) * (length(d) - lij) * (d/length(d));
        num_neighbor++;
    }

    positions_out[tid] = xi + dx_sum/num_neighbor;
}
