[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengthes of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }

    if (inv_masses[tid] == 0) {
        positions_out[tid] = positions_in[tid];
        return;
    }

    float3 dx_sum = 0;
    int num_neighbor = 0;

    uint neigh_start = row_ptr[tid];
    uint neigh_end = row_ptr[tid+1];
    float3 xi = positions_in[tid];
    for (int i = neigh_start; i < neigh_end; i++) {
        uint neigh_tid = neighbors[i];

        float3 xj = positions_in[neigh_tid];
        float wi = inv_masses[tid];
        float wj = inv_masses[neigh_tid];
        float lij = neighbor_rest[i]; 
        float3 d = xj - xi;

        dx_sum += (wi/(wi+wj)) * (length(d) - lij) * (d/length(d));
        num_neighbor++;
    }

    positions_out[tid] = xi + dx_sum/num_neighbor;
}
