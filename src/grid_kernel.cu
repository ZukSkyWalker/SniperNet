extern "C" __global__
void compute_grid_info(const float* zs, const int len_zs, float* base_map, float* height_map, 
                       unsigned int* density_map, const float dz, const float z_max, const float z_grid_gap,
                       const float sig_height_range) {

	int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = idx_x * gridDim.y + idx_y;  // Linearized index across the grid
	if (idx >= len_zs) return;  // If idx is out of bounds of zs

	// Histogram (using atomic operations to avoid race conditions)
	int bin_count = int(z_max / dz);
	extern __shared__ unsigned int cnts[];  // Allocate shared memory for histogram
	for (int i = threadIdx.x; i < bin_count; i += blockDim.x) {
			cnts[i] = 0;
	}
	__syncthreads();
	
	float z = zs[idx];
	int bin_idx = int(z / dz);
	atomicAdd(&cnts[bin_idx], 1);

	__syncthreads();

	// Following the rest of the logic in your function
	float base = z;  // As z.min() within zs will be z itself
	float height = z;  // Same logic

	unsigned int pre_i = 0, n_pts = 0;
	for (int i = 0; i < bin_count; i++) {
		if (cnts[i] > 0) {
			if (pre_i + z_grid_gap < i && (i * dz > base + sig_height_range)) {
				height = pre_i * dz;
				break;
			}

			pre_i = i;
			n_pts += 1;
		}
	}

	base_map[idx] = base;
	height_map[idx] = height;
	density_map[idx] = n_pts;
}