extern "C" __global__
void local_maxima(const float* image, float threshold, int delta, int delta_fit,
				  unsigned short* z_out, unsigned short* x_out, unsigned short* y_out,
				  unsigned int* count,
				  int depth, int height, int width, int max_points) {
	// Get flattened index
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= depth * height * width) {
		return;
	}

	// Convert flat index to 3D coordinates
	int z = idx / (height * width);
	int temp = idx % (height * width);
	int x = temp / width;
	int y = temp % width;


	// Check if above threshold
	if (image[idx] <= threshold) {
		return;
	}

	// Check if it's a local maximum in the neighborhood
	bool is_max = true;
	for (int dz = -delta; dz <= delta; dz++) {
		for (int dx = -delta; dx <= delta; dx++) {
			for (int dy = -delta; dy <= delta; dy++) {
				// Skip the center point
				if (dz == 0 && dx == 0 && dy == 0) {
					continue;
				}

				// Check if within spherical mask
				if ((dz*dz + dx*dx + dy*dy) > (delta*delta)) {
					continue;
				}

				int nz = z + dz;
				int nx = x + dx;
				int ny = y + dy;
				

				// Apply reflect only if out of bounds
				if (nz < 0 || nz >= depth) {
					nz = (nz < 0) ? -nz : 2 * depth - nz - 2;
				}
				if (nx < 0 || nx >= height) {
					nx = (nx < 0) ? -nx : 2 * height - nx - 2;
				}
				if (ny < 0 || ny >= width) {
					ny = (ny < 0) ? -ny : 2 * width - ny - 2;
				}

				if (image[idx] < image[nz * height * width + nx * width + ny]) {
					is_max = false;
					break;
				}
			}
			if (!is_max) break;
		}
		if (!is_max) break;
	}

	if (is_max) {
		// If it's a local maximum, add to output
		unsigned int pos = atomicAdd(count, 1);
		if (pos < max_points) {
		   	z_out[pos] = z;
		   	x_out[pos] = x;
		   	y_out[pos] = y;
		}
	}
}

#define MAX_KERNEL_POINTS 515
extern "C" __global__
void delta_fit_cross_corr(
	const float* __restrict__ image,
	const unsigned short* __restrict__ raw,
	unsigned short* __restrict__ z_out,   // (num_maxima)
	unsigned short* __restrict__ x_out,   // (num_maxima)
	unsigned short* __restrict__ y_out,   // (num_maxima)
	float* __restrict__ output,		// (num_maxima, 8) [zc, xc, yc, background, habs, h, cross_corr, delta_fit_value]
	int num_maxima,
	int Z, int X, int Y,
	int delta_fit,
	float sigmaZ, float sigmaXY
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_maxima) return;

	int z0 = z_out[idx];
	int x0 = x_out[idx];
	int y0 = y_out[idx];

	float sum_val = 0.0f;
	float sum_z = 0.0f;
	float sum_x = 0.0f;
	float sum_y = 0.0f;
	float min_val = 1e20f;

	// Cross-correlation computation (Gaussian weights)
	float norm_G[MAX_KERNEL_POINTS];
	float sample_vals[MAX_KERNEL_POINTS];
	float raw_vals[MAX_KERNEL_POINTS];
	int count = 0;

	// Step 1: Collect all samples and calculate Gaussian weights
	for (int dz = -delta_fit; dz <= delta_fit; ++dz) {
		for (int dx = -delta_fit; dx <= delta_fit; ++dx) {
			for (int dy = -delta_fit; dy <= delta_fit; ++dy) {
			    if (dz * dz + dx * dx + dy * dy > delta_fit * delta_fit) continue;
			    if (count >= 125) continue;  // Safety check

			    int zz = z0 + dz;
			    int xx = x0 + dx;
			    int yy = y0 + dy;

			    // Reflect if out of bounds
			    zz = zz < 0 ? -zz : (zz >= Z ? 2 * Z - zz - 2 : zz);
			    xx = xx < 0 ? -xx : (xx >= X ? 2 * X - xx - 2 : xx);
			    yy = yy < 0 ? -yy : (yy >= Y ? 2 * Y - yy - 2 : yy);

			    // Double-check bounds
			    if (zz < 0 || zz >= Z || xx < 0 || xx >= X || yy < 0 || yy >= Y) continue;

			    int idx_img = zz * (X * Y) + xx * Y + yy;
			    float val = image[idx_img];
			    sample_vals[count] = val;
				//raw_vals[count] = raw[idx_img];
				raw_vals[count] = (float)raw[idx_img];

			    if (val < min_val) min_val = val;

			    // Calculate Gaussian weight
			    float norm = expf(-(dz * dz / (2 * sigmaZ * sigmaZ) +
			                      dx * dx / (2 * sigmaXY * sigmaXY) +
			                      dy * dy / (2 * sigmaXY * sigmaXY)));
			    norm_G[count] = norm;

			    // For weighted center calculation
				// geometric center
			    //sum_val += val;
			    //sum_z += dz * val;
			    //sum_x += dx * val;
			    //sum_y += dy * val;
				// intensity center
				sum_val += val - min_val;  // Subtract background from each value
				sum_z += (zz) * (val - min_val);  // Use actual coordinate (not offset)
				sum_x += (xx) * (val - min_val);  // and subtract background
				sum_y += (yy) * (val - min_val);

			    count++;
			}
		}
	}
	if (count == 0) return;  // Safety check

	// Step 2: Calculate mean and std of Gaussian weights
	float mean_G = 0.0f;
	float var_G = 0.0f;

	for (int i = 0; i < count; i++) {
		mean_G += norm_G[i];
	}
	mean_G /= count;

	for (int i = 0; i < count; i++) {
		float diff = norm_G[i] - mean_G;
		var_G += diff * diff;
	}
	var_G /= count;
	float std_G = sqrtf(var_G);

	// Normalize Gaussian weights
	for (int i = 0; i < count; i++) {
		norm_G[i] = (norm_G[i] - mean_G) / std_G;
	}

	// Step 3: Calculate mean and std of image values
	float mean_sample = 0.0f;
	float var_sample = 0.0f;
	float mean_raw = 0.0f;
	float var_raw = 0.0f;

	for (int i = 0; i < count; i++) {
		mean_sample += sample_vals[i];
		mean_raw += raw_vals[i];
	}
	mean_sample /= count;
	mean_raw /= count;

	for (int i = 0; i < count; i++) {
		float diff_sample = sample_vals[i] - mean_sample;
		float diff_raw = raw_vals[i] - mean_raw;
		var_sample += diff_sample * diff_sample;
		var_raw += diff_raw * diff_raw;
	}
	var_sample /= count;
	var_raw /= count;
	float std_sample = sqrtf(var_sample);
	float std_raw = sqrtf(var_raw);
	// Step 4: Calculate normalized cross-correlation
	float hn = 0.0f;  // Cross-correlation with image
	float a = 0.0f;   // Cross-correlation with raw
	
	for (int i = 0; i < count; i++) {
		float norm_sample = (sample_vals[i] - mean_sample) / std_sample;
		float norm_raw = (raw_vals[i] - mean_raw) / std_raw;
		hn += norm_sample * norm_G[i];
		a += norm_raw * norm_G[i];
	}
	hn /= count;
	a /= count;

	// Calculate center coordinates
	// geometric center
	//float center_z = (count > 0) ? z0 + sum_z / (float)count : z0;
	//float center_x = (count > 0) ? x0 + sum_x / (float)count : x0;
	//float center_y = (count > 0) ? y0 + sum_y / (float)count : y0;
	// intensity center
	float center_z = (sum_val > 0) ? sum_z / sum_val : z0;
	float center_x = (sum_val > 0) ? sum_x / sum_val : x0;
	float center_y = (sum_val > 0) ? sum_y / sum_val : y0;
	
	int center_idx = z0 * (X * Y) + x0 * Y + y0;
	// Output: [zc, xc, yc, background, a, habs, hn, h]
	output[idx * 8 + 0] = center_z;
	output[idx * 8 + 1] = center_x;
	output[idx * 8 + 2] = center_y;
	output[idx * 8 + 3] = min_val;  // background
	output[idx * 8 + 4] = a;		// Cross-correlation with raw
	output[idx * 8 + 5] = image[center_idx];
	output[idx * 8 + 6] = hn;	   // Cross-correlation with image
	output[idx * 8 + 7] = (float)raw[center_idx];
}
