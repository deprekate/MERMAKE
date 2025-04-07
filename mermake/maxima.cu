extern "C" __global__
void local_maxima(const float* image, float threshold, int delta, int delta_fit,
				  float* z_out, float* x_out, float* y_out, unsigned int* count,
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

extern "C" __global__
void delta_fit(
    const float* __restrict__ image,
    //const float* __restrict__ im_raw,  // can be NULL
    const float* __restrict__ z_out,   // (num_maxima)
    const float* __restrict__ x_out,   // (num_maxima)
    const float* __restrict__ y_out,   // (num_maxima)
    float* __restrict__ output,        // (num_maxima, 6) [zc, xc, yc, background, habs, h]
    int num_maxima,
    int Z, int X, int Y,
    int delta_fit
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_maxima) return;

    int z0 = (int)z_out[idx];
    int x0 = (int)x_out[idx];
    int y0 = (int)y_out[idx];

    float sum_val = 0.0f;
    float sum_z = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float min_val = 1e20f;

    for (int dz = -delta_fit; dz <= delta_fit; ++dz) {
        for (int dx = -delta_fit; dx <= delta_fit; ++dx) {
            for (int dy = -delta_fit; dy <= delta_fit; ++dy) {
                if (dz * dz + dx * dx + dy * dy > delta_fit * delta_fit) continue;

                int zz = z0 + dz;
                int xx = x0 + dx;
                int yy = y0 + dy;

                // Reflect if out of bounds
                zz = zz < 0 ? -zz : (zz >= Z ? 2 * Z - zz - 2 : zz);
                xx = xx < 0 ? -xx : (xx >= X ? 2 * X - xx - 2 : xx);
                yy = yy < 0 ? -yy : (yy >= Y ? 2 * Y - yy - 2 : yy);

                float val = image[zz * (X * Y) + xx * Y + yy];

                if (val < min_val) min_val = val;
                sum_val += val;
                sum_z += dz * val;
                sum_x += dx * val;
                sum_y += dy * val;
            }
        }
    }

    float center_z = (sum_val > 0) ? z0 + sum_z / sum_val : z0;
    float center_x = (sum_val > 0) ? x0 + sum_x / sum_val : x0;
    float center_y = (sum_val > 0) ? y0 + sum_y / sum_val : y0;

	/*
    float habs = im_raw
        ? im_raw[z0 * (X * Y) + x0 * Y + y0]
        : image[z0 * (X * Y) + x0 * Y + y0];
	*/
    // Output: [zc, xc, yc, background, habs, h]
    output[idx * 8 + 0] = center_z;
    output[idx * 8 + 1] = center_x;
    output[idx * 8 + 2] = center_y;
    output[idx * 8 + 3] = min_val;
    //output[idx * 8 + 6] = habs;
    //output[idx * 8 + 7] = image[z0 * (X * Y) + x0 * Y + y0]; // h
}

