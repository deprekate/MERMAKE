extern "C" __global__
void box_1d(const float* input, float* output,
			int size_z, int size_x, int size_y,
			int delta, int axis)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int total_size = size_z * size_x * size_y;
	if (idx >= total_size) return;

	// Convert linear index to ZXY coordinates
	int z = idx / (size_x * size_y);
	int temp = idx % (size_x * size_y);
	int x = temp / size_y;
	int y = temp % size_y;

	float sum = 0.0f;
	int start, end;
	int num = 0;

	if (axis == 0) {  // Z-axis
		start = max(0, z - delta);
		end = min(size_z - 1, z + delta);
		for (int i = start; i <= end; ++i) {
			int pos = i * size_x * size_y + x * size_y + y;
			sum += input[pos];
			num++;
		}
	} else if (axis == 1) {  // X-axis
		start = max(0, x - delta);
		end = min(size_x - 1, x + delta);
		for (int i = start; i <= end; ++i) {
			int pos = z * size_x * size_y + i * size_y + y;
			sum += input[pos];
			num++;
		}
	} else if (axis == 2) {  // Y-axis
		start = max(0, y - delta);
		end = min(size_y - 1, y + delta);
		for (int i = start; i <= end; ++i) {
			int pos = z * size_x * size_y + x * size_y + i;
			sum += input[pos];
			num++;
		}
	}

	// Avoid division by zero and handle potential NaN values
	if (num > 0) {
		output[idx] = sum / num;
	} else {
		output[idx] = 0.0f;  // Default value if no elements were added
	}
}

extern "C" __global__
void optimized_box_1d(const float* input, float* output,
					 int size_x, int size_y, int size_z,
					 int delta, int axis)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int total_size = size_x * size_y * size_z;
	if (idx >= total_size) return;

	int x = idx / (size_y * size_z);
	int temp = idx % (size_y * size_z);
	int y = temp / size_z;
	int z = temp % size_z;

	float sum = 0.0f;
	int count = 0;
	int start, end;

	// Use register cache for faster access when possible
	if (axis == 0) {  // X-axis
		start = max(0, x - delta);
		end = min(size_x - 1, x + delta);

		// Use manual loop unrolling for performance
		for (int i = start; i <= end; i += 4) {
			if (i <= end) {
				int pos = i * size_y * size_z + y * size_z + z;
				sum += input[pos];
				count++;
			}
			if (i+1 <= end) {
				int pos = (i+1) * size_y * size_z + y * size_z + z;
				sum += input[pos];
				count++;
			}
			if (i+2 <= end) {
				int pos = (i+2) * size_y * size_z + y * size_z + z;
				sum += input[pos];
				count++;
			}
			if (i+3 <= end) {
				int pos = (i+3) * size_y * size_z + y * size_z + z;
				sum += input[pos];
				count++;
			}
		}
	}
	else if (axis == 1) {  // Y-axis
		start = max(0, y - delta);
		end = min(size_y - 1, y + delta);

		// Preserve the original loop but try to optimize memory access pattern
		for (int i = start; i <= end; i++) {
			int pos = x * size_y * size_z + i * size_z + z;
			sum += input[pos];
			count++;
		}
	}
	else if (axis == 2) {  // Z-axis
		start = max(0, z - delta);
		end = min(size_z - 1, z + delta);

		// Z-axis access is the most efficient (coalesced memory), so keep it simple
		for (int i = start; i <= end; i++) {
			int pos = x * size_y * size_z + y * size_z + i;
			sum += input[pos];
			count++;
		}
	}

	// Avoid division by zero
	if (count > 0) {
		output[idx] = sum / count;
	} else {
		output[idx] = 0.0f;
	}
}
extern "C" __global__
void optimized_box_1d_reflect(const float* input, float* output,
							int size_x, int size_y, int size_z,
							int delta, int axis)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int total_size = size_x * size_y * size_z;
	if (idx >= total_size) return;

	int x = idx / (size_y * size_z);
	int temp = idx % (size_y * size_z);
	int y = temp / size_z;
	int z = temp % size_z;

	float sum = 0.0f;
	int count = 0;

	// Check if we need reflection (i.e., we're near an edge)
	bool needs_reflection = false;

	if (axis == 0) {
		needs_reflection = (x - delta < 0) || (x + delta >= size_x);
	} else if (axis == 1) {
		needs_reflection = (y - delta < 0) || (y + delta >= size_y);
	} else if (axis == 2) {
		needs_reflection = (z - delta < 0) || (z + delta >= size_z);
	}

	// Use different code paths for better performance in the common case
	if (needs_reflection) {
		// Handle reflection case
		if (axis == 0) {  // X-axis
			for (int i = x - delta; i <= x + delta; i++) {
				// Apply reflection for out-of-bounds indices
				int reflected_i = i;
				if (reflected_i < 0) {
					reflected_i = -reflected_i;  // Reflect across 0
				} else if (reflected_i >= size_x) {
					reflected_i = 2 * size_x - reflected_i - 2;  // Reflect across size_x-1
				}

				int pos = reflected_i * size_y * size_z + y * size_z + z;
				sum += input[pos];
				count++;
			}
		}
		else if (axis == 1) {  // Y-axis
			for (int i = y - delta; i <= y + delta; i++) {
				// Apply reflection for out-of-bounds indices
				int reflected_i = i;
				if (reflected_i < 0) {
					reflected_i = -reflected_i;  // Reflect across 0
				} else if (reflected_i >= size_y) {
					reflected_i = 2 * size_y - reflected_i - 2;  // Reflect across size_y-1
				}

				int pos = x * size_y * size_z + reflected_i * size_z + z;
				sum += input[pos];
				count++;
			}
		}
		else if (axis == 2) {  // Z-axis
			for (int i = z - delta; i <= z + delta; i++) {
				// Apply reflection for out-of-bounds indices
				int reflected_i = i;
				if (reflected_i < 0) {
					reflected_i = -reflected_i;  // Reflect across 0
				} else if (reflected_i >= size_z) {
					reflected_i = 2 * size_z - reflected_i - 2;  // Reflect across size_z-1
				}

				int pos = x * size_y * size_z + y * size_z + reflected_i;
				sum += input[pos];
				count++;
			}
		}
	} else {
		// Regular case (no reflection needed) - use optimized code
		int start, end;

		if (axis == 0) {  // X-axis
			start = x - delta;  // This is safe because we checked needs_reflection
			end = x + delta;

			// Use unrolled loop for better performance
			for (int i = start; i <= end; i += 4) {
				if (i <= end) {
					int pos = i * size_y * size_z + y * size_z + z;
					sum += input[pos];
					count++;
				}
				if (i+1 <= end) {
					int pos = (i+1) * size_y * size_z + y * size_z + z;
					sum += input[pos];
					count++;
				}
				if (i+2 <= end) {
					int pos = (i+2) * size_y * size_z + y * size_z + z;
					sum += input[pos];
					count++;
				}
				if (i+3 <= end) {
					int pos = (i+3) * size_y * size_z + y * size_z + z;
					sum += input[pos];
					count++;
				}
			}
		}
		else if (axis == 1) {  // Y-axis
			start = y - delta;
			end = y + delta;

			for (int i = start; i <= end; i++) {
				int pos = x * size_y * size_z + i * size_z + z;
				sum += input[pos];
				count++;
			}
		}
		else if (axis == 2) {  // Z-axis
			start = z - delta;
			end = z + delta;

			for (int i = start; i <= end; i++) {
				int pos = x * size_y * size_z + y * size_z + i;
				sum += input[pos];
				count++;
			}
		}
	}

	// Avoid division by zero
	if (count > 0) {
		output[idx] = sum / count;
	} else {
		output[idx] = 0.0f;
	}
}

extern "C" __global__
void box_1d_reflect(const float* input, float* output,
				   int size_x, int size_y, int size_z,
				   int delta, int axis)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int total_size = size_x * size_y * size_z;
	if (idx >= total_size) return;

	int x = idx / (size_y * size_z);
	int temp = idx % (size_y * size_z);
	int y = temp / size_z;
	int z = temp % size_z;

	float sum = 0.0f;
	int count = 0;

	if (axis == 0) {  // X-axis
		for (int i = x - delta; i <= x + delta; i++) {
			// Apply reflection for out-of-bounds indices
			int reflected_i = i;
			if (reflected_i < 0) {
				reflected_i = -reflected_i;  // Reflect across 0
			} else if (reflected_i >= size_x) {
				reflected_i = 2 * size_x - reflected_i - 2;  // Reflect across size_x-1
			}

			int pos = reflected_i * size_y * size_z + y * size_z + z;
			sum += input[pos];
			count++;
		}
	}
	else if (axis == 1) {  // Y-axis
		for (int i = y - delta; i <= y + delta; i++) {
			// Apply reflection for out-of-bounds indices
			int reflected_i = i;
			if (reflected_i < 0) {
				reflected_i = -reflected_i;  // Reflect across 0
			} else if (reflected_i >= size_y) {
				reflected_i = 2 * size_y - reflected_i - 2;  // Reflect across size_y-1
			}

			int pos = x * size_y * size_z + reflected_i * size_z + z;
			sum += input[pos];
			count++;
		}
	}
	else if (axis == 2) {  // Z-axis
		for (int i = z - delta; i <= z + delta; i++) {
			// Apply reflection for out-of-bounds indices
			int reflected_i = i;
			if (reflected_i < 0) {
				reflected_i = -reflected_i;  // Reflect across 0
			} else if (reflected_i >= size_z) {
				reflected_i = 2 * size_z - reflected_i - 2;  // Reflect across size_z-1
			}

			int pos = x * size_y * size_z + y * size_z + reflected_i;
			sum += input[pos];
			count++;
		}
	}

	// Avoid division by zero
	if (count > 0) {
		output[idx] = sum / count;
	} else {
		output[idx] = 0.0f;
	}
}

