import cupy as cp
from math import prod

# Define CUDA kernel that handles everything: thresholding and local maxima detection
local_maxima_kernel = cp.RawKernel(r'''
extern "C" __global__
void find_local_maxima(const float* image, float threshold, 
					  int* z_out, int* x_out, int* y_out, unsigned int* count,
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
	
	float center_value = image[idx];
	
	// Check if above threshold
	if (center_value <= threshold) {
		return;
	}
	
	// Check if it's a local maximum
	bool is_max = true;
	int delta = 1; // Radius of spherical mask
	
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
				
				// Boundary check
				if (nz >= 0 && nz < depth && nx >= 0 && nx < height && ny >= 0 && ny < width) {
					float neighbor_value = image[nz * height * width + nx * width + ny];
					if (center_value < neighbor_value) {
					    is_max = false;
					    break;
					}
				}
			}
			if (!is_max) break;
		}
		if (!is_max) break;
	}
	
	// If it's a local maximum, add to output
	if (is_max) {
		unsigned int pos = atomicAdd(count, 1);
		if (pos < max_points) {
			z_out[pos] = z;
			x_out[pos] = x;
			y_out[pos] = y;
		}
	}
}
''', 'find_local_maxima')

def find_local_maxima(image, threshold):
	"""
	Find local maxima in a 3D image directly on GPU.
	
	Args:
		image: 3D CuPy array
		threshold: Minimum value for considering a local maximum
		max_points: Maximum number of local maxima to return
	
	Returns:
		Tuple of (z, x, y) coordinates for local maxima
	"""
	# Ensure the image is in C-contiguous order for the kernel
	if not image.flags.c_contiguous:
		image = cp.ascontiguousarray(image)

	max_points = prod(image.shape) + 1
	
	# Allocate output arrays
	#z_out = cp.zeros(max_points, dtype=cp.int32)
	#x_out = cp.zeros(max_points, dtype=cp.int32)
	#y_out = cp.zeros(max_points, dtype=cp.int32)
	out = cp.zeros([max_points, 3], dtype=cp.int32)
	count = cp.zeros(1, dtype=cp.uint32)
	
	# Set up kernel parameters
	threads_per_block = 256
	blocks_per_grid = (image.size + threads_per_block - 1) // threads_per_block
	
	# Call the kernel
	local_maxima_kernel((blocks_per_grid,), (threads_per_block,), 
					  (image.ravel(), threshold, 
						  out[:,0], out[:,1], out[:,2], count,
					   image.shape[0], image.shape[1], image.shape[2], max_points))
	
	# Get the actual count
	actual_count = int(count.item())
	
	# Check if we hit the max_points limit
	if actual_count >= max_points:
		print(f"Warning: Found more local maxima ({actual_count}) than max_points ({max_points})")
	
	# Return only the valid parts of the output arrays
	#return z_out[:actual_count], x_out[:actual_count], y_out[:actual_count]
	mask = out.all(axis=1)
	return out[mask]

if __name__ == "__main__":
	# Example Usage
	A = cp.random.rand(10, 3000, 3000)
	local = find_local_maxima(A, 0.9)
	print(local)
	print(local.shape)
