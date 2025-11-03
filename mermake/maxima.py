import os
import cupy as cp
import time

# Get path relative to this file's location
this_dir = os.path.dirname(__file__)
cu_path = os.path.join(this_dir, "maxima.cu")
with open(cu_path, "r") as f:
	kernel_code = f.read()

# Define the kernels separately
local_maxima_count_kernel = cp.RawKernel(kernel_code, "local_maxima_count")
local_maxima_kernel = cp.RawKernel(kernel_code, "local_maxima")
#delta_fit_kernel = cp.RawKernel(kernel_code, "delta_fit")
delta_fit_cross_corr_kernel = cp.RawKernel(kernel_code, "delta_fit_cross_corr")

def find_local_maxima(image, threshold=None, delta=1, delta_fit=0, raw=None, sigmaZ=1, sigmaXY=1.5, **kwargs):
	# Ensure the image is in C-contiguous order for the kernel
	#if not image.flags.c_contiguous:
	#	print('not contiguous')
	#	image = cp.ascontiguousarray(image)
	
	# Ensure raw is also contiguous
	#if raw is not None and not raw.flags.c_contiguous:
	#	raw = cp.ascontiguousarray(raw)

	if delta > 5:
		raise TypeError("Delta must be an less than or equal to 5 due to MAX_KERNEL_POINTS")
	
	depth, height, width = image.shape
	total_voxels = depth * height * width
	
	# Convert parameters to appropriate types
	threshold = cp.float32(threshold)
	sigmaZ = cp.float32(sigmaZ)
	sigmaXY = cp.float32(sigmaXY)
	
	count = cp.zeros(1, dtype=cp.uint32)
	threads = 256
	blocks = (total_voxels + threads - 1) // threads
	
	# 1st pass: Count only to help limit GPU RAM
	local_maxima_count_kernel((blocks,), (threads,),
		(image.ravel(), threshold, delta, count, depth, height, width, total_voxels))
	# Use stream.synchronize() instead of global synchronize for better performance
	stream = cp.cuda.get_current_stream()
	stream.synchronize()
	#cp.cuda.Device().synchronize()
	
	num = int(count.get()[0])
	
	if num == 0:
		return cp.zeros((0, 8), dtype=cp.float32)
	
	# Allocate output arrays
	z_out = cp.zeros(total_voxels, dtype=cp.uint16)
	x_out = cp.zeros_like(z_out)
	y_out = cp.zeros_like(z_out)
	count[:] = 0  # reset counter
	
	# Call the kernel to find local maxima and store coordinates
	local_maxima_kernel((blocks,), (threads,), 
		(image.ravel(), threshold, delta, delta_fit,
		 z_out, x_out, y_out, count, depth, height, width, num))
	#cp.cuda.Device().synchronize()
	stream.synchronize()
	
	# Get the actual number found
	num_found = int(count.get()[0])
	
	if num_found == 0:
		return cp.zeros((0, 8), dtype=cp.float32)
	
	# Trim arrays
	z_out = z_out[:num_found]
	x_out = x_out[:num_found]
	y_out = y_out[:num_found]
	
	# Allocate output array for refined coordinates
	output = cp.zeros((num_found, 8), dtype=cp.float32)

	# always cast the uint16 raw to float32
	if raw.dtype != cp.float32:
		raw = raw.astype(cp.float32)

	# Call delta fit kernel
	blocks = (num_found + threads - 1) // threads
	delta_fit_cross_corr_kernel((blocks,), (threads,), 
		(image.ravel(), raw.ravel(), z_out, x_out, y_out, output, 
		 num_found, depth, height, width, delta_fit, sigmaZ, sigmaXY))
	
	#cp.cuda.runtime.deviceSynchronize()
	stream.synchronize()
	# Clean up
	del z_out, x_out, y_out
	cp._default_memory_pool.free_all_blocks()
	
	return output

if __name__ == "__main__":
	import numpy as np
	np.set_printoptions(suppress=True, linewidth=100)
	# Example Usage
	#cim = cp.random.rand(40, 300, 300).astype(cp.float32)
	cim = cp.random.rand(4, 4, 4).astype(cp.float32)
	print(cim)
	local = find_local_maxima(cim, 0.97, 1, 3, raw=cim)
	print('local.shape',local.shape, flush=True)
	print(local)
