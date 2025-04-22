import os
import cupy as cp
import time

# Get path relative to this file's location
this_dir = os.path.dirname(__file__)
cu_path = os.path.join(this_dir, "maxima.cu")
with open(cu_path, "r") as f:
	kernel_code = f.read()

# Define the kernels separately
local_maxima_kernel = cp.RawKernel(kernel_code, "local_maxima")
#delta_fit_kernel = cp.RawKernel(kernel_code, "delta_fit")
delta_fit_cross_corr_kernel = cp.RawKernel(kernel_code, "delta_fit_cross_corr")


def find_local_maxima(image, threshold, delta, delta_fit, raw=None, sigmaZ=1, sigmaXY=1.5 ):
	"""
	Find and refine local maxima in a 3D image directly on GPU, including delta fitting.
	
	Args:
		image: 3D CuPy array
		threshold: Minimum value for local maxima detection
		delta_fit: Size of the fitting neighborhood
	
	Returns:
		Tuple of (z, x, y) coordinates for refined local maxima
	"""
	# Ensure the image is in C-contiguous order for the kernel
	#if not image.flags.c_contiguous:
	#	print('not contiguous')
	#	image = cp.ascontiguousarray(image)

	depth, height, width = image.shape
	max_points = depth * height * width

	# Allocate output arrays
	z_out = cp.zeros(max_points, dtype=cp.uint16)
	x_out = cp.zeros_like(z_out)
	y_out = cp.zeros_like(z_out)

	count = cp.zeros(1, dtype=cp.uint32)
	# Set up kernel parameters
	threads = 256
	blocks = (max_points + threads - 1) // threads
	
	threshold = cp.float32(threshold)
	sigmaZ = cp.float32(sigmaZ)
	sigmaXY = cp.float32(sigmaXY)
	# Call the kernel
	local_maxima_kernel((blocks,), (threads,), 
					(image.ravel(), threshold, delta, delta_fit,
					 z_out, x_out, y_out, count,
					 depth, height, width, max_points))
	cp.cuda.Device().synchronize()
	num = int(count.get()[0])
	if num == 0:
		# Return empty result if no local maxima found
		return cp.zeros((0, 8), dtype=cp.float32)

	z_out = z_out[:num]
	x_out = x_out[:num]
	y_out = y_out[:num]

	count = cp.zeros(1, dtype=cp.uint32)
	output = cp.zeros((num, 8), dtype=cp.float32)

	delta_fit_cross_corr_kernel((blocks,), (threads,), (image.ravel(), raw.ravel(), z_out, x_out, y_out, output, num, depth, height, width, delta_fit, sigmaZ, sigmaXY))
	del z_out, x_out, y_out 	
	cp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool
	cp._default_pinned_memory_pool.free_all_blocks()  # Free pinned memory pool
	cp.cuda.runtime.deviceSynchronize()  # Ensure all operations are completed
	
	return output


if __name__ == "__main__":
	import numpy as np
	np.set_printoptions(suppress=True, linewidth=100)
	import torch
	from ioMicro import get_local_maxfast_tensor, get_local_maxfast
	# Example Usage
	cim = cp.random.rand(40, 300, 300).astype(cp.float32)
	im = cp.asnumpy(cim)
	#print(cim)
	start = time.time()
	local = find_local_maxima(cim, 0.97, 1, 3, raw=cim)
	end = time.time()
	print(f"time: {end - start:.6f} seconds")
	print('local.shape',local.shape, flush=True)
	print(local)
	print(cp.min(local, axis=0))
	print(cp.max(local, axis=0))
