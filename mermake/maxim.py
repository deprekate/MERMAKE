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
delta_fit_kernel = cp.RawKernel(kernel_code, "delta_fit")
delta_fit_cross_corr_kernel = cp.RawKernel(kernel_code, "delta_fit_cross_corr")


import itertools
def compute_crosscorr_score(image, raw, z_out, x_out, y_out, delta_fit, sigmaZ, sigmaXY):
	coords = cp.stack([z_out, x_out, y_out], axis=1)  # shape (N, 3)
	n_points = coords.shape[0]

	# Step 1: Spherical offsets within radius delta_fit
	offsets = []
	for dz, dx, dy in itertools.product(range(-delta_fit, delta_fit + 1), repeat=3):
		if dz*dz + dx*dx + dy*dy <= delta_fit*delta_fit:
			offsets.append((dz, dx, dy))
	offsets = cp.array(offsets, dtype=cp.int32)  # shape (P, 3)
	Xft = offsets.astype(cp.float32)
	P = Xft.shape[0]

	# Step 2: Build absolute coordinates
	neighborhood = coords[:, None, :] + offsets[None, :, :]  # shape (N, P, 3)

	def reflect_index(index, max_val):
		index = cp.where(index < 0, -index, index)  # reflect negative
		index = cp.where(index >= max_val, 2 * max_val - index - 2, index)  # reflect over bounds
		return index


	zi = reflect_index(neighborhood[..., 0], image.shape[0]).astype(cp.int32)
	xi = reflect_index(neighborhood[..., 1], image.shape[1]).astype(cp.int32)
	yi = reflect_index(neighborhood[..., 2], image.shape[2]).astype(cp.int32)


	# Step 3: Compute Gaussian weights
	sigma = cp.array([sigmaZ, sigmaXY, sigmaXY], dtype=cp.float32)[None, :]
	Xft_scaled = Xft / sigma
	norm_G = cp.exp(-cp.sum(Xft_scaled * Xft_scaled, axis=-1) / 2.0)  # shape (P,)
	norm_G = (norm_G - norm_G.mean()) / norm_G.std()

	# Step 4: Sample the image at all (zi, xi, yi)
	sample = image[zi, xi, yi]  # shape (N, P)

	# Step 5: Normalize sample rows and compute correlation
	sample_norm = (sample - sample.mean(axis=1, keepdims=True)) / sample.std(axis=1, keepdims=True)
	hn = cp.mean(sample_norm * norm_G[None, :], axis=1)  # shape (N,)

	# sample the raw image
	sample = raw[zi, xi, yi]  # shape (N, P)
	sample_norm = (sample - sample.mean(axis=1, keepdims=True)) / sample.std(axis=1, keepdims=True)
	a = cp.mean(sample_norm * norm_G[None, :], axis=1)  # shape (N,)

	return hn,a


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
	if not image.flags.c_contiguous:
		print('not contiguous')
		image = cp.ascontiguousarray(image)

	depth, height, width = image.shape
	max_points = depth * height * width

	# Allocate output arrays
	z_out = cp.zeros(max_points, dtype=cp.float32)
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
	#output[:,0] = z_out	
	#output[:,1] = x_out	
	#output[:,2] = y_out	
	# Create integer coordinate arrays once
	#zi, xi, yi = z_out.astype(int), x_out.astype(int), y_out.astype(int)

	# Use them for both indexing operations
	#output[:,7] = image[zi, xi, yi]
	#output[:,5] = raw[zi, xi, yi]

	# Adjust blocks for the number of points found
	#blocks = (num + threads - 1) // threads
	#delta_fit_kernel((blocks,), (threads,), (image.ravel(), z_out, x_out, y_out, output, num, depth, height, width, delta_fit))


	delta_fit_cross_corr_kernel((blocks,), (threads,), (image.ravel(), raw.ravel(), z_out, x_out, y_out, output, num, depth, height, width, delta_fit, sigmaZ, sigmaXY))
	del z_out, x_out, y_out 	
	cp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool
	cp._default_pinned_memory_pool.free_all_blocks()  # Free pinned memory pool
	cp.cuda.runtime.deviceSynchronize()  # Ensure all operations are completed
	#print('    ', output.shape)
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
	exit()
	start = time.time()
	old = get_local_maxfast_tensor(im,th_fit=0.97,im_raw=im,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
	end = time.time()
	print(f"time: {end - start:.6f} seconds")
	#tem = get_local_maxfast(im,th_fit=0.97,im_raw=im,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
	print('old.shape', old.shape)
	print(old)
