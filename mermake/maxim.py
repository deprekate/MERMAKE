import cupy as cp


with open("maxima.cu", "r") as f:
	kernel_code = f.read()

# Define the kernels separately
local_maxima_kernel = cp.RawKernel(kernel_code, "local_maxima")
delta_fit_kernel = cp.RawKernel(kernel_code, "delta_fit")

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


def find_local_maxima(image, threshold, delta, delta_fit, raw=None):
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

	# Call the kernel
	local_maxima_kernel((blocks,), (threads,), 
					(image.ravel(), cp.float32(threshold), delta, delta_fit,
					 z_out, x_out, y_out, count,
					 depth, height, width, max_points))
	cp.cuda.Device().synchronize()
	num = int(count.get()[0])
	z_out = z_out[:num]
	x_out = x_out[:num]
	y_out = y_out[:num]

	count = cp.zeros(1, dtype=cp.uint32)
	output = cp.zeros((num, 8), dtype=cp.float32)
	
	indices = z_out[:num].astype(int) * (height * width) + x_out[:num].astype(int) * width + y_out[:num].astype(int)
	output[:,7] = image.ravel()[indices]
	output[:,5] = raw.ravel()[indices]
	print(cp.stack([z_out, x_out, y_out], axis=1))

	delta_fit_kernel((blocks,), (threads,), (image.ravel(), z_out, x_out, y_out, output, num, depth, height, width, delta_fit))

	'''
	hn,a = compute_crosscorr_score(image, raw, z_out, x_out, y_out, delta_fit=delta_fit, sigmaZ=1.0, sigmaXY=1.5)
	output[:,4] = hn
	output[:,6] = a
	'''
	return output
	#num = int(count.get()[0])
	#coords = cp.stack([z_out[:num], x_out[:num], y_out[:num]], axis=1)
	#return coords


if __name__ == "__main__":
	import numpy as np
	np.set_printoptions(suppress=True, linewidth=100)
	import torch
	from ioMicro import get_local_maxfast_tensor, get_local_maxfast
	# Example Usage
	cim = cp.random.rand(40, 3000, 3000).astype(cp.float32)
	im = cp.asnumpy(cim)
	print(cim)
	local = find_local_maxima(cim, 0.97, 1, 3, raw=cim)
	print(local.shape)
	print(local)
	tem = get_local_maxfast_tensor(im,th_fit=0.97,im_raw=im,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
	#tem = get_local_maxfast(im,th_fit=0.97,im_raw=im,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5)
	print(tem.shape)
	print(tem)
