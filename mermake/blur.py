import os
import cupy as cp

# Get path relative to this file's location
this_dir = os.path.dirname(__file__)
cu_path = os.path.join(this_dir, "blur.cu")
with open(cu_path, "r") as f:
	kernel_code = f.read()

# Define the kernels
box_1d_kernel = cp.RawKernel(kernel_code, "box_1d")
#optimized_box_1d_kernel = cp.RawKernel(kernel_code, "optimized_box_1d")
#box_plane_kernel = cp.RawKernel(kernel_code, "box_plane")

import cupy as cp

def box(image, size, output=None):
	"""
	Apply separable box blur on 2D or 3D cupy arrays.

	Parameters
	----------
	image : cupy.ndarray
		Input 2D or 3D array.
	size : int or tuple of int
		Blur size per axis. If int, the same size is used for all axes.
	output : cupy.ndarray, optional
		Output array. If None, a new one is created.
	"""
	if image.ndim not in (2, 3):
		raise ValueError("Only 2D or 3D arrays are supported")

	if isinstance(size, int):
		size = (size,) * image.ndim
	elif isinstance(size, tuple):
		if len(size) != image.ndim:
			raise ValueError(f"Size tuple must have {image.ndim} elements for {image.ndim}D input")
	else:
		raise TypeError("size must be an int or a tuple of ints")

	if output is None:
		output = cp.empty_like(image)

	inp, out = image, output

	for axis, sz in enumerate(size):
		delta = sz // 2
		box_1d(inp, delta, axis=axis, output=out)
		inp, out = out, inp  # Swap roles

	if inp is not output:
		output[...] = inp

	return output

def box_1d(image, size, axis=0, output=None):
	if image.dtype != cp.float32:
		image = image.astype(cp.float32)

	if output is None:
		output = cp.empty_like(image)

	delta = size // 2

	if image.ndim == 2:
		size_x, size_y = image.shape
		size_z = 1  # Dummy dimension
	elif image.ndim == 3:
		size_x, size_y, size_z = image.shape
	else:
		raise ValueError("Only 2D or 3D arrays are supported")

	threads_per_block = 256
	blocks = (size_x * size_y * size_z + threads_per_block - 1) // threads_per_block

	box_1d_kernel((blocks,), (threads_per_block,),
					(image, output, size_x, size_y, size_z, delta, axis))

	return output

def optimized_box_1d(image, size, axis=0, output=None):
	if image.dtype != cp.float32:
		image = image.astype(cp.float32)
	if output is None:
		output = cp.empty_like(image)

	delta = size // 2
	size_x, size_y, size_z = image.shape

	threads_per_block = 256
	blocks = (size_x * size_y * size_z + threads_per_block - 1) // threads_per_block

	optimized_box_1d_kernel((blocks,), (threads_per_block,),
						  (image, output, size_x, size_y, size_z, delta, axis))

	return output

