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
		Input 2D or 3D array (will not be modified).
	size : int or tuple of int
		Blur size per axis. If int, the same size is used for all axes.
	output : cupy.ndarray, optional
		Output array. If None, a new one is created.

	Returns
	-------
	cupy.ndarray
		Blurred array.
	"""

	# the kernel expects float32
	if image.dtype != cp.float32:
		image = image.astype(cp.float32)

	# Input validation
	if image.ndim not in (2, 3):
		raise ValueError("Only 2D or 3D arrays are supported")

	# Handle size parameter
	if isinstance(size, int):
		sizes = (size,) * image.ndim
	elif isinstance(size, (tuple, list)):
		if len(size) != image.ndim:
			raise ValueError(f"Size tuple must have {image.ndim} elements for {image.ndim}D input")
		sizes = size
	else:
		raise TypeError("size must be an int, tuple, or list of ints")

	# Create output if necessary
	if output is None:
		output = cp.empty_like(image)

	# Create temporary buffer
	temp = cp.empty_like(image)

	# First blur: image -> temp
	box_1d(image, sizes[0], axis=0, output=temp)
	
	if image.ndim == 2:
		# Second blur: temp -> output (for 2D)
		box_1d(temp, sizes[1], axis=1, output=output)
	else:  # 3D
		# Second blur: temp -> output
		box_1d(temp, sizes[1], axis=1, output=output)
		# For 3D, need a third pass
		box_1d(output, sizes[2], axis=2, output=temp)
		# Copy result back to output
		cp.copyto(output, temp)

	return output


def box_1d(image, size, axis=0, output=None):
	if image.dtype != cp.float32:
		image = image.astype(cp.float32)
	if output is None:
		output = cp.empty_like(image)
	delta = size // 2
	
	if image.ndim == 2:
		# For 2D arrays, assume XY format
		size_x, size_y = image.shape
		size_z = 1  # Dummy dimension
	elif image.ndim == 3:
		# For 3D arrays, use ZXY format
		size_z, size_x, size_y = image.shape
	else:
		raise ValueError("Only 2D or 3D arrays are supported")
	
	threads_per_block = 256
	blocks = (size_z * size_x * size_y + threads_per_block - 1) // threads_per_block
	box_1d_kernel((blocks,), (threads_per_block,),
				  (image, output, size_z, size_x, size_y, delta, axis))
	return output



