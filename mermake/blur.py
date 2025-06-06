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

def box(image, size, axes=None, output=None, temp=None):
	"""
	Apply separable box blur on 2D or 3D cupy arrays.

	Parameters
	----------
	image : cupy.ndarray
		Input 2D or 3D array (will not be modified).
	size : int or tuple of int
		Blur size per axis. If int, the same size is used for all specified axes.
	axes : tuple of int, optional
		Axes to blur. If None, blur all axes.
		For 2D: (0, 1) means blur both dimensions
		For 3D: (0, 1, 2) means blur all dimensions
		Examples: (0,) for axis 0 only, (1, 2) for axes 1 and 2
	temp : cupy.ndarray, optional
		Temporary buffer for multi-axis blurs. If None, one is created.
		Must be same shape and dtype as input.
	temp : cupy.ndarray, optional
		Temporary buffer. If None, one is created.
	output : cupy.ndarray, optional
		Output array. If None, a new one is created.

	Returns
	-------
	cupy.ndarray
		Blurred array.
	"""
	# The kernel expects float32
	if image.dtype != cp.float32:
		image = image.astype(cp.float32)

	# Input validation
	if image.ndim not in (2, 3):
		raise ValueError("Only 2D or 3D arrays are supported")

	# Handle axes parameter
	if axes is None:
		axes = tuple(range(image.ndim))  # All axes
	elif isinstance(axes, int):
		axes = (axes,)  # Single axis
	elif not isinstance(axes, (tuple, list)):
		raise TypeError("axes must be None, int, tuple, or list")

	# Validate and normalize axes (handle negative indices)
	normalized_axes = []
	for axis in axes:
		if axis < 0:
			axis = image.ndim + axis
		if axis < 0 or axis >= image.ndim:
			raise ValueError(f"axis {axis} is out of bounds for array of dimension {image.ndim}")
		normalized_axes.append(axis)

	# Remove duplicates and sort
	axes = tuple(sorted(set(normalized_axes)))

	if len(axes) == 0:
		# No axes to blur, just copy
		if output is None:
			return cp.copy(image)
		else:
			cp.copyto(output, image)
			return output

	# Handle size parameter
	if isinstance(size, int):
		sizes = {axis: size for axis in axes}
	elif isinstance(size, (tuple, list)):
		if len(size) != len(axes):
			raise ValueError(f"Size must have {len(axes)} elements for {len(axes)} axes")
		sizes = dict(zip(axes, size))
	else:
		raise TypeError("size must be an int, tuple, or list of ints")

	# Create output if necessary
	if output is None:
		output = cp.empty_like(image)

	# If only one axis, do single blur
	if len(axes) == 1:
		axis = axes[0]
		box_1d(image, sizes[axis], axis=axis, output=output)
		return output

	# For multiple axes, we can't avoid at least one temporary buffer
	# because we can't do in-place blurring (causes race conditions)
	# But we'll minimize to just one temp buffer that gets reused

	# For multiple axes, we need exactly one temporary buffer
	# (can't avoid this due to CUDA kernel race condition constraints)

	if temp is None:
		temp = cp.empty_like(image)
	else:
		# Validate provided temp buffer
		if temp.shape != image.shape or temp.dtype != image.dtype:
			raise ValueError("temp buffer must have same shape and dtype as input")

	# First blur: image -> output
	box_1d(image, sizes[axes[0]], axis=axes[0], output=output)

	# Remaining blurs alternate between output and temp
	for i in range(1, len(axes)):
		axis = axes[i]
		if i % 2 == 1:  # Odd iterations: output -> temp
			box_1d(output, sizes[axis], axis=axis, output=temp)
		else:  # Even iterations: temp -> output
			box_1d(temp, sizes[axis], axis=axis, output=output)

	# If we ended on temp, copy back to output
	if len(axes) % 2 == 0:  # Even number of axes means we ended on temp
		cp.copyto(output, temp)

	return output


def box_2d(image, size, axes=(-2, -1), output=None):
	"""
	Convenience function for 2D blur on specified axes.

	Parameters
	----------
	image : cupy.ndarray
		Input array (2D or 3D).
	size : int or tuple of 2 ints
		Blur size for the two axes.
	axes : tuple of 2 ints, default (-2, -1)
		Which two axes to blur.
	output : cupy.ndarray, optional
		Output array. If None, a new one is created.

	Returns
	-------
	cupy.ndarray
		Blurred array.
	"""
	if len(axes) != 2:
		raise ValueError("axes must contain exactly 2 elements for 2D blur")

	return box(image, size, axes=axes, output=output)

def box_1d(image, size, axis=0, output=None):
	if image.dtype != cp.float32:
		image = image.astype(cp.float32)
	if output is None:
		output = cp.empty_like(image)

	# Prevent in-place operations that cause race conditions
	assert output is not image, "In-place operation not supported - input and output must be different"

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



