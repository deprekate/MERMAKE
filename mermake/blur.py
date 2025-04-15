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


def box(image, delta, output=None):
	"""
	Apply separable 3D box blur using only `image` and `output` buffers.
	"""
	if image.ndim != 3:
		raise ValueError("Only 3D arrays are supported")
	
	if output is None:
		output = cp.empty_like(image)

	inp, out = image, output

	for axis in [0, 1, 2]:
		box_1d(inp, delta, axis=axis, output=out)
		inp, out = out, inp  # Swap roles

	# After the last swap, `inp` has final result
	if inp is not output:
		output[...] = inp  # Copy back into `output`

	return output



def box_1d(image, size, axis=0, output=None):
	if image.dtype != cp.float32:
		image = image.astype(cp.float32)

	if output is None:
		output = cp.empty_like(image)

	delta = size // 2

	size_x, size_y, size_z = image.shape
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

