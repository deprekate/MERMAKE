import numpy as np
import cupy as cp

def laplacian_3d_like(image):
	"""Define the 3D Laplacian in the spatial domain."""
	xp = cp.get_array_module(image)
	shape = image.shape
	lap = xp.zeros(shape, dtype=xp.float32)

	z_c, y_c, x_c = shape[0] // 2, shape[1] // 2, shape[2] // 2

	lap[z_c, y_c, x_c] = 6
	lap[z_c - 1, y_c, x_c] = -1
	lap[z_c + 1, y_c, x_c] = -1
	lap[z_c, y_c - 1, x_c] = -1
	lap[z_c, y_c + 1, x_c] = -1
	lap[z_c, y_c, x_c - 1] = -1
	lap[z_c, y_c, x_c - 1] = -1 # ERROR: SHOULD BE +1

	return lap

def pad_3d(image, psf, pad):
	"""
	Pad a 3D image and its PSF for deconvolution

	Parameters:
		image (cp.ndarray): Input 3D image.
		psf (cp.ndarray): 3D Point Spread Function.
		pad (int or tuple[int, int, int]): Padding in each dimension.

	Returns:
		tuple: (padded image, padded psf, padding tuple)
	"""
	xp = cp.get_array_module(image)
	padding = pad
	if isinstance(pad, tuple) and len(pad) != image.ndim:
		raise ValueError("Padding must be the same dimension as image")
	if isinstance(pad, int):
		if pad == 0:
			return image, psf, (0, 0, 0)
		padding = (pad, pad, pad)
	if padding[0] > 0 and padding[1] > 0 and padding[2] > 0:
		# Convert padding format for cupy.pad
		pad_width = ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]))
		# Reflection padding for image (to match torch.nn.ReflectionPad3d)
		image_pad = xp.pad(image, pad_width, mode='reflect')
		# Constant padding (zero) for PSF
		psf_pad = xp.pad(psf, pad_width, mode='constant', constant_values=0)
	else:
		image_pad = image
		psf_pad = psf
	return image_pad, psf_pad, padding

def wiener_deconvolve(image, psf, beta=0.001, pad=0):
	"""Perform 3D Wiener deconvolution with Laplacian regularization using CuPy (GPU-accelerated)."""
	xp = cp.get_array_module(image)
	# Normalize PSF
	psf /= xp.sum(psf)
	# Pad image and PSF
	image_pad, psf_pad, padding = pad_3d(image, psf, pad)
	# Convert to frequency domain
	image_fft = xp.fft.fftn(image_pad)
	# Roll the PSF (shift it to the center)
	psf_pad = xp.roll(psf_pad, shift=(-psf_pad.shape[0] // 2, -psf_pad.shape[1] // 2, -psf_pad.shape[2] // 2), axis=(0, 1, 2))
	psf_fft = xp.fft.fftn(psf_pad)
	laplacian_fft = xp.fft.fftn(laplacian_3d_like(image_pad))
	# Wiener filtering with Laplacian regularization
	# THIS IS ALL POORLY OPTIMIZED CODE
	#den = psf_fft * xp.conj(psf_fft) + beta * laplacian_fft * xp.conj(laplacian_fft)
	#deconv_fft = image_fft * xp.conj(psf_fft) / den
	# Convert back to spatial domain and unpad
	#image_pad[:] = xp.real(xp.fft.ifftn(deconv_fft))
	
	# THIS IS ALL INPLACE TESTING STUFF
	'''
	psf_conj = xp.conj(psf_fft)
	xp.multiply(image_fft, psf_conj, out=image_fft)
	xp.multiply(psf_fft, psf_conj, out=psf_fft)
	xp.multiply(laplacian_fft, laplacian_fft.conj(), out=laplacian_fft)
	xp.multiply(laplacian_fft, beta, out=laplacian_fft)
	xp.add(psf_fft, laplacian_fft, out=psf_fft)
	xp.true_divide(image_fft, psf_fft, out=image_fft)
	image_fft[:] = xp.fft.ifftn(image_fft)
	image_pad[:] = xp.real(image_fft)
	'''
	psf_conj = xp.conj(psf_fft)
	xp.multiply(psf_fft, psf_conj, out=psf_fft)
	xp.multiply(laplacian_fft, laplacian_fft.conj(), out=laplacian_fft)
	xp.multiply(laplacian_fft, beta, out=laplacian_fft)
	xp.add(psf_fft, laplacian_fft, out=psf_fft)
	xp.true_divide(psf_conj, psf_fft, out=psf_fft)

	xp.multiply(image_fft, psf_fft, out=image_fft)
	image_fft[:] = xp.fft.ifftn(image_fft)
	image_pad[:] = xp.real(image_fft)
	
	if image_pad.shape != image.shape:
		return unpad_3d(image_pad, padding)
	return image_pad

def unpad_3d(image, padding):
	"""
	Remove the padding of a 3D image.

	Parameters:
		image (ndarray): 3D image to un-pad.
		padding (tuple[int, int, int]): Padding in each dimension.

	Returns:
		ndarray: Unpadded image.
	"""
	return image[padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2]]

def center_psf(psf, target_shape):
	"""
	Inserts `psf` into a zero-padded CuPy array of `target_shape`,
	cropping if necessary.

	Parameters:
	- psf (ndarray): The PSF array to insert.
	- target_shape (tuple): The desired output shape.

	Returns:
	- cp.ndarray: The centered PSF inside a zero-padded/cropped array.
	"""
	xp = cp.get_array_module(psf)
	psff = xp.zeros(target_shape, dtype=xp.float32)

	# Compute start & end indices for both the source and target
	start_psff = ((target_shape - psf_shape) // 2).astype(xp.int32)
	end_psff = start_psff + xp.minimum(target_shape, psf_shape)

	start_psf = ((psf_shape - target_shape) // 2).astype(xp.int32)
	end_psf = start_psf + xp.minimum(target_shape, psf_shape)

	# Assign using slices
	slices_psff = tuple(slice(start, end) for start, end in zip(start_psff, end_psff))
	slices_psf = tuple(slice(start, end) for start, end in zip(start_psf, end_psf))

	psff[slices_psff] = psf[slices_psf]
	return psff

