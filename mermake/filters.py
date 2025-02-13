#import numpy as np
import cupy as cp

def laplacian_3d(shape: tuple[int, int, int]) -> cp.ndarray:
	"""Define the 3D Laplacian in the spatial domain."""
	lap = cp.zeros(shape, dtype=cp.float32)

	z_c, y_c, x_c = shape[0] // 2, shape[1] // 2, shape[2] // 2

	lap[z_c, y_c, x_c] = 6
	lap[z_c - 1, y_c, x_c] = -1
	lap[z_c + 1, y_c, x_c] = -1
	lap[z_c, y_c - 1, x_c] = -1
	lap[z_c, y_c + 1, x_c] = -1
	lap[z_c, y_c, x_c - 1] = -1
	lap[z_c, y_c, x_c + 1] = -1

	return lap

def pad_3d(image: cp.ndarray, psf: cp.ndarray, pad: int | tuple[int, int, int]) -> tuple[cp.ndarray, cp.ndarray, tuple[int, int, int]]:
	"""
	Pad a 3D image and its PSF for deconvolution

	Parameters:
		image (cp.ndarray): Input 3D image.
		psf (cp.ndarray): 3D Point Spread Function.
		pad (int or tuple[int, int, int]): Padding in each dimension.

	Returns:
		tuple: (padded image, padded psf, padding tuple)
	"""
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
		image_pad = cp.pad(image, pad_width, mode='reflect')
		# Constant padding (zero) for PSF
		psf_pad = cp.pad(psf, pad_width, mode='constant', constant_values=0)
	else:
		image_pad = image
		psf_pad = psf
	return image_pad, psf_pad, padding

def wiener_deconvolve(image, psf, beta=0.001, pad=0):
	"""Perform 3D Wiener deconvolution with Laplacian regularization using CuPy (GPU-accelerated)."""
	# Normalize PSF
	psf /= cp.sum(psf)
	# Pad image and PSF
	#print(image.dtype)
	image_pad, psf_pad, padding = pad_3d(image, psf, pad)
	#print(image_pad.dtype)
	# Convert to frequency domain
	image_fft = cp.fft.fftn(image_pad)
	# Roll the PSF (shift it to the center)
	psf_roll = cp.roll(psf_pad, shift=(-psf_pad.shape[0] // 2, -psf_pad.shape[1] // 2, -psf_pad.shape[2] // 2), axis=(0, 1, 2))
	psf_fft = cp.fft.fftn(psf_roll)
	laplacian_fft = cp.fft.fftn(laplacian_3d(image_pad.shape))
	# Wiener filtering with Laplacian regularization
	#den = psf_fft * cp.conj(psf_fft) + beta * laplacian_fft * cp.conj(laplacian_fft)
	# faster power spectrum calculation
	den = cp.abs(psf_fft) ** 2 + beta * cp.abs(laplacian_fft) ** 2
	deconv_fft = image_fft * cp.conj(psf_fft) / den
	# Convert back to spatial domain and unpad
	deconv_image = cp.real(cp.fft.ifftn(deconv_fft))
	#deconv_image = (deconv_image - cp.min(deconv_image)) / (cp.max(deconv_image) - cp.min(deconv_image))
	if image_pad.shape != image.shape:
		return unpad_3d(deconv_image, padding)
	return deconv_image

def unpad_3d(image: cp.ndarray, padding: tuple[int, int, int]) -> cp.ndarray:
	"""
	Remove the padding of a 3D image.

	Parameters:
		image (cp.ndarray): 3D image to un-pad.
		padding (tuple[int, int, int]): Padding in each dimension.

	Returns:
		cp.ndarray: Unpadded image.
	"""
	return image[padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2]]

def center_psf(psf, target_shape):
	"""
	Inserts `psf` into a zero-padded CuPy array of `target_shape`,
	cropping if necessary.

	Parameters:
	- psf (cp.ndarray): The PSF array to insert.
	- target_shape (tuple): The desired output shape.

	Returns:
	- cp.ndarray: The centered PSF inside a zero-padded/cropped array.
	"""
	psff = cp.zeros(target_shape, dtype=cp.float32)

	target_shape_cp = cp.array(target_shape, dtype=cp.int32)
	psf_shape_cp = cp.array(psf.shape, dtype=cp.int32)

	# Compute start & end indices for both the source and target
	start_psff = ((target_shape_cp - psf_shape_cp) // 2).astype(cp.int32)
	end_psff = start_psff + cp.minimum(target_shape_cp, psf_shape_cp)

	start_psf = ((psf_shape_cp - target_shape_cp) // 2).astype(cp.int32)
	end_psf = start_psf + cp.minimum(target_shape_cp, psf_shape_cp)

	# Assign using slices
	slices_psff = tuple(slice(start, end) for start, end in zip(start_psff, end_psff))
	slices_psf = tuple(slice(start, end) for start, end in zip(start_psf, end_psf))

	psff[slices_psff] = psf[slices_psf]
	return psff

