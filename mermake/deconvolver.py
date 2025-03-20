import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change "1" to the desired GPU ID

import time
import gc
import cupy as cp
import numpy as np


def norm_slices(image, ksize=50):
	xp = cp.get_array_module(image)
	if xp == np:
		from scipy.ndimage import convolve
	else:
		from cupyx.scipy.ndimage import convolve

	image = image.astype(xp.float32)  # Ensure correct type
	kernel = xp.ones((ksize, ksize), dtype=xp.float32) / (ksize * ksize)

	# Apply blur to each slice in parallel without looping in Python
	blurred = convolve(image, kernel[None, :, :], mode="constant")

	return image - blurred  # Equivalent to original list comprehension

def laplacian_3d(shape):
	"""Create a 3D Laplacian kernel for a given shape."""
	lap = cp.zeros(shape, dtype=cp.float16)
	z_c, y_c, x_c = shape[0] // 2, shape[1] // 2, shape[2] // 2
	lap[z_c, y_c, x_c] = 6
	lap[z_c - 1, y_c, x_c] = -1
	lap[z_c + 1, y_c, x_c] = -1
	lap[z_c, y_c - 1, x_c] = -1
	lap[z_c, y_c + 1, x_c] = -1
	lap[z_c, y_c, x_c - 1] = -1
	lap[z_c, y_c, x_c - 1] = -1  # Bug fix (previously had two -1s at the same position)
	return lap

def batch_laplacian_fft(batch_size, shape):
	"""Compute the 3D Laplacian in frequency space and prepare for batch processing."""
	lap = laplacian_3d(shape)  # Create a single 3D Laplacian
	z_c, y_c, x_c = shape[0] // 2, shape[1] // 2, shape[2] // 2
	#lap_fft = cp.fft.fftn(cp.fft.ifftshift(lap))  # Shift Laplacian to center before FFT
	lap_fft = cp.fft.fftn(lap)
	return lap_fft[None, ...]  # Add batch dimension without copying memory



class Deconvolver:
	def __init__(self, psfs, zpad, tile_size = None, overlap = 89, beta = 0.001, xp = cp):

		psf_stack = np.stack(list(map(center_psf, list(psfs.values()))))
		psf_stack = np.pad(psf_stack, ((0, 0), (zpad, zpad), (overlap, overlap), (overlap, overlap)), mode='constant')
		shift = -np.array(psf_stack.shape[1:]) // 2
		psf_stack[:] = np.roll(psf_stack, shift=shift, axis=(1,2,3))

		psf_fft = xp.empty_like(psf_stack, dtype=xp.complex64)
		# have to do this by zslice due to gpu ram ~ 48GB
		for z in range(len(psf_fft)):
			psf_fft[z] = xp.fft.fftn(cp.asarray(psf_stack[z]))
			psf_conj = xp.conj(psf_fft[z])
			psf_fft[z] *= psf_conj
			laplacian_fft = xp.fft.fftn(laplacian_3d(psf_conj.shape))
			laplacian_fft *= laplacian_fft.conj()
			laplacian_fft *= beta
			psf_fft[z] += laplacian_fft
			psf_fft[z] = psf_conj / psf_fft[z]
		del laplacian_fft, psf_conj, psf_stack
		self.psf_fft = psf_fft
		gc.collect()  # Standard Python garbage collection
		try:
			cp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool
			cp._default_pinned_memory_pool.free_all_blocks()  # Free pinned memory pool
			cp.cuda.runtime.deviceSynchronize()  # Ensure all operations are completed
		except Exception as e:
			print(f"Warning: Failed to free GPU memory: {e}")

		if tile_size:
			# Preallocate tile arrays
			self.tile = xp.empty((40 + 2 * zpad, 300 + 2*overlap, 300 + 2*overlap), dtype=xp.float16)
			self.tile_fft = xp.empty_like(self.tile, dtype=xp.complex64)
			self.tile_res = xp.empty((40 , 300 + 2*overlap, 300 + 2*overlap), dtype=xp.float16)
			self.overlap = overlap
			self.zpad = zpad
		self.tile_size = tile_size
		self.xp = xp

	def tile_wise(self, image):
		zpad = self.zpad
		tile = self.tile
		tile_fft = self.tile_fft
		tile_res = self.tile_res
		psf_fft = self.psf_fft
		tiles = self.tiled(image)
		#tiles = self.xp.asarray(tiles)
		for z in range(len(tiles)):
			tile[	 : zpad, :, :] = tiles[z][0:1]
			tile[ zpad:-zpad, :, :] = tiles[z]
			tile[-zpad:	 , :, :] = tiles[z][-1:]

			tile_fft[:] = self.xp.fft.fftn(tile)
			self.xp.multiply(tile_fft, psf_fft[z], out=tile_fft)
			tile_res[:] = self.xp.fft.ifftn(tile_fft)[zpad:-zpad].real

		#im_cupy = tiler.untile(im_cupy)
		#untiled = cp.asnumpy(im_cupy)

	def tiled(self, image):
		"""
		Tile an image into overlapping tiles.
		"""
		xp = cp.get_array_module(image)

		# Get image dimensions
		sz, sx, sy = image.shape

		# Store for later use in untile
		self.sz = sz
		self.sx = sx
		self.sy = sy

		# Calculate number of tiles in each dimension
		self.nx = int(xp.ceil(sx / self.tile_size))
		self.ny = int(xp.ceil(sy / self.tile_size))

		# Pad the image
		padded = xp.pad(image, ((0, 0), (self.overlap, self.overlap), (self.overlap, self.overlap)), mode='edge')

		# Pre-allocate output array for better performance
		num_tiles = self.nx * self.ny
		tiles = xp.empty((num_tiles, sz, self.tile_size+2*self.overlap, self.tile_size+2*self.overlap), dtype=image.dtype)

		# Extract tiles
		idx = 0
		for y in range(0, sy, self.tile_size):
			for x in range(0, sx, self.tile_size):
				tiles[idx] = padded[:, y:y+self.tile_size+2*self.overlap, x:x+self.tile_size+2*self.overlap]
				idx += 1

		return tiles

	def untiled(self, image):
		"""
		Reconstruct the original image from tiled representation.

		Parameters:
		-----------
		tiled_image : ndarray
			Stacked tiles with shape (num_tiles, sz, tile_size+2*overlap, tile_size+2*overlap)

		Returns:
		--------
		ndarray
			Reconstructed image with shape (sz, sx, sy)
		"""

		# Extract the usable part of each tile (removing the overlap)
		out = image[:, :, self.overlap:-self.overlap, self.overlap:-self.overlap]

		# Reshape to organize tiles in a grid
		out = out.reshape(self.ny, self.nx, self.sz, self.tile_size, self.tile_size)
		
		# Transpose and reshape to reconstruct the image
		#out = out.transpose(2, 0, 3, 1, 4).reshape(self.sz, self.ny * self.tile_size, self.nx * self.tile_size)

		out = out.reshape(self.ny, self.nx, self.sz, self.tile_size, self.tile_size)
		out = cp.ascontiguousarray(out.transpose(2, 0, 3, 1, 4))  # Single contiguous conversion
		out = out.reshape(self.sz, self.ny * self.tile_size, self.nx * self.tile_size)

		return out
	def center_psf(psf):
		"""
		Inserts `psf` into a zero-padded array of `target_shape`, cropping if necessary.
	
		Parameters:
		- psf (ndarray): The PSF array (NumPy or CuPy).
		- target_shape (tuple): The desired output shape.
	
		Returns:
		- ndarray: The centered PSF inside a zero-padded/cropped array.
		"""
		target_shape = (40, self.tile_size, self.tile_size)
		xp = cp.get_array_module(psf)  # Handle NumPy or CuPy
		target_shape = xp.array(target_shape)
		psf_shape = xp.array(psf.shape)
	
		psff = xp.zeros(target_shape, dtype=psf.dtype)  # Use same dtype for consistency
		psf /= psf.sum()  # Normalize
	
		# Compute start & end indices for both source (psf) and target (psff)
		start_psff = xp.maximum(0, (target_shape - psf_shape) // 2)
		end_psff = start_psff + xp.minimum(target_shape, psf_shape)
	
		start_psf = xp.maximum(0, (psf_shape - target_shape) // 2)
		end_psf = start_psf + xp.minimum(target_shape, psf_shape)
	
		# Assign using slices
		slices_psff = tuple(slice(int(s), int(e)) for s, e in zip(start_psff, end_psff))
		slices_psf = tuple(slice(int(s), int(e)) for s, e in zip(start_psf, end_psf))
		psff[slices_psff] = psf[slices_psf]
	
		return psff

