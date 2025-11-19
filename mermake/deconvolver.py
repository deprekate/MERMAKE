import gc
from itertools import zip_longest, chain, repeat, cycle

import cupy as cp
import numpy as np

from . import blur
from .fill import reflect

def repeat_last(iterable):
	it = iter(iterable)
	try:
		last = next(it)  # Get the first element
	except StopIteration:
		return  # Empty iterable, nothing to repeat
	yield last
	for item in it:
		yield item
		last = item  # Update last element
	while True:
		yield last  # Repeat last element indefinitely


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
	def __init__(self, psfs, channel_shape, tile_size=300, zpad=0, overlap=89, beta=0.001, xp=cp, **kwargs):
		self.tile_size = tile_size
		self.tile_height = channel_shape[0]

		if isinstance(psfs, dict):
			# new method using multiple psfs taken across fov
			psf_stack = np.stack(list(map(self.center_psf, list(psfs.values()))))
		else:
			# old single psf method
			batch_size = (channel_shape[-1] // tile_size) ** 2
			psf_stack = self.center_psf(psfs)
			psf_stack = self.center_psf(psfs)[None, ...]
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
		if xp == cp:
			cp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool
			cp._default_pinned_memory_pool.free_all_blocks()  # Free pinned memory pool
			cp.cuda.runtime.deviceSynchronize()  # Ensure all operations are completed

		# Preallocate tile arrays
		if tile_size:
			tile_shape = (tile_size + 2*overlap, tile_size + 2*overlap)
			self.tile_pad = xp.empty((2*zpad + self.tile_height, *tile_shape), dtype=xp.float32)
			self.tile_res = xp.empty((		 self.tile_height, *tile_shape), dtype=xp.float32)
			self.tile_fft = xp.empty_like(self.tile_pad, dtype=xp.complex64)
			self.tile_buf = xp.empty_like(self.tile_res)
			self.tile_tem = xp.empty_like(self.tile_res)

		self.overlap = overlap
		self.zpad = zpad
		self.xp = xp

	def tile_wise(self, image, flat_field=None, blur_radius=None, **kwargs):
		xp = cp.get_array_module(image)
		zpad = self.zpad
		overlap = self.overlap
		tile_size = self.tile_size
		tile_pad = self.tile_pad
		tile_fft = self.tile_fft
		tile_res = self.tile_res
		tile_buf = self.tile_buf
		tile_tem = self.tile_tem
	
		# use cycle to repeat the single PSF or iterate normally if multiple PSFs exist
		psf_ffts = cycle(self.psf_fft) if len(self.psf_fft) == 1 else iter(self.psf_fft)
		tiles = self.tiled(image)
		flat_field = flat_field if flat_field is None or flat_field.ndim == image.ndim else flat_field[np.newaxis] 
		flats = cycle([(None,None,None)]) if flat_field is None else self.tiled(flat_field)
	
		# the big loop
		for (x,y,tile),(_,_,flat),psf_fft in zip(tiles, flats, psf_ffts):
			zdim,xdim,ydim = tile.shape
			xstart = overlap if x == 0 else 0
			ystart = overlap if y == 0 else 0
			xend = xdim
			yend = ydim
			
			tile_pad[:] = 0.0
			tile_pad[ zpad:zpad+zdim, xstart:xstart+xdim, ystart:ystart+ydim] = tile      # fill
		
			# flat field correction
			if flat_field is not None:
				mask = (slice(zpad,-zpad), slice(xstart,xstart+xdim), slice(ystart,ystart+ydim))
				cp.divide(tile_pad[mask], flat, out=tile_pad[mask])

			# x reflect at left and right
			if x == 0:
				reflect(tile_pad, overlap, mode='in', axis=1, out=tile_pad)
				xend += overlap
			elif xdim < tile_size + (2 * overlap):
				reflect(tile_pad, xdim-1, mode='out', axis=1, out=tile_pad)
				xend += overlap
			# y reflect at top and bottom
			if y == 0:
				reflect(tile_pad, overlap, mode='in', axis=2, out=tile_pad)
				yend += overlap
			elif ydim < tile_size + (2 * overlap):
				reflect(tile_pad, ydim-1, mode='out', axis=2, out=tile_pad)
				yend += overlap
			# z reflect down
			reflect(tile_pad, zpad, mode='in', axis=0, out=tile_pad)
			# z relfect up
			reflect(tile_pad, zpad+zdim-1, mode='out', axis=0, out=tile_pad)
			#tile_pad[:zpad, :, :] = tile_pad[zpad, :, :]
			#tile_pad[-zpad:, :, :] = tile_pad[zpad+zdim-1, :, :]
			#tile_pad[zpad - zdim:zpad, :, :] = tile_pad[zpad:zpad+zdim, :, :][::-1, :, :]
			#tile_pad[zpad + zdim:zpad + zdim + zpad, :, :] = tile_pad[zdim:zpad+zdim, :, :][::-1, :, :]

			# the fft convolution and deconvolution
			tile_fft[:] = xp.fft.fftn(tile_pad)
			xp.multiply(tile_fft, psf_fft, out=tile_fft)
			tile_res[:] = xp.fft.ifftn(tile_fft)[zpad:-zpad].real

			# optional blur subtraction
			if blur_radius is not None:
				blur.box_1d(tile_res, blur_radius, axis=1, out=tile_buf)
				blur.box_1d(tile_buf, blur_radius, axis=2, out=tile_tem)
				xp.subtract(tile_res, tile_tem, out=tile_res)
			
			# flat field uncorrection - put tile_pad back so it matches the raw tile
			if flat_field is not None:
				mask = (slice(zpad,-zpad), slice(xstart,xstart+xdim), slice(ystart,ystart+ydim))
				cp.multiply(tile_pad[mask], flat, out=tile_pad[mask])
			
			yield x,y,tile_res[:,:xend, :yend],tile_pad[zpad:-zpad, :xend, :yend]

	def apply(self, image, flat_field=None, blur_radius=None, output=None, **kwargs):
		xp = cp.get_array_module(image)
		if output is None:
			output = xp.empty_like(image, dtype=xp.float32)

		overlap = self.overlap
		sz, sx, sy = image.shape
		for x, y, tile, _ in self.tile_wise(image, flat_field=flat_field, blur_radius=blur_radius):
				zdim,xdim,ydim = tile.shape
				xdim -= (2 * overlap)
				ydim -= (2 * overlap)
				output[:, x:x+xdim, y:y+ydim] = tile[:,overlap:-overlap, overlap:-overlap]
		return output

	def tiled(self, image):
		"""
		Yield tiles of the image do reflected padding later
		"""
		xp = cp.get_array_module(image)
		sz, sx, sy = image.shape
		self.nx = int(xp.ceil(sx / self.tile_size))
		self.ny = int(xp.ceil(sy / self.tile_size))

		tile_size = self.tile_size
		for x in range(0, sx, tile_size):
			for y in range(0, sy, tile_size):
				# Compute bounds with overlap
				x_start = max(x - self.overlap, 0)
				y_start = max(y - self.overlap, 0)
				x_end = min(x + self.tile_size + self.overlap, sx)
				y_end = min(y + self.tile_size + self.overlap, sy)
	
				# Extract tile region
				tile = image[:, x_start:x_end, y_start:y_end]
				yield x, y, tile

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

	def center_psf(self, psf):
		"""
		Inserts `psf` into a zero-padded array of `target_shape`, cropping if necessary.
	
		Parameters:
		- psf (ndarray): The PSF array (NumPy or CuPy).
		- target_shape (tuple): The desired output shape.
	
		Returns:
		- ndarray: The centered PSF inside a zero-padded/cropped array.
		"""
		xp = cp.get_array_module(psf)  # Handle NumPy or CuPy

		target_shape = xp.array([self.tile_height, self.tile_size, self.tile_size])
		psf_shape = xp.array(psf.shape)
		psff = xp.zeros(tuple(target_shape.tolist()), dtype=psf.dtype)  # Use same dtype for consistency
	
		# Compute start & end indices for both source (psf) and target (psff)
		start_psff = xp.maximum(0, (target_shape - psf_shape) // 2)
		end_psff = start_psff + xp.minimum(target_shape, psf_shape)
	
		start_psf = xp.maximum(0, (psf_shape - target_shape) // 2)
		end_psf = start_psf + xp.minimum(target_shape, psf_shape)
	
		# Assign using slices
		slices_psff = tuple(slice(int(s), int(e)) for s, e in zip(start_psff, end_psff))
		slices_psf = tuple(slice(int(s), int(e)) for s, e in zip(start_psf, end_psf))
		psff[slices_psff] = psf[slices_psf]

		# normalize at the end
		psff /= psff.sum()

		return psff

def full_deconv(image, psfs, flat_field = None, tile_size=300, zpad = None, overlap = 1, beta = 0.001):
	xp = cp.get_array_module(image)

	shape = image.shape
	if zpad is None:
		zpad = shape[0]
	deconvolver = Deconvolver(psfs, channel_shape=shape, zpad=zpad, tile_size=tile_size, overlap=overlap, beta=beta)
	deconv = deconvolver.apply(image, flat_field)

	del deconvolver
	gc.collect()  # Standard Python garbage collection
	if xp == cp:
		cp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool
		cp._default_pinned_memory_pool.free_all_blocks()  # Free pinned memory pool
		cp.cuda.runtime.deviceSynchronize()  # Ensure all operations are completed
	return deconv



