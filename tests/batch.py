import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change "1" to the desired GPU ID

import time
import gc
import cupy as cp
import numpy as np
import scipy.interpolate
import napari
from ioMicro import *
#from sdeconv.deconv import SWiener
from time import sleep
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


import torch
from sdeconv.core import SSettings
from sdeconv.deconv.interface import SDeconvFilter
from sdeconv.deconv._utils import pad_2d, pad_3d, unpad_3d, psf_parameter

from filters import wiener_deconvolve


def parse_psf(im,psf):
	psff = np.zeros(im.shape,dtype=np.float32)

	slices = [(slice((s_psff-s_psf_full_)//2,(s_psff+s_psf_full_)//2),slice(None)) if s_psff>s_psf_full_ else (slice(None),slice((s_psf_full_-s_psff)//2,(s_psf_full_+s_psff)//2)) for s_psff,s_psf_full_ in zip(psff.shape,psf.shape)]
	sl_psff,sl_psf_full_ = list(zip(*slices))
	psff[sl_psff]=psf[sl_psf_full_]
	return np.array(psff,dtype=np.float32)

def center_psf(psf, target_shape=(40, 300, 300)):
    """
    Inserts `psf` into a zero-padded array of `target_shape`,
    cropping if necessary.

    Parameters:
    - psf (cp.ndarray): The PSF array to insert.
    - target_shape (tuple): The desired output shape.

    Returns:
    - cp.ndarray: The centered PSF inside a zero-padded/cropped array.
    """
    psff = cp.zeros(target_shape, dtype=cp.float16)
    psf = cp.asarray(psf)

	# Normalize
    psf /= cp.sum(psf)

    # Compute start & end indices for both the source and target
    start_psff = cp.asnumpy(cp.maximum(0, (cp.array(target_shape) - cp.array(psf.shape)) // 2)).tolist()
    end_psff = (cp.array(start_psff) + cp.minimum(cp.array(target_shape), cp.array(psf.shape))).tolist()

    start_psf = cp.asnumpy(cp.maximum(0, (cp.array(psf.shape) - cp.array(target_shape)) // 2)).tolist()
    end_psf = (cp.array(start_psf) + cp.minimum(cp.array(target_shape), cp.array(psf.shape))).tolist()

    # Assign using slices (convert CuPy arrays to native Python integers)
    psff[tuple(slice(int(s), int(e)) for s, e in zip(start_psff, end_psff))] = psf[tuple(slice(int(s), int(e)) for s, e in zip(start_psf, end_psf))]
    return psff

import cupy as cp
def laplacian_3d(shape):
    """Create a 3D Laplacian kernel for a given shape."""
    lap = cp.zeros(shape, dtype=cp.float32)
    z_c, y_c, x_c = shape[0] // 2, shape[1] // 2, shape[2] // 2
    lap[z_c, y_c, x_c] = 6
    lap[z_c - 1, y_c, x_c] = -1
    lap[z_c + 1, y_c, x_c] = -1
    lap[z_c, y_c - 1, x_c] = -1
    lap[z_c, y_c + 1, x_c] = -1
    lap[z_c, y_c, x_c - 1] = -1
    lap[z_c, y_c, x_c - 1] = -1  # Bug fix (previously had two -1s at the same position)
    return lap

def batch_laplacian_fft(batch_size, spatial_shape):
    """Compute the 3D Laplacian in frequency space and prepare for batch processing."""
    lap = laplacian_3d(spatial_shape)  # Create a single 3D Laplacian
    lap_fft = cp.fft.fftn(cp.fft.ifftshift(lap))  # Shift Laplacian to center before FFT
    return lap_fft[None, ...]  # Add batch dimension without copying memory


def full_deconv_map(im__, psfs):  # Accept exactly two arguments
	out = full_deconv(im__, s_=500, pad=100, psf=psfs, parameters={'method':'wiener','beta':0.001}, gpu=True, force=True)
	del out
	gc.collect()
	torch.cuda.empty_cache()  # Releases unused memory back to the system
	torch.cuda.ipc_collect()  # Helps with memory fragmentation
	return None

cp.cuda.Device(1).use()
#streams = [cp.cuda.Stream(non_blocking=True ) for _ in range(6)]  # Create separate streams

def full_deconv_cupy_map(im__, psfs):  # Accept exactly two arguments
#def full_deconv_cupy_map(im__, psfs, stream_id):
	#cp.cuda.Device(1).use()
	#with streams[stream_id]:
	out = full_deconv(im__, s_=500, pad=100, psf=psfs, parameters={'method':'cupy','beta':0.001}, gpu=True, force=True)
	del out
	gc.collect()
	cp.get_default_memory_pool().free_all_blocks()
	cp.get_default_pinned_memory_pool().free_all_blocks()
	return None

def process_image(im, psfs, stream):
    """Run full_deconv for one image using its own stream."""
    return full_deconv_cupy_map(im, psfs, stream)


def wrap_pad(tiles, pad):
    H, W, C, T_H, T_W = tiles.shape  # (10, 10, 40, 300, 300)

    # Initialize padded array
    padded = np.zeros((H, W, C, T_H + 2 * pad, T_W + 2 * pad), dtype=tiles.dtype)

    # Center region (original tiles)
    padded[:, :, :, pad:-pad, pad:-pad] = tiles

    # Wrap edges
    padded[:, :, :, :pad, pad:-pad] = tiles[(np.arange(H) - 1) % H, :, :, -pad:, :]  # Top from bottom neighbor
    padded[:, :, :, -pad:, pad:-pad] = tiles[(np.arange(H) + 1) % H, :, :, :pad, :]  # Bottom from top neighbor
    padded[:, :, :, pad:-pad, :pad] = tiles[:, (np.arange(W) - 1) % W, :, :, -pad:]  # Left from right neighbor
    padded[:, :, :, pad:-pad, -pad:] = tiles[:, (np.arange(W) + 1) % W, :, :, :pad]  # Right from left neighbor

    # Wrap corners correctly
    padded[:, :, :, :pad, :pad] = tiles[(np.arange(H) - 1) % H, (np.arange(W) - 1) % W, :, -pad:, -pad:]  # Top-left from bottom-right
    padded[:, :, :, :pad, -pad:] = tiles[(np.arange(H) - 1) % H, (np.arange(W) + 1) % W, :, -pad:, :pad]  # Top-right from bottom-left
    padded[:, :, :, -pad:, :pad] = tiles[(np.arange(H) + 1) % H, (np.arange(W) - 1) % W, :, :pad, -pad:]  # Bottom-left from top-right
    padded[:, :, :, -pad:, -pad:] = tiles[(np.arange(H) + 1) % H, (np.arange(W) + 1) % W, :, :pad, :pad]  # Bottom-right from top-left

    return padded

def extract_overlapping_tiles(image, tile_size, pad):
    """
    Extracts overlapping tiles from an image ensuring exactly 100 tiles.

    Args:
        image (ndarray): Input array of shape (C, H, W).
        tile_size (int): Size of each tile before padding.
        pad (int): Overlap/padding size.

    Returns:
        ndarray: Overlapping tiles of shape (100, C, tile_size+2*pad, tile_size+2*pad).
    """
    C, H, W = image.shape
    full_tile_size = tile_size + 2 * pad  # Tile size including padding
    step = tile_size  # Non-overlapping stride

    # Ensure exactly 10 x 10 tiles = 100 tiles
    y_positions = np.linspace(0, H - full_tile_size, 10, dtype=int)
    x_positions = np.linspace(0, W - full_tile_size, 10, dtype=int)

    tiles = []
    for y in y_positions:
        for x in x_positions:
            tile = image[:, y:y+full_tile_size, x:x+full_tile_size]
            tiles.append(tile)

    return np.stack(tiles, axis=0)


if __name__ == "__main__":
	psf_file = 'psfs/dic_psf_60X_cy5_Scope5.pkl'
	psfs = np.load(psf_file, allow_pickle=True)
	fov = 'Conv_zscan1_002.zarr'
	fld = '/data/07_22_2024__PFF_PTBP1/H0_AER_set1'
	icol = 2
	im_ = read_im(fld+os.sep+fov)
	#im__ = np.array(im_[icol],dtype=np.float32)
	im__ = np.array(im_[-1])
	### new method
	fl_med = 'flat_field/Scope5_med_col_raw'+str(icol)+'.npz'
	if os.path.exists(fl_med):
		im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
		im_med = cv2.blur(im_med,(20,20))
		im__ = im__/im_med*np.median(im_med)
		im__ = im__.astype(np.float16)
	else:
		print(fl_med)
		print("Did not find flat field")
	mempool = cp.get_default_memory_pool()
	import gc
	pad = 50
	n = 50
	beta = 0.001
	items = [ (im__,psfs) for _ in range(n)]
	psf_batch = cp.stack(list(map(center_psf, list(psfs.values()))))
	psf_batch = cp.pad(psf_batch, ((0, 0), (20, 20), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
	psf_batch = cp.roll(psf_batch,
                   shift=(-psf_batch.shape[1] // 2, -psf_batch.shape[2] // 2, -psf_batch.shape[3] // 2),
                   axis=(1, 2, 3))
	'''
	import gc
	gc.collect()
	for obj in gc.get_objects():
		if isinstance(obj, cp.ndarray):
			print(f"CuPy array with shape {obj.shape} and dtype {obj.dtype}")
			print(f"Memory usage: {obj.nbytes / 1024**2:.2f} MB")  # Convert to MB
	print(f"Used memory after: {mempool.used_bytes() / 1024**2:.2f} MB")
	'''
	psf_fft = cp.fft.fftn(psf_batch, axes=(-3, -2, -1))
	del psf_batch
	laplacian_fft = batch_laplacian_fft(100, (40+40,300+2*pad,300+2*pad))  # Shape: (1, 40, 300, 300)
	psf_conj = cp.conj(psf_fft)
	cp.multiply(psf_fft, psf_conj, out=psf_fft)
	cp.multiply(laplacian_fft, laplacian_fft.conj(), out=laplacian_fft)
	cp.multiply(laplacian_fft, beta, out=laplacian_fft)
	cp.add(psf_fft, laplacian_fft, out=psf_fft)
	del laplacian_fft
	#gc.collect()
	#cp.get_default_memory_pool().free_all_blocks()
	#cp.get_default_pinned_memory_pool().free_all_blocks()
	start = time.time()
	for _ in range(n):
		#im_batch = im__.reshape(40, 10, 300, 10, 300)
		#im_batch = im_batch.transpose(1, 3, 0, 2, 4).reshape(100, 40, 300, 300)
		im_batch = extract_overlapping_tiles(im__, 300, 50)
		im_batch = cp.asarray(im_batch, dtype=np.float16)
		im_batch = cp.pad(im_batch, ((0, 0), (20, 20), (0, 0), (0, 0)), mode='constant', constant_values=0)
		#im_batch = cp.pad(im_batch, ((0, 0), (20, 20), (pad, pad), (pad, pad)), mode='constant')
		image_fft = cp.fft.fftn(im_batch, axes=(-3, -2, -1))
		del im_batch
		# Wiener deconvolution in frequency domain
		#den = psf_fft * cp.conj(psf_fft) + beta * laplacian_fft * cp.conj(laplacian_fft)
		#psf_conj = cp.conj(psf_fft)
		# do all the operations in place
		cp.multiply(image_fft, psf_conj, out=image_fft)
		cp.true_divide(image_fft, psf_fft, out=image_fft)
		#del psf_fft , psf_conj
		image_fft[:] = cp.fft.ifftn(image_fft, axes=(-3, -2, -1))
		#image_fft = cp.real(image_fft)
		image_fft = image_fft.real  # This creates a new view, but still complex64
		image_fft = cp.ascontiguousarray(image_fft, dtype=cp.float16)  # Forces in-place conversion
		'''	
		viewer = napari.Viewer()
		viewer.add_image(cp.asnumpy(im_batch), name="original", scale=(1, 1, 1))
		viewer.add_image(cp.asnumpy(image_fft), name="deconv", scale=(1, 1, 1))
		napari.run()
		exit()
		'''
		del image_fft
		gc.collect()
		cp.get_default_memory_pool().free_all_blocks()  # Free up memory immediately
	del psf_fft, psf_conj

	end = time.time()
	print(f"cupy (untiled) time: {end - start:.6f} seconds")
	#print(f"CuPy Max GPU Memory Used: {max_usage / 1e6:.2f} MB")
	gc.collect()
	cp.get_default_memory_pool().free_all_blocks()
	cp.get_default_pinned_memory_pool().free_all_blocks()
	exit()

	start = time.time()
	for _ in range(n):
		out_im = full_deconv(im__, s_=500, pad=100, psf=psfs, parameters={'method':'cupy','beta':0.001}, gpu=True, force=True)
	end = time.time()
	print(f"cupy (tiled) time: {end - start:.6f} seconds")
	del out_im
	gc.collect()
	cp.get_default_memory_pool().free_all_blocks()
	cp.get_default_pinned_memory_pool().free_all_blocks()
	
	start = time.time()
	with Pool(processes=4) as pool:
		results = pool.starmap(full_deconv_cupy_map, items)  # Use starmap for argument unpacking
	#tasks = [(im, psfs, i % 6) for i, (im, psfs) in enumerate(items)]
	#with ThreadPoolExecutor(max_workers=6) as executor:
		#results = list(executor.map(lambda args: full_deconv_cupy_map(*args), items))
	#	results = list(executor.map(lambda args: full_deconv_cupy_map(*args), tasks))


	end = time.time()
	print(f"cupy (tiled parallel) time: {end - start:.6f} seconds")
	del results
	gc.collect()
	cp.get_default_memory_pool().free_all_blocks()
	cp.get_default_pinned_memory_pool().free_all_blocks()
	cp.cuda.Device(1).synchronize()
		
	start = time.time()
	for _ in range(n):
		out_im = full_deconv(im__, s_=500, pad=100, psf=psfs, parameters={'method':'wiener','beta':0.001}, gpu=True, force=True)
	end = time.time()
	print(f"torch (tiled) time: {end - start:.6f} seconds")
	del out_im
	gc.collect()
	torch.cuda.empty_cache()  # Releases unused memory back to the system
	torch.cuda.ipc_collect()  # Helps with memory fragmentation

	start = time.time()
	with Pool(processes=4) as pool:
		results = pool.starmap(full_deconv_map, items)  # Use starmap for argument unpacking
	end = time.time()
	print(f"torch (tiled parallel) time: {end - start:.6f} seconds")
	exit()

	viewer = napari.Viewer()
	viewer.add_image(cp.asnumpy(im_batch), name="original", scale=(1, 1, 1))
	viewer.add_image(cp.asnumpy(deconvolved_images), name="deconv", scale=(1, 1, 1))
	napari.run()
	exit()

