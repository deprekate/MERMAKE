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

import torch
from sdeconv.core import SSettings
obj = SSettings.instance()
obj.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
from sdeconv.core import SSettings
from sdeconv.deconv.interface import SDeconvFilter
from sdeconv.deconv._utils import pad_2d, pad_3d, unpad_3d, psf_parameter

from mermake.filters import wiener_deconvolve


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

    # Compute start & end indices for both the source and target
    start_psff = cp.asnumpy(cp.maximum(0, (cp.array(target_shape) - cp.array(psf.shape)) // 2)).tolist()
    end_psff = (cp.array(start_psff) + cp.minimum(cp.array(target_shape), cp.array(psf.shape))).tolist()

    start_psf = cp.asnumpy(cp.maximum(0, (cp.array(psf.shape) - cp.array(target_shape)) // 2)).tolist()
    end_psf = (cp.array(start_psf) + cp.minimum(cp.array(target_shape), cp.array(psf.shape))).tolist()

    # Assign using slices (convert CuPy arrays to native Python integers)
    psff[tuple(slice(int(s), int(e)) for s, e in zip(start_psff, end_psff))] = psf[tuple(slice(int(s), int(e)) for s, e in zip(start_psf, end_psf))]
    psff /= cp.sum(psff)
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
    lap[z_c, y_c, x_c + 1] = -1  # Bug fix (previously had two -1s at the same position)
    return lap

def batch_laplacian(batch_size, spatial_shape):
    """Efficiently create a 4D Laplacian for batch processing."""
    lap = laplacian_3d(spatial_shape)  # Create a single 3D Laplacian
    return lap[None, ...]  # Add batch dim without repeating memory

def batch_laplacian_fft(batch_size, spatial_shape):
    """Compute the 3D Laplacian in frequency space and prepare for batch processing."""
    lap = laplacian_3d(spatial_shape)  # Create a single 3D Laplacian
    lap_fft = cp.fft.fftn(cp.fft.ifftshift(lap))  # Shift Laplacian to center before FFT
    return lap_fft[None, ...]  # Add batch dimension without copying memory

def pad_patch(patch, pad_width):
    """
    Pads each patch (z, x, y) individually with zeros.

    Parameters:
    - patch: cupy.ndarray, input patch with shape (z, x, y).
    - pad_width: tuple, padding size for (x, y).

    Returns:
    - padded_patch: cupy.ndarray, padded patch.
    """
    return cp.pad(patch, ((0, 0), (pad_width[0], pad_width[0]),
                          (pad_width[1], pad_width[1])),
                  mode='constant', constant_values=0)

def full_deconv_map(im__, psfs):  # Accept exactly two arguments
	out = full_deconv(im__, s_=500, pad=100, psf=psfs, parameters={'method':'wiener','beta':0.001}, gpu=True, force=True)
	del out
	gc.collect()
	torch.cuda.empty_cache()  # Releases unused memory back to the system
	torch.cuda.ipc_collect()  # Helps with memory fragmentation
	return None

def full_deconv_cupy_map(im__, psfs):  # Accept exactly two arguments
	out = full_deconv_cupy(im__, s_=500, pad=100, psf=psfs, parameters={'method':'wiener','beta':0.001}, gpu=True, force=True)
	del out
	return None

if __name__ == "__main__":
	psf_file = '../psfs/dic_psf_60X_cy5_Scope5.pkl'
	psfs = np.load(psf_file, allow_pickle=True)
	
	fov = 'Conv_zscan1_002.zarr'
	fld = '/data/07_22_2024__PFF_PTBP1/H0_AER_set1'
	icol = 2
	im_ = read_im(fld+os.sep+fov)
	#im__ = np.array(im_[icol],dtype=np.float32)
	im__ = np.array(im_[icol])
	### new method
	fl_med = '../flat_field/Scope5_med_col_raw'+str(icol)+'.npz'
	if os.path.exists(fl_med):
		im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
		im_med = cv2.blur(im_med,(20,20))
		im__ = im__/im_med*np.median(im_med)
		im__ = im__.astype(np.float16)
	else:
		print(fl_med)
		print("Did not find flat field")

	import gc
	pad = 50
	n = 20
	items = [ (im__,psfs) for _ in range(n)]
	
	psf_batch = cp.stack(list(map(center_psf, psfs.values())))
	psf_batch = cp.roll(psf_batch,
                   shift=(-psf_batch.shape[1] // 2, -psf_batch.shape[2] // 2, -psf_batch.shape[3] // 2),
                   axis=(1, 2, 3))
	psf_batch = cp.pad(psf_batch, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
	psf_fft = cp.fft.fftn(psf_batch, axes=(-3, -2, -1))
	laplacian_fft = batch_laplacian_fft(100, (40,400,400))  # Shape: (1, 40, 300, 300)
	psf_conj = cp.conj(psf_fft)
	cp.abs(psf_fft, out=psf_fft)
	cp.square(psf_fft, out=psf_fft)
	cp.add(psf_fft, 0.001, out=psf_fft)
	cp.abs(laplacian_fft, out=laplacian_fft)
	cp.square(laplacian_fft, out=laplacian_fft)
	cp.multiply(psf_fft, laplacian_fft, out=psf_fft) 
	del psf_batch, laplacian_fft
	
	mempool = cp.get_default_memory_pool()
	pinned_mempool = cp.get_default_pinned_memory_pool()
	start = time.time()
	for _ in range(n):
		im_batch = im__.reshape(40, 10, 300, 10, 300)
		im_batch = im_batch.transpose(1, 3, 0, 2, 4).reshape(100, 40, 300, 300)
		im_batch = cp.asarray(im_batch, dtype=np.float16)
		im_batch = cp.pad(im_batch, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
		image_fft = cp.fft.fftn(im_batch, axes=(-3, -2, -1))
		del im_batch
	
		# Wiener deconvolution in frequency domain
		#den = psf_fft * cp.conj(psf_fft) + beta * laplacian_fft * cp.conj(laplacian_fft)
		#den = cp.abs(psf_fft) ** 2 + 0.01 * cp.abs(laplacian_fft_batch) ** 2
		#deconv_fft = image_fft * cp.conj(psf_fft) / den
		#deconvolved_images = cp.fft.ifftn(deconv_fft, axes=(-3, -2, -1)).real
		#psf_conj = cp.conj(psf_fft)

		# do all the operations in place
		cp.multiply(image_fft, psf_conj, out=image_fft)
		cp.true_divide(image_fft, psf_fft, out=image_fft)
		image_fft[:] = cp.fft.ifftn(image_fft, axes=(-3, -2, -1))

		image_fft = image_fft.real  # This creates a new view, but still complex64
		image_fft = cp.ascontiguousarray(image_fft, dtype=cp.float16)  # Forces in-place conversion
		cp.get_default_memory_pool().free_all_blocks()  # Free up memory immediately
	del psf_fft, psf_conj
	end = time.time()
	print(f"cupy (untiled) time: {end - start:.6f} seconds")
	gc.collect()
	cp.get_default_memory_pool().free_all_blocks()
	cp.get_default_pinned_memory_pool().free_all_blocks()
	
	start = time.time()
	for _ in range(n):
		out_im = full_deconv_cupy(im__, s_=500, pad=100, psf=psfs, parameters={'method':'wiener','beta':0.001}, gpu=True, force=True)
	end = time.time()
	print(f"cupy (tiled) time: {end - start:.6f} seconds")
	del out_im
	gc.collect()
	cp.get_default_memory_pool().free_all_blocks()
	cp.get_default_pinned_memory_pool().free_all_blocks()
	
	start = time.time()
	with Pool(processes=3) as pool:
		results = pool.starmap(full_deconv_cupy_map, items)  # Use starmap for argument unpacking
	end = time.time()
	print(f"cupy (tiled parallel) time: {end - start:.6f} seconds")
	del results
	gc.collect()
	cp.get_default_memory_pool().free_all_blocks()
	cp.get_default_pinned_memory_pool().free_all_blocks()
	
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
	with Pool(processes=2) as pool:
		results = pool.starmap(full_deconv_map, items)  # Use starmap for argument unpacking
	end = time.time()
	print(f"torch (tiled parallel) time: {end - start:.6f} seconds")
	exit()

	import napari
	viewer = napari.Viewer()
	viewer.add_image(cp.asnumpy(im_batch), name="original", scale=(1, 1, 1))
	viewer.add_image(cp.asnumpy(deconvolved_images), name="deconv", scale=(1, 1, 1))
	napari.run()
	exit()

