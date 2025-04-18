import os
import sys
import concurrent.futures
import time

# put this first to make sure to capture the correct gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change "1" to the desired GPU ID
import cupy as cp
cp.cuda.Device(0).use() # The above export doesnt always work so force CuPy to use GPU 0
import numpy as np
import cv2

from mermake.deconvolver import Deconvolver
from mermake.maxima import find_local_maxima
#from other.maxim import find_local_maxima
from mermake.io import image_generator, save_data, save_data_dapi, get_files
import mermake.blur as blur

def profile():
	import gc
	mempool = cp.get_default_memory_pool()
	# Loop through all objects in the garbage collector
	for obj in gc.get_objects():
		if isinstance(obj, cp.ndarray):
			# Check if it's a view (not a direct memory allocation)
			if obj.base is not None:
				# Skip views as they do not allocate new memory
				continue
			print(f"CuPy array with shape {obj.shape} and dtype {obj.dtype}")
			print(f"Memory usage: {obj.nbytes / 1024**2:.2f} MB")  # Convert to MB
	print(f"Used memory after: {mempool.used_bytes() / 1024**2:.2f} MB")

if __name__ == "__main__":

	# this is all stuff that will eventually be replaced with a toml settings file
	psf_file = 'psfs/dic_psf_60X_cy5_Scope5.pkl'
	master_data_folders = ['/data/07_22_2024__PFF_PTBP1']
	save_folder = 'output_new'
	iHm = 1 ; iHM = 16
	shape = (4,40,3000,3000)
	items = [(set_,ifov) for set_ in ['_set1'] for ifov in range(1,5)]
	hybs = list()
	fovs = list()
	for item in items[:4]:
		all_flds,fov = get_files(master_data_folders, item, iHm=iHm, iHM=iHM)
		hybs.append(all_flds)
		fovs.append(fov)

	im_med = list()
	for icol in [0,1,2,3]:
		fl_med = 'flat_field/Scope5_med_col_raw'+str(icol)+'.npz'
		med = np.array(np.load(fl_med)['im'],dtype=np.float32)
		if icol != 3:
			# the dapi flat field is not blurred
			med = cv2.blur(med,(20,20))
		med = med / np.median(med)
		im_med.append(med)
	im_med = cp.asarray(np.stack(im_med))

	# eventually make a smart psf loader method to handle the different types of psf files
	psfs = np.load(psf_file, allow_pickle=True)

	# this mimics the behavior if there is only a single psf
	#key = (0,1500,1500)
	#psfs = { key : psfs[key] }
	
	# settings
	tile_size = 300
	overlap = 89
	# these can be very large objects in gpu ram, adjust accoringly to suit gpu specs
	hybs_deconvolver = Deconvolver(psfs, shape[1:], tile_size=tile_size, overlap=overlap, zpad=39, beta=0.0001)
	dapi_deconvolver = Deconvolver(psfs, shape[1:], tile_size=tile_size, overlap=overlap-10, zpad=19, beta=0.01)

	# the save file executor to do the saving in parallel with computations
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

	deconv = cp.empty(shape[1:], dtype=cp.float32)	
	buffer_channel = cp.empty(shape[1:], dtype=cp.float32)	
	buffer_tile = cp.empty( (shape[1], tile_size+2*overlap, tile_size+2*overlap), dtype=cp.float32 )
	for cim in image_generator(hybs, fovs):
		print(cim[0].path, flush=True)
		for icol in [0,1,2]:
			# there is probably a better way to do the Xh stacking
			Xhf = list()
			view = cim[icol]
			flat = im_med[icol]
			for x,y,tile,raw in hybs_deconvolver.tile_wise(view, flat):
				# do two separate 1d blurs for now, put into a method later
				# i think a blur with the output also being the input works?
				blur.box_1d(tile, 30, axis=1, output=buffer_tile)
				blur.box_1d(buffer_tile, 30, axis=2, output=buffer_tile)
				cp.subtract(tile, buffer_tile, out=tile)
				Xh = find_local_maxima(tile, 3600.0, 1, 3, sigmaZ = 1, sigmaXY = 1.5, raw = raw)
				keep = cp.all((Xh[:,1:3] >= overlap) & (Xh[:,1:3] < tile_size + overlap), axis=-1)
				Xh = Xh[keep]
				Xh[:,1] += x - overlap
				Xh[:,2] += y - overlap
				Xhf.append(Xh)
			Xhf = [x for x in Xhf if x.shape[0] > 0]
			Xhf = cp.vstack(Xhf)
			cp.cuda.runtime.deviceSynchronize()
			executor.submit(save_data, save_folder, view.path, icol, Xhf)
			view.clear()
			del view, Xh, Xhf
			#cp._default_memory_pool.free_all_blocks()

		view = cim[3]
		flat = im_med[3]
		del cim
		cp._default_memory_pool.free_all_blocks()
		cp.cuda.runtime.deviceSynchronize()
		
		# now do dapi, but first clear some stuff from gpu ram to fit in 12GB
		# Deconvolve in-place into `deconv`
		dapi_deconvolver.apply(view, flat_field=flat, output=deconv)
		blur.box_1d(deconv, 50, axis=1, output=buffer_channel)
		blur.box_1d(buffer_channel, 50, axis=2, output=buffer_channel)
		cp.subtract(deconv, buffer_channel, out=deconv)
		std_val = float(cp.asnumpy(cp.linalg.norm(deconv.ravel()) / cp.sqrt(deconv.size)))
		cp.divide(deconv, std_val, out=deconv)
		Xh_plus = find_local_maxima(deconv, 3.0, 5, 5, sigmaZ = 1, sigmaXY = 1.5, raw = view )
		cp.multiply(deconv, -1, out=deconv)
		Xh_minus = find_local_maxima(deconv, 3.0, 5, 5, sigmaZ = 1, sigmaXY = 1.5, raw = view )
		cp.cuda.runtime.deviceSynchronize()
		executor.submit(save_data_dapi, save_folder, view.path, icol, Xh_plus, Xh_minus)
		view.clear()
		del view
		#cp._default_memory_pool.free_all_blocks()



