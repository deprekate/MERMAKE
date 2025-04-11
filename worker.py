import os
import sys
import concurrent.futures

# put this first to make sure to capture the correct gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change "1" to the desired GPU ID
import cupy as cp
cp.cuda.Device(0).use() # The above export doesnt always work so force CuPy to use GPU 0
import numpy as np
import cv2

from mermake.utils import Utils
from mermake.maxima import Maxima
from mermake.deconvolver import Deconvolver
#from mermake.maxima_gpu import find_local_maxima
from mermake.maxim import find_local_maxima
from mermake.io import image_generator, save_data, save_data_dapi, get_files

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
		med = cv2.blur(med,(20,20))
		med = med / np.median(med)
		im_med.append(med)
	im_med = cp.asarray(np.stack(im_med))

	
	psfs = np.load(psf_file, allow_pickle=True)

	# this mimics the behavior if there is only a single psf
	key = (0,1500,1500)
	psfs = { key : psfs[key] }

	# various classes to do the computations efficiently
	utils = Utils(ksize=30)
	tile_size = 500
	hyb_deconvolver = Deconvolver(psfs, shape[1:], tile_size=500, overlap=89, zpad=39, beta=0.0001)
	dapi_deconvolver = Deconvolver(psfs, shape[1:], tile_size=500, overlap=59, zpad=13, beta=0.01)
	#hyb_maxima = Maxima(threshold = 3600, delta = 1, delta_fit = 3,sigmaZ = 1, sigmaXY = 1.5)	
	#dapi_maxima = Maxima(threshold = 3, delta = 5, delta_fit = 5, sigmaZ = 1, sigmaXY = 1.5)	

	# the save file executor to do the saving in parallel with computations
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
	
	for cim in image_generator(hybs, fovs):
		path = cim.path
		print(path)
		for icol in [0,1,2]:
			# there is probably a better way to do the Xh stacking
			Xhf = list()
			view = cim[icol]
			flat = im_med[icol]
			for x,y,tile,raw in hyb_deconvolver.tile_wise(view, flat):
				tile[:] = utils.norm_image(tile)
				#Xh0 = hyb_maxima.apply(tile, im_raw=raw)
				Xh = find_local_maxima(tile, 3600.0, 1, 3, sigmaZ = 1, sigmaXY = 1.5, raw = raw)
				keep = cp.all(Xh[:,1:3] < tile_size+89, axis=-1)
				keep &= cp.all(Xh[:,1:3] >= 89, axis=-1)
				Xh = Xh[keep]
				Xh[:,1] += x - 89
				Xh[:,2] += y - 89
				if len(Xh):
					Xhf.append(Xh)
			Xhf = cp.vstack(Xhf)
			executor.submit(save_data, save_folder, path, icol, Xhf)
			del view, flat
		# now do dapi, but first clear some stuff from gpu ram
		icol += 1
		dapi = cim[icol].copy()
		cim.clear()
		del cim
		cp._default_memory_pool.free_all_blocks()  # Free the standard memory pool
		cp._default_pinned_memory_pool.free_all_blocks()  # Free pinned memory pool
		import gc
		gc.collect()  # Force Python's garbage collection to clean up
		cp.cuda.runtime.deviceSynchronize()  # Ensure all GPU operations are completed
		deconv = dapi_deconvolver.apply(dapi)
		deconv[:] /= deconv.std()
		Xh_plus = find_local_maxima(deconv, 3.0, 1, 3, sigmaZ = 1, sigmaXY = 1.5, raw = dapi)
		deconv *= -1
		Xh_minus = find_local_maxima(deconv, 3.0, 1, 3, sigmaZ = 1, sigmaXY = 1.5, raw = dapi )
		del deconv
		executor.submit(save_data_dapi, save_folder, path, icol, Xh_plus, Xh_minus)




