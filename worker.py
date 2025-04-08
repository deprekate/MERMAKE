import os
import concurrent.futures

# put this first to make sure to capture the correct gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change "1" to the desired GPU ID
import cupy as cp
cp.cuda.Device(0).use() # The above export doesnt always work so force CuPy to use GPU 0
import numpy as np
import cv2

from mermake.utils import Utils
from mermake.maxima import Maxima
from mermake.deconvolver import Deconvolver
#from mermake.maxima_gpu import find_local_maxima
from mermake.maxim import find_local_maxima
from mermake.io import image_generator, save_data, get_files



if __name__ == "__main__":

	# this is all stuff that will eventually be replaced with a toml settings file
	psf_file = 'psfs/dic_psf_60X_cy5_Scope5.pkl'
	master_data_folders = ['/data/07_22_2024__PFF_PTBP1']
	save_folder = 'output_old'
	iHm = 1 ; iHM = 16
	shape = (4,40,3000,3000)
	items = [(set_,ifov) for set_ in ['_set1'] for ifov in range(1,11)]
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
	hyb_deconvolver = Deconvolver(psfs, shape[1:], tile_size=300, overlap=89, zpad=39, beta=0.0001)
	dapi_deconvolver = Deconvolver(psfs, shape[1:], tile_size=300, overlap=89, zpad=13, beta=0.01)
	hyb_maxima = Maxima(threshold = 3600, delta = 1, delta_fit = 3,sigmaZ = 1, sigmaXY = 1.5)	
	#dapi_maxima = Maxima(threshold = 3, delta = 5, delta_fit = 5, sigmaZ = 1, sigmaXY = 1.5)	

	# the save file executor to do the saving in parallel with computations
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

	for cim in image_generator(hybs, fovs):
		print(cim.path)
		cim /= im_med[:, cp.newaxis, :, :] 
		for icol in [0,1,2]:
			# there is probably a better way to do the Xh stacking
			Xhf = list()
			for x,y,tile,raw in hyb_deconvolver.tile_wise(cim[icol], im_raw=True):
				tile_norm = utils.norm_image(tile)
				#Xh = hyb_maxima.apply(tile_norm, im_raw=raw)
				Xh = find_local_maxima(tile_norm, 3600.0, 1, 3, sigmaZ = 1, sigmaXY = 1.5, raw = raw)
				print(cp.max(Xh, axis=0))
				exit()
				keep = cp.all(Xh[:,1:3] < 300+89, axis=-1)
				keep &= cp.all(Xh[:,1:3] >= 89, axis=-1)
				Xh = Xh[keep]
				Xh[:,1] += x - 89
				Xh[:,2] += y - 89
				if len(Xh):
					Xhf.append(Xh)
			Xhf = cp.vstack(Xhf)
			executor.submit(save_data, save_folder, cim.path, icol, Xhf)
		# the local maxima for the dapi is not finished
		#cim[-1] = dapi_deconvolver.apply(cim[-1])
		#Xhf = find_local_maxima(cim[-1].astype(cp.float32), 3.0)





