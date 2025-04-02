import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change "1" to the desired GPU ID

import time
import gc
import cupy as cp
import numpy as np
from time import sleep
import glob
import cv2
import zarr
from dask import array as da

from mermake.maxima import Maxima
from mermake.deconvolver import Deconvolver
from mermake.utils import norm_image
from mermake.maxima_gpu import find_local_maxima
import dask.array as da

class DaskArrayWithMetadata:
	def __init__(self, dask_array, path):
		self.dask_array = dask_array
		self.path = path

	def __getattr__(self, attr):
		# This will forward any unknown attributes to the Dask array
		return getattr(self.dask_array, attr)

	def __repr__(self):
		return f"<DaskArrayWithMetadata(shape={self.dask_array.shape}, path={self.path})>"


def read_im(path,return_pos=False):
	dirname = os.path.dirname(path)
	fov = os.path.basename(path).split('_')[-1].split('.')[0]
	file_ = dirname+os.sep+fov+os.sep+'data'
	image = da.from_zarr(file_)[1:]

	shape = image.shape
	#nchannels = 4
	xml_file = os.path.dirname(path)+os.sep+os.path.basename(path).split('.')[0]+'.xml'
	if os.path.exists(xml_file):
		txt = open(xml_file,'r').read()
		tag = '<z_offsets type="string">'
		zstack = txt.split(tag)[-1].split('</')[0]
		
		tag = '<stage_position type="custom">'
		x,y = eval(txt.split(tag)[-1].split('</')[0])
		
		nchannels = int(zstack.split(':')[-1])
		nzs = (shape[0]//nchannels)*nchannels
		image = image[:nzs].reshape([shape[0]//nchannels,nchannels,shape[-2],shape[-1]])
		image = image.swapaxes(0,1)
	image = DaskArrayWithMetadata(image, path)
	if return_pos:
		return image,x,y
	return image

def get_iH(fld): return int(os.path.basename(fld).split('_')[0][1:])
def get_files(set_ifov,iHm=None,iHM=None):
	if not os.path.exists(save_folder): os.makedirs(save_folder)
	all_flds = []
	for master_folder in master_data_folders:
		all_flds += glob.glob(master_folder+os.sep+r'H*_AER_*')
		#all_flds += glob.glob(master_folder+os.sep+r'H*_Igfbpl1_Aldh1l1_Ptbp1*')
	### reorder based on hybe
	all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
	set_,ifov = set_ifov
	all_flds = [fld for fld in all_flds if set_ in os.path.basename(fld)]
	all_flds = [fld for fld in all_flds if ((get_iH(fld)>=iHm) and (get_iH(fld)<=iHM))]
	fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
	if not os.path.exists(fovs_fl):
		folder_map_fovs = all_flds[0]#[fld for fld in all_flds if 'low' not in os.path.basename(fld)][0]
		fls = glob.glob(folder_map_fovs+os.sep+'*.zarr')
		fovs = np.sort([os.path.basename(fl) for fl in fls])
		np.save(fovs_fl,fovs)
	else:
		fovs = np.sort(np.load(fovs_fl))
	fov=None
	if ifov<len(fovs):
		fov = fovs[ifov]
		all_flds = [fld for fld in all_flds if os.path.exists(fld+os.sep+fov)]
	return save_folder,all_flds,fov


import concurrent.futures
def image_generator(hybs, fovs):
	"""Generator that prefetches the next image while processing the current one."""
	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
		future = None  # Holds the future for the next image
		for all_flds, fov in zip(hybs, fovs):
			for hyb in all_flds:
				file = os.path.join(hyb, fov)

				# Submit the next image read operation
				next_future = executor.submit(read_im, file)
				# If there was a previous future, yield its result
				if future:
					yield future.result()
				# Move to the next future
				future = next_future

		# Yield the last remaining image
		if future:
			yield future.result()

from pathlib import Path
def path_parts(path):
	parts = Path(path).parts
	return Path(parts[-1]).stem , parts[-2] 
# Function to handle saving the file
def save_data(path, icol, Xhf):
	fov,tag = path_parts(path)
	save_fl = save_folder + os.sep + fov + '--' + tag + '--col' + str(icol) + '__Xhfits.npz'
	np.savez_compressed(save_fl, Xh=Xhf)
# Executor for threading
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)



if __name__ == "__main__":
	master_data_folders = ['/data/07_22_2024__PFF_PTBP1']
	save_folder = 'output_new'
	iHm = 1 ; iHM = 16

	shape = (4,40,3000,3000)
	
	items = [(set_,ifov) for set_ in ['_set1'] for ifov in range(1,11)]
	hybs = list()
	fovs = list()
	for item in items[:4]:
		save_folder,all_flds,fov = get_files(item, iHm=iHm, iHM=iHM)
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
	
	psf_file = 'psfs/dic_psf_60X_cy5_Scope5.pkl'
	psfs = np.load(psf_file, allow_pickle=True)
	hyb_deconvolver = Deconvolver(psfs, shape[1:], tile_size=300, overlap=89, zpad=39, beta=0.0001)
	#dapi_deconvolver = Deconvolver(psfs, shape[1:], tile_size=300, overlap=69, zpad=13, beta=0.01)
	hyb_maxima = Maxima(threshold = 3600, delta = 1, delta_fit = 3,sigmaZ = 1, sigmaXY = 1.5)	
	dapi_maxima = Maxima(threshold = 3, delta = 5, delta_fit = 5, sigmaZ = 1, sigmaXY = 1.5)	

	cim = cp.empty(shape, dtype=cp.float32)
	for im in image_generator(hybs, fovs):
		cim[...] = cp.asarray(im, dtype=cp.float32)
		cim /= im_med[:, cp.newaxis, :, :] 
		for icol in [0,1,2]:
			Xhf = list()
			for x,y,tile,raw in hyb_deconvolver.tile_wise(cim[icol], im_raw=True):
				tile_norm = norm_image(tile, ksize = 30)
				Xh = hyb_maxima.apply(tile_norm, im_raw=raw)
				keep = cp.all(Xh[:,1:3] < 300+89, axis=-1)
				keep &= cp.all(Xh[:,1:3] >= 89, axis=-1)
				Xh = Xh[keep]
				Xh[:,1] += x - 89
				Xh[:,2] += y - 89
				if len(Xh):
					Xhf.append(Xh)
			Xhf = cp.vstack(Xhf)
			executor.submit(save_data, im.path, icol, Xhf)
		del Xhf
		# the local maxima for the dapi is not finished
		#cim[-1] = dapi_deconvolver.apply(cim[-1])
		#Xhf = find_local_maxima(cim[-1].astype(cp.float32), 3.0)
		#del Xhf





