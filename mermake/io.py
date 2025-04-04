import os
import gc
import glob

import zarr
from dask import array as da
import cupy as cp
import numpy as np

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
	#image = DaskArrayWithMetadata(image, path)
	if return_pos:
		return image,x,y
	return image

class cupy_asarray(cp.ndarray):
	''' this class is so metadata can be wrapped into a cupy array'''
	def __new__(cls, array, path=None, dtype=None):
		if dtype is None:
			dtype = cp.dtype(array.dtype)
		obj = cp.asarray(array, dtype=dtype).view(cls)
		obj.path = path
		return obj
	def __array_finalize__(self, obj):
		if obj is None: return
		self.path = getattr(obj, 'path', None)
def read_cim(path):
	im = read_im(path)
	cim = cupy_asarray(im, path, dtype=cp.float32)
	return cim

def get_iH(fld): return int(os.path.basename(fld).split('_')[0][1:])
def get_files(master_data_folders, set_ifov,iHm=None,iHM=None):
	#if not os.path.exists(save_folder): os.makedirs(save_folder)
	all_flds = []
	for master_folder in master_data_folders:
		all_flds += glob.glob(master_folder+os.sep+r'H*_AER_*')
		#all_flds += glob.glob(master_folder+os.sep+r'H*_Igfbpl1_Aldh1l1_Ptbp1*')
	### reorder based on hybe
	all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
	set_,ifov = set_ifov
	all_flds = [fld for fld in all_flds if set_ in os.path.basename(fld)]
	all_flds = [fld for fld in all_flds if ((get_iH(fld)>=iHm) and (get_iH(fld)<=iHM))]
	#fovs_fl = save_folder+os.sep+'fovs__'+set_+'.npy'
	folder_map_fovs = all_flds[0]#[fld for fld in all_flds if 'low' not in os.path.basename(fld)][0]
	fls = glob.glob(folder_map_fovs+os.sep+'*.zarr')
	fovs = np.sort([os.path.basename(fl) for fl in fls])
	fov = fovs[ifov]
	all_flds = [fld for fld in all_flds if os.path.exists(fld+os.sep+fov)]
	return all_flds,fov


import concurrent.futures
def image_generator(hybs, fovs):
	"""Generator that prefetches the next image while processing the current one."""
	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
		future = None  # Holds the future for the next image
		for all_flds, fov in zip(hybs, fovs):
			for hyb in all_flds:
				file = os.path.join(hyb, fov)

				# Submit the next image read operation
				next_future = executor.submit(read_cim, file)
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
    path_obj = Path(path)
    fov = path_obj.stem  # The filename without extension
    tag = path_obj.parent.name  # The parent directory name (which you seem to want)
    return fov, tag

# Function to handle saving the file
def save_data(save_folder, path, icol, Xhf):
	fov,tag = path_parts(path)
	save_fl = save_folder + os.sep + fov + '--' + tag + '--col' + str(icol) + '__Xhfits.npz'
	os.makedirs(save_folder, exist_ok = True)
	cp.savez_compressed(save_fl, Xh=Xhf)
	del Xhf





