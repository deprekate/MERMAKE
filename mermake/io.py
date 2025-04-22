import os
import gc
import glob

import zarr
from dask import array as da
import cupy as cp
import numpy as np


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

def read_im(path, return_pos=False):
    dirname = os.path.dirname(path)
    fov = os.path.basename(path).split('_')[-1].split('.')[0]
    file_ = os.path.join(dirname, fov, 'data')

    # Force eager loading from Zarr
    z = zarr.open(file_, mode='r')
    image = np.array(z[1:])  # use np.array(), not np.asarray()

    shape = image.shape
    xml_file = os.path.splitext(path)[0] + '.xml'
    if os.path.exists(xml_file):
        txt = open(xml_file, 'r').read()
        tag = '<z_offsets type="string">'
        zstack = txt.split(tag)[-1].split('</')[0]

        tag = '<stage_position type="custom">'
        x, y = eval(txt.split(tag)[-1].split('</')[0])

        nchannels = int(zstack.split(':')[-1])
        nzs = (shape[0] // nchannels) * nchannels
        image = image[:nzs].reshape([shape[0] // nchannels, nchannels, shape[-2], shape[-1]])
        image = image.swapaxes(0, 1)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) ** 2

    if return_pos:
        return image, x, y
    return image


class Container:
	def __init__(self, data, **kwargs):
		# Store the array and any additional metadata
		self.data = data
		self.metadata = kwargs
	def __getitem__(self, item):
		# Allow indexing into the container
		return self.data[item]
	def __array__(self):
		# Return the underlying array
		return self.data
	def __repr__(self):
		# Custom string representation showing the metadata or basic info
		return f"Container(shape={self.data.shape}, dtype={self.data.dtype}, metadata={self.metadata})"
	def __getattr__(self, name):
		# If attribute is not found on the container, delegate to the CuPy object
		if hasattr(self.data, name):
			return getattr(self.data, name)
		raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
	def clear(self):
		# Explicitly delete the CuPy array and synchronize
		if hasattr(self, 'data') and self.data is not None:
			del self.data
			self.data = None
def read_cim(path):
	""" store channels as separate objects so tey can be sequentially deleted from ram"""
	im = read_im(path)  # shape: (n_channels, z, y, x)
	channel_containers = []
	for icol in range(im.shape[0]):
		chan = cp.asarray(im[icol])
		container = Container(chan)
		container.path = path
		container.channel = icol
		channel_containers.append(container)
	return channel_containers  # List[Container]

#def read_cim(path):
#	im = read_im(path)
#	cim = cp.asarray(im)
#	container = Container(cim)
#	container.path = path
#	return container

import concurrent.futures
def image_generator(hybs, fovs):
	"""Generator that prefetches the next image while processing the current one."""
	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
		future = None
		for all_flds, fov in zip(hybs, fovs):
			for hyb in all_flds:
				file = os.path.join(hyb, fov)
				next_future = executor.submit(read_cim, file)
				if future:
					yield future.result()
				future = next_future
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
def save_data_dapi(save_folder, path, icol, Xh_plus, Xh_minus):
	fov, tag = path_parts(path)
	save_fl = os.path.join(save_folder, f"{fov}--{tag}--dapiFeatures.npz")
	os.makedirs(save_folder, exist_ok=True)
	cp.savez_compressed(save_fl, Xh_plus=Xh_plus, Xh_minus=Xh_minus)
	del Xh_plus, Xh_minus

