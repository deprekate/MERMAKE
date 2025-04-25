import os
import sys
import glob
import argparse
import contextlib
from time import sleep,time
from typing import Generator
# Try to import the appropriate TOML library
if sys.version_info >= (3, 11):
	import tomllib  # Python 3.11+ standard library
else:
	import tomli as tomllib  # Backport for older Python versions
from types import SimpleNamespace
import concurrent.futures

def dict_to_namespace(d):
	"""Recursively convert dictionary into SimpleNamespace."""
	for key, value in d.items():
		if isinstance(value, dict):
			value = dict_to_namespace(value)  # Recursively convert nested dictionaries
		elif isinstance(value, list):
			# Handle lists of dicts
			value = [dict_to_namespace(item) if isinstance(item, dict) else item for item in value] 
		d[key] = value
	return SimpleNamespace(**d)

# Validator and loader for the TOML file
def is_valid_file(path):
	if not os.path.exists(path):
		raise argparse.ArgumentTypeError(f"{path} does not exist.")
	if path.endswith('.zarr'):
		return
	try:
		with open(path, "rb") as f:
			config = tomllib.load(f)
			return config
	except Exception as e:
		raise argparse.ArgumentTypeError(f"Error loading TOML file {path}: {e}")

class CustomArgumentParser(argparse.ArgumentParser):
	def error(self, message):
		# Customizing the error message
		if "the following arguments are required: config" in message:
			message = message.replace("config", "config.toml")
		super().error(message)


# put this first to make sure to capture the correct gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change "1" to the desired GPU ID
import cupy as cp
cp.cuda.Device(0).use() # The above export doesnt always work so force CuPy to use GPU 0
import numpy as np
import cv2

from mermake.utils import set_data
from mermake.deconvolver import Deconvolver
#from mermake.maxima import find_local_maxima
from other.maxima import find_local_maxima
from mermake.io import image_generator, save_data, save_data_dapi, get_files
import mermake.blur as blur

def load_flats(tag):
	from more_itertools import peekable
	im_med = list()
	files = glob.glob(tag + '*')
	items = peekable(sorted(files))
	for file in items:
		med = np.array(np.load(file)['im'],dtype=np.float32)
		if items.peek(default=None) is not None:
			# the dapi flat field is not blurred
			med = cv2.blur(med,(20,20))
		med = med / np.median(med)
		im_med.append(med)
	return cp.asarray(np.stack(im_med))

if __name__ == "__main__":
	'''
	usage = '%s [-opt1, [-opt2, ...]] config.toml' % __file__
	parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser = CustomArgumentParser(description='',formatter_class=argparse.RawTextHelpFormatter,usage=usage)
	parser.add_argument('config', type=is_valid_file, help='config file')
	#parser.add_argument('-c', '--check', action="store_true", help="Check a single zarr")
	args = parser.parse_args()
	set_data(args)
	print(args)
	exit()
	'''
	# this is all stuff that will eventually be replaced with a toml settings file
	#----------------------------------------------------------------------------#
	#----------------------------------------------------------------------------#
	flat_field_tag = 'flat_field/Scope5_med_col_raw'
	psf_file = 'psfs/dic_psf_60X_cy5_Scope5.pkl'
	master_data_folders = ['/data/07_22_2024__PFF_PTBP1']
	save_folder = 'output_newnew'
	iHm = 1 ; iHM = 16
	shape = (4,40,3000,3000)
	tile_size = 500
	overlap = 89
	items = [(set_,ifov) for set_ in ['_set1'] for ifov in range(1,13)]
	#----------------------------------------------------------------------------#
	#----------------------------------------------------------------------------#
	hybs = list()
	fovs = list()
	for item in items:
		all_flds,fov = get_files(master_data_folders, item, iHm=iHm, iHM=iHM)
		hybs.append(all_flds)
		fovs.append(fov)
	im_med = load_flats(flat_field_tag)

	# eventually make a smart psf loader method to handle the different types of psf files
	psfs = np.load(psf_file, allow_pickle=True)

	# this mimics the behavior if there is only a single psf
	key = (0,1500,1500)
	psfs = { key : psfs[key] }
	
	# these can be very large objects in gpu ram, adjust accoringly to suit gpu specs
	hybs_deconvolver = Deconvolver(psfs, shape[1:], tile_size=tile_size, overlap=overlap, zpad=39, beta=0.0001)
	dapi_deconvolver = Deconvolver(psfs, shape[1:], tile_size=tile_size, overlap=overlap-20, zpad=19, beta=0.01)

	# the save file executor to do the saving in parallel with computations
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

	# this is a buffer to use for copying into 
	buffer = cp.empty(shape[1:], dtype=cp.float32)	

	#from other.io import stream_based_prefetcher
	for cim in image_generator(hybs, fovs):
		print(cim[0].path, flush=True)
		for icol in [0,1,2]:
			# there is probably a better way to do the Xh stacking
			Xhf = list()
			view = cim[icol]
			flat = im_med[icol]
			for x,y,tile,raw in hybs_deconvolver.tile_wise(view, flat, blur_radius=30):
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
			del view, Xhf, Xh
			cp._default_memory_pool.free_all_blocks()

		cp._default_memory_pool.free_all_blocks()
		# now do dapi, but first clear some stuff from gpu ram to fit in 12GB
		view = cim[3]
		flat = im_med[3]
		# Deconvolve in-place into the buffer
		dapi_deconvolver.apply(view, flat_field=flat, blur_radius=50, output=buffer)
		# the dapi channel is further normalized by the stdev
		std_val = float(cp.asnumpy(cp.linalg.norm(buffer.ravel()) / cp.sqrt(buffer.size)))
		cp.divide(buffer, std_val, out=buffer)
		Xh_plus = find_local_maxima(buffer, 3.0, 5, 5, sigmaZ = 1, sigmaXY = 1.5, raw = view )
		cp.multiply(buffer, -1, out=buffer)
		Xh_minus = find_local_maxima(buffer, 3.0, 5, 5, sigmaZ = 1, sigmaXY = 1.5, raw = view )
		cp.cuda.runtime.deviceSynchronize()
		executor.submit(save_data_dapi, save_folder, view.path, icol, Xh_plus, Xh_minus)
		view.clear()
		del cim, view, Xh_plus, Xh_minus
		cp._default_memory_pool.free_all_blocks()



