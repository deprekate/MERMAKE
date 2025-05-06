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

# put this first to make sure to capture the correct gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change "1" to the desired GPU ID
import cupy as cp
cp.cuda.Device(0).use() # The above export doesnt always work so force CuPy to use GPU 0
import numpy as np

#sys.path.pop(0)
from mermake.deconvolver import Deconvolver
from mermake.maxima import find_local_maxima
#from more.maxima import find_local_maxima
from mermake.io import image_generator, save_data, save_data_dapi, get_files, find_files
from mermake.io import ImageQueue
import mermake.blur as blur

def dict_to_namespace(d):
    """Recursively convert dictionary into SimpleNamespace."""
    for key, value in d.items():
        if isinstance(value, dict):
            value = dict_to_namespace(value)
        elif isinstance(value, list):
            value = [dict_to_namespace(i) if isinstance(i, dict) else i for i in value]
        d[key] = value
    return SimpleNamespace(**d)

# Validator for the TOML file
def is_valid_file(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"{path} does not exist.")
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)  # Return raw dict
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Error loading TOML file {path}: {e}")

toml_text = """
        [paths]
        codebook = '/home/katelyn/develop/MERMAKE/codebooks/codebook_code_color2__ExtraAaron_8_6_blank.csv' ###
        psf_file = '/home/katelyn/develop/MERMAKE/psfs/dic_psf_60X_cy5_Scope5.pkl'  ### Scope5 psf
        flat_field_tag = '/home/katelyn/develop/MERMAKE/flat_field/Scope5_'
        hyb_range = 'H1_AER_set1:H16_AER_set1'
        hyb_folders = [
                        '/data/07_22_2024__PFF_PTBP1',
                        ]
        output_folder = '/home/katelyn/develop/MERMAKE/MERFISH_Analysis_AER'
        
        #---------------------------------------------------------------------------------------#
        #---------------------------------------------------------------------------------------#
        #           you probably dont have to change any of the settings below                  #
        #---------------------------------------------------------------------------------------#
        #---------------------------------------------------------------------------------------#
        [hybs]
        tile_size = 300
        overlap = 89
        beta = 0.0001
        threshold = 3600
        blur_radius = 30
        delta = 1
        delta_fit = 3
        sigmaZ = 1
        sigmaXY = 1.5
        
        [dapi]
        tile_size = 300
        overlap = 89
        beta = 0.01
        threshold = 3.0
        blur_radius = 50
        delta = 5
        delta_fit = 5
        sigmaZ = 1
        sigmaXY = 1.5"""

class CustomArgumentParser(argparse.ArgumentParser):
	def error(self, message):
		# Customizing the error message
		if "the following arguments are required: settings" in message:
			message = message.replace("settings", "settings.toml")
		message += '\n'
		message += 'The format for the toml file is shown below'
		message += '\n'
		message += toml_text
		super().error(message)



def load_flats(flat_field_tag, **kwargs):
	stack = list()
	files = glob.glob(flat_field_tag + '*')
	for file in files:
		im = np.load(file)['im']
		cim = cp.array(im,dtype=cp.float32)
		blurred = blur.box(cim, (20,20))
		blurred = blurred / cp.median(blurred)
		stack.append(blurred)
	return cp.stack(stack)



if __name__ == "__main__":
	usage = '%s [-opt1, [-opt2, ...]] settings.toml' % __file__
	#parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser = CustomArgumentParser(description='',formatter_class=argparse.RawTextHelpFormatter,usage=usage)
	parser.add_argument('settings', type=is_valid_file, help='settings file')
	#parser.add_argument('-c', '--check', action="store_true", help="Check a single zarr")
	args = parser.parse_args()
	# Convert settings to namespace and attach each top-level section to args
	for key, value in vars(dict_to_namespace(args.settings)).items():
		setattr(args, key, value)
	#----------------------------------------------------------------------------#
	#----------------------------------------------------------------------------#
	flats = load_flats(**vars(args.paths))
	
	files = find_files(**vars(args.paths))
	# maybe do check here if files have already been processed
	if False:
		files = exclude_processed(file, **vars(args.paths)) 

	files = sorted(files)[1:100]

	# I still cant quite get the prefetcher 100% gpu util
	#from other.io import stream_based_prefetcher
	#queue = stream_based_prefetcher(files)
	queue = ImageQueue(files)

	# set some things based on input images
	shape = queue.shape
	ncol = shape[0]
	zpad = shape[1] - 1 # this needs to be about the same size as the input z depth

	# eventually make a smart psf loader method to handle the different types of psf files
	psfs = np.load(args.paths.psf_file, allow_pickle=True)

	# this mimics the behavior if there is only a single psf
	key = (0,1500,1500)
	psfs = { key : psfs[key] }
	
	# these can be very large objects in gpu ram, adjust accoringly to suit gpu specs
	hybs_deconvolver = Deconvolver(psfs, shape, zpad = zpad, **vars(args.hybs) )
	# shrink the zpad to limit the loaded psfs in ram since dapi isnt deconvolved as strongly
	dapi_deconvolver = Deconvolver(psfs, shape, zpad = zpad//2, **vars(args.dapi))

	# the save file executor to do the saving in parallel with computations
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

	# this is a buffer to use for copying into 
	buffer = cp.empty(shape[1:], dtype=cp.float32)	

	overlap = args.hybs.overlap
	tile_size = args.hybs.tile_size

	for image in queue:
		print(image.path, flush=True)
		for icol in range(ncol - 1):
			# there is probably a better way to do the Xh stacking
			Xhf = list()
			chan = image[icol]
			flat = flats[icol]
			for x,y,tile,raw in hybs_deconvolver.tile_wise(chan, flat, **vars(args.hybs)):
				Xh = find_local_maxima(tile, raw = raw, **vars(args.hybs))
				keep = cp.all((Xh[:,1:3] >= overlap) & (Xh[:,1:3] < tile_size + overlap), axis=-1)
				Xh = Xh[keep]
				Xh[:,1] += x - overlap
				Xh[:,2] += y - overlap
				Xhf.append(Xh)
			executor.submit(save_data, image.path, icol, Xhf, **vars(args.paths))
			del chan, Xhf # Xh

		# now do dapi
		chan = image[-1]
		flat = flats[-1]
		# Deconvolve in-place into the buffer
		dapi_deconvolver.apply(chan, flat_field=flat, output=buffer, **vars(args.dapi))
		# the dapi channel is further normalized by the stdev
		std_val = float(cp.asnumpy(cp.linalg.norm(buffer.ravel()) / cp.sqrt(buffer.size)))
		cp.divide(buffer, std_val, out=buffer)
		Xh_plus = find_local_maxima(buffer, raw = chan, **vars(args.dapi) )
		cp.multiply(buffer, -1, out=buffer)
		Xh_minus = find_local_maxima(buffer, raw = chan, **vars(args.dapi) )
		# save the data
		#executor.submit(save_data_dapi, image.path, icol, Xh_plus, Xh_minus, **vars(args.paths))
		image.clear()
		del chan, Xh_plus, Xh_minus, image


