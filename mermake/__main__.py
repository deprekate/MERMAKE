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
import concurrent.futures

import numpy as np
import dask.array as da

#import faulthandler, signal
#faulthandler.register(signal.SIGUSR1)

#sys.path.pop(0)

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
        codebook = 'codebook_code_color2__ExtraAaron_8_6_blank.csv' ###
        psf_file = 'dic_psf_60X_cy5_Scope5.pkl'  ### Scope5 psf
        flat_field_tag = 'Scope5_'
        hyb_range = 'H1_AER_set1:H16_AER_set1'
        hyb_folders = [
                        '/data/07_22_2024__PFF_PTBP1',
                        ]
        output_folder = 'MERFISH_Analysis_AER'
        redo = false
        
        #---------------------------------------------------------------------------------------#
        #---------------------------------------------------------------------------------------#
        #           you probably dont have to change any of the settings below                  #
        #---------------------------------------------------------------------------------------#
        #---------------------------------------------------------------------------------------#
        hyb_save = '{fov}--{tag}--col{icol}.npz'
        dapi_save = '{fov}--{tag}--dapiFeatures.npz'
        regex = '''([A-z]+)(\d+)_([^_]+)_set(\d+)(.*)''' #use triple quotes to avoid double escape
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

def view_napari(queue, deconvolver, args ):
	import cupy as cp
	#cp.cuda.Device(0).use() # The above export doesnt always work so force CuPy to use GPU 0
	from mermake.deconvolver import Deconvolver
	from mermake.maxima import find_local_maxima
	from mermake.io import load_flats
	from mermake.io import ImageQueue, dict_to_namespace

	image = next(queue)
	buffer = deconvolver.buffer
	flats = deconvolver.flats
	import napari
	viewer = napari.Viewer()
	color = ['red','green','blue', 'white']
	ncol = queue.shape[0]
	for icol in range(ncol-1):
		chan = cp.asarray(image[icol].compute())
		flat = flats[icol]
		deconvolver.hybs.apply(chan, flat_field=flat, output=buffer, blur_radius=None)
		deco = cp.asnumpy(buffer)
		deconvolver.hybs.apply(chan, flat_field=flat, output=buffer, **vars(args.dapi))
		Xh = find_local_maxima(buffer, raw = chan, **vars(args.hybs))
		norm = cp.asnumpy(buffer)
		# Stack 3D images: original, deco, norm
		stacked = np.stack([cp.asnumpy(chan), deco, norm], axis=0)
		viewer.add_image(stacked, name=f"channel {icol}", colormap=color[icol], blending='additive')
		# Add corresponding points for this stack
		points = cp.asnumpy(Xh[:, :3])
		viewer.add_points(points, size=7, border_color=color[icol],face_color='transparent', opacity=0.6, name=f"maxima {icol}")
	icol += 1
	chan = cp.asarray(image[icol].compute())
	flat = flats[icol]
	deconvolver.dapi.apply(chan, flat_field=flat, output=buffer, blur_radius=None)
	deco = cp.asnumpy(buffer)
	deconvolver.dapi.apply(chan, flat_field=flat, output=buffer, **vars(args.dapi))
	std_val = float(cp.asnumpy(cp.linalg.norm(buffer.ravel()) / cp.sqrt(buffer.size)))
	cp.divide(buffer, std_val, out=buffer)
	Xh_plus = find_local_maxima(buffer, raw = chan, **vars(args.dapi) )
	norm = cp.asnumpy(buffer)
	# Stack 3D images: original, deco, norm
	stacked = np.stack([cp.asnumpy(chan), deco, norm], axis=0)
	viewer.add_image(stacked, name="dapi",  colormap=color[icol], blending='additive')
	points = cp.asnumpy(Xh_plus[:, :3])
	viewer.add_points(points, size=11, border_color=color[icol], face_color='transparent', opacity=0.6, name=f"maxima dapi")
	napari.run()
	exit()


def print_clean(msg, last_len=[0]):
	# Clear previous line
	sys.stdout.write('\r' + ' ' * last_len[0] + '\r')
	sys.stdout.write(msg)
	sys.stdout.flush()
	last_len[0] = len(msg)  # Save new message length


def main():
	prog = sys.argv[0] if sys.argv[0] else "mermake"
	usage = f'{prog} [-opt1, [-opt2, ...]] settings.toml'
	#parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser = CustomArgumentParser(description='',formatter_class=argparse.RawTextHelpFormatter,usage=usage)
	parser.add_argument('settings', type=is_valid_file, help='settings file')
	parser.add_argument('-g', '--gpu', type=int, default=0, help="The gpu to use [0]")
	parser.add_argument('-c', '--check', action="store_true", help="Check a single zarr")
	args = parser.parse_args()

	# put this here to make sure to capture the correct gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	import cupy as cp
	#cp.cuda.Device(0).use() # The above export doesnt always work so force CuPy to use GPU 0
	from mermake.deconvolver import Deconvolver
	from mermake.maxima import find_local_maxima
	from mermake.io import load_flats
	from mermake.io import ImageQueue, dict_to_namespace, Block
	#from mermake.io import dict_to_namespace
	#from mermake.image_queue import ImageQueue

	from mermake.align import Aligner 
	import mermake.blur as blur
	from mermake.align import drift, drift_save

	# Convert settings to namespace and attach each top-level section to args
	for key, value in vars(dict_to_namespace(args.settings)).items():
		setattr(args, key, value)
	#----------------------------------------------------------------------------#
	#----------------------------------------------------------------------------#
	psfs = np.load(args.paths.psf_file, allow_pickle=True)

	# the save file executor to do the saving in parallel with computations
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=9)

	message = 'Finding input image files.'
	print_clean(message)

	with ImageQueue(args) as queue:
		print_clean(queue.summary)
		# set some things based on input images
		ncol,*chan_shape = queue.shape
		sz,sx,sy = chan_shape
		zpad = sz - 1 # this needs to be about the same size as the input z depth
		# this is a buffer to use for copying into 
		buffer = cp.empty(chan_shape, dtype=cp.float32)	
		chan = cp.empty(chan_shape, dtype=cp.uint16)	
	
		flats = load_flats(shape = queue.shape, **vars(args.paths))

		message = 'Loading PSFs into GPU ram.'
		print_clean(message)
		# these can be very large objects in gpu ram, adjust accoringly to suit gpu specs
		deconvolver = lambda : None
		deconvolver.hybs = Deconvolver(psfs, chan_shape, zpad = zpad, **vars(args.hybs) )
		# shrink the zpad to limit the loaded psfs in ram since dapi isnt deconvolved as strongly
		# or you could just use a single psf, ie (0,1500,1500)
		deconvolver.dapi = Deconvolver(psfs, chan_shape, zpad = zpad//2, **vars(args.dapi))
		block = Block()

		if args.check:
			deconvolver.buffer = buffer
			deconvolver.flats = flats
			view_napari(queue, deconvolver, args)

		overlap = args.hybs.overlap
		tile_size = args.hybs.tile_size

		message = 'Starting image processing.\n'
		print_clean(message)
		for image in queue:
			print('running on:', image.path, flush=True)
			icol = -1
			data = image[icol].data
			if not isinstance(data, da.Array):
				chan.set(data)
				flat = flats[icol]
				# Deconvolve in-place into the buffer
				deconvolver.dapi.apply(chan, flat_field=flat, output=buffer, **vars(args.dapi))
				# the dapi channel is further normalized by the stdev
				std_val = float(cp.asnumpy(cp.linalg.norm(buffer.ravel()) / cp.sqrt(buffer.size)))
				cp.divide(buffer, std_val, out=buffer)
				image.Xh_plus = find_local_maxima(buffer, raw = chan, **vars(args.dapi) )
				cp.multiply(buffer, -1, out=buffer)
				image.Xh_minus = find_local_maxima(buffer, raw = chan, **vars(args.dapi) )
				# save the data
				executor.submit(queue.save_dapi, image.path, icol, image.Xh_plus, image.Xh_minus)
		
				#if block.background:
				#	slices_back, slices_chan = aligner.get_shifted_slices( Xh_plus[:,:3], chan.shape )
		
			for icol in range(ncol - 1):
				data = image[icol].data
				if not isinstance(data, da.Array):
					chan.set(data)
					flat = flats[icol]
					# there is probably a better way to do the Xh stacking
					Xhf = []
					for x,y,tile,raw in deconvolver.hybs.tile_wise(chan, flat, **vars(args.hybs)):
						Xh = find_local_maxima(tile, raw = raw, **vars(args.hybs))
						keep = cp.all((Xh[:,1:3] >= overlap) & (Xh[:,1:3] < cp.array([tile.shape[1] - overlap, tile.shape[2] - overlap])), axis=-1)
						Xh = Xh[keep]
						Xh[:,1] += x - overlap
						Xh[:,2] += y - overlap
						Xhf.append(Xh)
					executor.submit(queue.save_hyb, image.path, icol, Xhf)
					del Xhf, Xh, keep
			# this block of images for a fov is over
			if not block.add(image) or hasattr(image, 'last'):
				if (result := drift(block, **vars(args.paths))):
					executor.submit(drift_save, *result)
					del result
				block.clear()
				block.add(image)
			del image
		executor.shutdown(wait=True)


if __name__ == "__main__":
	main()
