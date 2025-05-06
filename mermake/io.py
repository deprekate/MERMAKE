import os
import re
import gc
import glob


import xml.etree.ElementTree as ET
import zarr
from dask import array as da
import cupy as cp
import numpy as np

import concurrent.futures

def find_files(hyb_folders, hyb_range, **kwargs):
	import re
	# I guess brute force the matching
	# Regular expression to match the filename pattern
	pattern = r'(\w+)(\d+)_([^_]+)_set(\d+)'
	# Split the range into start and end parts
	start, end = hyb_range.split(':')
	# Match the start and end using regex
	match_start = re.match(pattern, start)
	match_end = re.match(pattern, end)
	# Extract the components from the matches
	start_letter, start_prefix, start_middle, start_set = match_start.groups()
	end_letter, end_prefix, end_middle, end_set = match_end.groups()
	# Convert the numeric parts to integers for the range generation
	start_num = int(start_prefix)
	end_num = int(end_prefix)
	start_set = int(start_set)
	end_set = int(end_set)
	
	# Generate the list of acceptable names
	names = list()
	for i in range(start_num, end_num + 1):
		for j in range(start_set, end_set + 1):
			name = f'{start_letter}{i}_{start_middle}_set{j}'
			names.append(name)
	names = set(names)

	# Iterate over the zarrs to see which match the previous names
	# should we look for zarrs or perhaps xmls?
	matches = list()
	for path in hyb_folders:
		files = glob.glob(os.path.join(path,'*','*.zarr'))
		for file in files:
			dirname = os.path.basename(os.path.dirname(file))
			if dirname in names:
				matches.append(file)
	return matches

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
	im = read_im(path)
	cim = cp.asarray(im)
	container = Container(cim)
	container.path = path
	return container

class ImageQueue:
    def __init__(self, files):
        self.files = iter(files)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Preload the first image
        try:
            first_file = next(self.files)
        except StopIteration:
            raise ValueError("No image files provided.")

        future = self.executor.submit(read_cim, first_file)
        self._first_image = future.result()
        self.shape = self._first_image.shape
        self.dtype = self._first_image.dtype

        # Start prefetching the next image
        self.future = None
        try:
            next_file = next(self.files)
            self.future = self.executor.submit(read_cim, next_file)
        except StopIteration:
            pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._first_image is not None:
            image = self._first_image
            self._first_image = None
            return image

        if self.future is None:
            raise StopIteration

        image = self.future.result()

        # Prefetch the next image
        try:
            next_file = next(self.files)
            self.future = self.executor.submit(read_cim, next_file)
        except StopIteration:
            self.future = None

        return image

    def close(self):
        self.executor.shutdown(wait=True)

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
def save_data(save_folder, path, icol, Xhf, **kwargs):
	fov,tag = path_parts(path)
	save_fl = save_folder + os.sep + fov + '--' + tag + '--col' + str(icol) + '__Xhfits.npz'
	os.makedirs(save_folder, exist_ok = True)
	cp.savez_compressed(save_fl, Xh=Xhf)
	del Xhf
def save_data_dapi(save_folder, path, icol, Xh_plus, Xh_minus, **kwargs):
	fov, tag = path_parts(path)
	save_fl = os.path.join(save_folder, f"{fov}--{tag}--dapiFeatures.npz")
	os.makedirs(save_folder, exist_ok=True)
	cp.savez_compressed(save_fl, Xh_plus=Xh_plus, Xh_minus=Xh_minus)
	del Xh_plus, Xh_minus

from .utils import *
def read_xml(path):
	# Open and parse the XML file
	tree = None
	with open(path, "r", encoding="ISO-8859-1") as f:
		tree = ET.parse(f)
	return tree.getroot()

def get_xml_field(file, field):
	xml = read_xml(file)
	return xml.find(f".//{field}").text
def set_data(args):
	from wcmatch import glob as wc
	from natsort import natsorted
	pattern = args.paths.hyb_range
	batch = dict()
	files = list()
	# parse hybrid folders
	files = find_files(**vars(args.paths))
	for file in files:
		sset = re.search('_set[0-9]*', file).group()
		hyb = os.path.basename(os.path.dirname(file))
		#hyb = re.search(pattern, file).group()
		if sset and hyb:
			batch.setdefault(sset, {}).setdefault(os.path.basename(file), {})[hyb] = {'zarr' : file}
	# parse xml files
	points = list()
	for sset in sorted(batch):
		for fov in sorted(batch[sset]):
			point = list()
			for hyb,dic in natsorted(batch[sset][fov].items()):
				path = dic['zarr']
				#file = glob.glob(os.path.join(dirname,'*' + basename + '.xml'))[0]
				file = path.replace('zarr','xml')
				point.append(list(map(float, get_xml_field(file, 'stage_position').split(','))))
			mean = np.mean(np.array(point), axis=0)
			batch[sset][fov]['stage_position'] = mean
			points.append(mean)
	points = np.array(points)
	mins = np.min(points, axis=0)
	step = estimate_step_size(points)
	#coords = points_to_coords(points)
	for sset in sorted(batch):
		for i,fov in enumerate(sorted(batch[sset])):
			point = batch[sset][fov]['stage_position']
			point -= mins
			batch[sset][fov]['grid_position'] = np.round(point / step).astype(int)
	args.batch = batch
	#counts = Counter(re.search(pattern, file).group().split('_set')[0] for file in files if re.search(pattern, file))
	#hybrid_count = {key: counts[key] for key in natsorted(counts)}

