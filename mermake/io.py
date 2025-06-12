import os
import re
import gc
import glob


import xml.etree.ElementTree as ET
import zarr
import cupy as cp
import numpy as np

import concurrent.futures

from . import blur

def load_flats(flat_field_tag, **kwargs):
	stack = list()
	files = sorted(glob.glob(flat_field_tag + '*'))
	for file in files:
		im = np.load(file)['im']
		cim = cp.array(im,dtype=cp.float32)
		blurred = blur.box(cim, (20,20))
		blurred = blurred / cp.median(blurred)
		stack.append(blurred)
	return cp.stack(stack)

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

	z = zarr.open(file_, mode='r')
	image = np.array(z[1:])

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
		image = image.astype(np.uint16) ** 2

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

def read_ccim(path, return_pos=False):
	dirname = os.path.dirname(path)
	fov = os.path.basename(path).split('_')[-1].split('.')[0]
	file_ = os.path.join(dirname, fov, 'data')

	z = zarr.open(file_, mode='r')
	nz = z.shape[0]

	# Skip z[0], start at z[1]
	slices = []
	for i in range(1, nz):
		slices.append(cp.asarray(z[i]))

	image = cp.stack(slices)
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
		image = image[:nzs].reshape((shape[0] // nchannels, nchannels, shape[-2], shape[-1]))
		image = cp.swapaxes(image, 0, 1)

	if image.dtype == cp.uint8:
		image = image.astype(cp.uint16) ** 2

	container = Container(image)
	container.path = path

	if return_pos:
		return container, x, y
	return container

def get_ifov(filepath):
	basename = os.path.basename(filepath)
	matches = []
	for m in re.finditer(r'\d+', basename):
		start, end = m.span()
		before = basename[start - 1] if start > 0 else ''
		after = basename[end] if end < len(basename) else ''
		if not before.isalpha() and not after.isalpha():
			matches.append(m.group())
	assert len(matches) == 1
	return int(matches[0])

class ImageQueue:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		os.makedirs(self.output_folder, exist_ok = True)
		# I guess brute force the matching
		# Regular expression to match the filename pattern
		pattern = re.compile(self.regex, re.IGNORECASE)
		# Split the range into start and end parts
		start, end = self.hyb_range.split(':')
		# Match the start and end using regex
		match_start = pattern.match(start)
		match_end = pattern.match(end)
		# Extract the components from the matches
		start_letter, start_prefix, start_middle, start_set, start_extra = match_start.groups()
		end_letter, end_prefix, end_middle, end_set, start_extra = match_end.groups()
		# Convert the numeric parts to integers for the range generation
		start_num = int(start_prefix)
		end_num = int(end_prefix)
		start_set = int(start_set)
		end_set = int(end_set)
		
		# Generate the list of acceptable names
		names = list()
		for i in range(start_num, end_num + 1):
			for j in range(start_set, end_set + 1):
				name = f'{start_letter}{i}_{start_middle}_set{j}{start_extra}'
				names.append(name)
		self.names = set(names)
	
		self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

		# Iterate over the zarrs to see which match the previous names
		# should we look for zarrs or perhaps xmls?
		self.matches = self._find_matching_files()
		self.matches.sort()
		self.files = iter(self.matches)

		# Preload the first valid image
		self._first_image = self._load_first_valid_image()
		self.shape = self._first_image.shape
		self.dtype = self._first_image.dtype

		# Only redo analysis if it is true
		if hasattr(self, "redo") and not self.redo:
			# Filter out already processed files
			filtered = [f for f in self.files if not self._is_done(f)]
			# Reload first valid image from sorted list
			self.files = iter(filtered)
			self._first_image = self._load_first_valid_image()

		# Start prefetching the next image
		self.future = None
		self._prefetch_next_image()

	def _is_done(self, path):
		fov, tag = self.path_parts(path)
		for icol in range(self.shape[0] - 1):
			filename = self.hyb_save.format(fov=fov, tag=tag, icol=icol)
			filepath = os.path.join(self.output_folder, filename)
			if not os.path.exists(filepath):
				return False
		filename = self.dapi_save.format(fov=fov, tag=tag, icol=icol)
		filepath = os.path.join(self.output_folder, filename)
		if not os.path.exists(filepath):
			return False
		return True

	def _prefetch_next_image(self):
		try:
			next_file = next(self.files)
			self.future = self.executor.submit(read_cim, next_file)
		except StopIteration:
			self.future = None

	def __iter__(self):
		return self

	def __next__(self):
		if self._first_image is not None:
			image = self._first_image
			self._first_image = None
			return image
		while self.future is not None:
			try:
				image = self.future.result()
			except Exception as e:
				#print(f"Warning: Failed to load image: {e}")
				image = None
			# Try to prefetch the next image regardless
			try:
				next_file = next(self.files)
				self.future = self.executor.submit(read_cim, next_file)
			except StopIteration:
				self.future = None
			# If we got a valid image, return it
			if image is not None:
				return image

		# If we reach here, there are no more images in the current batch
		if False:
			# In watch mode, look for new files
			import time
			time.sleep(60)
			
			# Find any new files
			new_matches = self._find_matching_files()
			# Filter to only files we haven't processed yet
			new_matches = [f for f in new_matches if f not in self.processed_files]
			
			if new_matches:
				# New files found!
				new_matches.sort()
				self.matches = new_matches
				self.files = iter(self.matches)
				self.processed_files.update(new_matches)  # Mark as seen
				
				# Prefetch the first new image
				self._prefetch_next_image()
				
				# Try again to get the next image
				return self.__next__()
			else:
				# No new files yet, but we'll keep watching
				return self.__next__()

		self.close()
		raise StopIteration

	def _load_first_valid_image(self):
		"""Try loading images one by one until one succeeds."""
		for file in self.files:
			try:
				future = self.executor.submit(read_cim, file)
				image = future.result()
				return image
			except Exception as e:
				#print(f"Warning: Failed to load image {file}: {e}")
				continue
		raise RuntimeError("No valid images could be found.")

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()

	def close(self):
		self.executor.shutdown(wait=True)
	
	def _find_matching_files(self):
		"""Find all matching files across all hyb_folders, by checking which existing folders match self.names"""
		matches = []

		# Limit infiles to be within the range of fovs if given
		if hasattr(self, "fov_range"):
			ifov_min, ifov_max = map(int,self.fov_range.split(':'))
		else:
			ifov_min, ifov_max = -float('Inf') , float('Inf')
		for base_path in self.hyb_folders:
			# Get all immediate subdirectories in the base path
			try:
				subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
				
				# Filter subdirectories to only those that are in self.names
				matching_dirs = [d for d in subdirs if d in self.names]
				
				# For each matching directory, find the zarr files
				for dirname in matching_dirs:
					zarr_files = glob.glob(os.path.join(base_path, dirname, '*.zarr'))
					for zarr_file in sorted(zarr_files):
						ifov = get_ifov(zarr_file)
						if ifov_min <= ifov <= ifov_max:
							matches.append(zarr_file)
			except (FileNotFoundError, PermissionError) as e:
				print(f"Warning: Could not access directory {base_path}: {e}")
				continue
		return matches

	def path_parts(self, path):
		path_obj = Path(path)
		fov = path_obj.stem  # The filename without extension
		tag = path_obj.parent.name  # The parent directory name (which you seem to want)
		return fov, tag

	def save_hyb(self, path, icol, Xhf):
		fov,tag = self.path_parts(path)
		filename = self.hyb_save.format(fov=fov, tag=tag, icol=icol)
		filepath = os.path.join(self.output_folder, filename)
	
		Xhf = [x for x in Xhf if x.shape[0] > 0]
		if Xhf:
			xp = cp.get_array_module(Xhf[0])
			Xhf = xp.vstack(Xhf)
		else:
			xp = np
			Xhf = np.array([])
		if not os.path.exists(filepath) or (hasattr(self, "redo") and self.redo):
			xp.savez_compressed(filepath, Xh=Xhf)
		del Xhf
		if xp == cp:
			xp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool

	def save_dapi(self, path, icol, Xh_plus, Xh_minus):
		fov, tag = self.path_parts(path)
		filename = self.dapi_save.format(fov=fov, tag=tag, icol=icol)
		filepath = os.path.join(self.output_folder, filename)
	
		xp = cp.get_array_module(Xh_plus)
		if not os.path.exists(filepath) or (hasattr(self, "redo") and self.redo):
			xp.savez_compressed(filepath, Xh_plus=Xh_plus, Xh_minus=Xh_minus)
		del Xh_plus, Xh_minus
		if xp == cp:
			xp._default_memory_pool.free_all_blocks()


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


def save_data_dapi(path, icol, Xh_plus, Xh_minus, output_folder, **kwargs):
	xp = cp.get_array_module(Xh_plus)
	fov, tag = path_parts(path)
	filepath = os.path.join(output_folder, f"{fov}--{tag}--dapiFeatures.npz")
	os.makedirs(output_folder, exist_ok=True)
	xp.savez_compressed(filepath, Xh_plus=Xh_plus, Xh_minus=Xh_minus)
	del Xh_plus, Xh_minus
	if xp == cp:
		xp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool

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

