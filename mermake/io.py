import os
import re
import gc
import glob
from fnmatch import fnmatch
from types import SimpleNamespace
from typing import List, Tuple, Optional
import json
import argparse
from argparse import Namespace

import xml.etree.ElementTree as ET
import zarr
import cupy as cp
import numpy as np

import concurrent.futures

from . import blur
from . import __version__

def center_crop(A, shape):
	"""Crop numpy array to (h, w) from center."""
	h, w = shape[-2:]
	H, W = A.shape
	top = (H - h) // 2
	left = (W - w) // 2
	return A[top:top+h, left:left+w]

def load_flats(flat_field_tag, shape=None, **kwargs):
	stack = list()
	files = sorted(glob.glob(flat_field_tag + '*'))
	for file in files:
		im = np.load(file)['im']
		if shape is not None:
			im = center_crop(im, shape)
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

def get_ifov(zarr_file_path):
	"""Extract ifov from filename - finds last digits before .zarr"""
	filename = Path(zarr_file_path).name  # Keep full filename with extension
	match = re.search(r'([0-9]+)[^0-9]*\.zarr', filename)
	if match:
		return int(match.group(1))
	raise ValueError(f"No digits found before .zarr in filename: {filename}")

class FolderFilter:
	def __init__(self, hyb_range: str, regex_pattern: str):
		self.hyb_range = hyb_range
		self.regex = re.compile(regex_pattern)
		self.start_pattern, self.end_pattern = self.hyb_range.split(':')
		
		# Parse start and end patterns
		self.start_parts = self._parse_pattern(self.start_pattern)
		self.end_parts = self._parse_pattern(self.end_pattern)
		
	def _parse_pattern(self, pattern: str) -> Optional[Tuple]:
		"""Parse a pattern using the regex to extract components"""
		match = self.regex.match(pattern)
		if match:
			return match.groups()
		return None
	
	def _extract_numeric_part(self, text: str) -> int:
		"""Extract numeric part from text like 'H1' -> 1"""
		match = re.search(r'\d+', text)
		return int(match.group()) if match else 0
	
	def _compare_patterns(self, file_parts: Tuple, start_parts: Tuple, end_parts: Tuple) -> bool:
		"""
		Compare if file_parts falls within the range defined by start_parts and end_parts
		Groups: (prefix, number, middle, set_number, suffix)
		"""
		if not all([file_parts, start_parts, end_parts]):
			return False
			
		# Extract components
		file_prefix, file_num, file_middle, file_set, file_suffix = file_parts
		start_prefix, start_num, start_middle, start_set, start_suffix = start_parts
		end_prefix, end_num, end_middle, end_set, end_suffix = end_parts
		
		# Convert to integers for comparison
		file_num = int(file_num)
		file_set = int(file_set)
		start_num = int(start_num)
		start_set = int(start_set)
		end_num = int(end_num)
		end_set = int(end_set)
	
		# Check if middle part matches (e.g., 'MER')
		if start_middle == '*':
			pass
		elif file_middle != start_middle or file_middle != end_middle:
			return False
			
		# Check if prefix matches
		if file_prefix != start_prefix or file_prefix != end_prefix:
			return False
			
		num_in_range = start_num <= file_num <= end_num
		set_in_range = start_set <= file_set <= end_set
		
		return num_in_range and set_in_range
	
	def isin(self, text: str) -> bool:
		"""Check if a single text/filename falls within the specified range"""
		file_parts = self._parse_pattern(text)
		if not file_parts:
			return False
		return self._compare_patterns(file_parts, self.start_parts, self.end_parts)
	
	def filter_files(self, filenames: List[str]) -> List[str]:
		"""Filter filenames that fall within the specified range"""
		matching_files = []
		
		for filename in filenames:
			if self.isin(filename):
				matching_files.append(filename)
				
		return matching_files

class ImageQueue:
	__version__ = __version__
	def __init__(self, args):
		self.args = args
		self.__dict__.update(vars(args.paths))

		os.makedirs(self.output_folder, exist_ok = True)
		
		self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

		folder_filter = FolderFilter(self.hyb_range, self.regex)
		
		fov_min, fov_max = (-float('inf'), float('inf'))
		if hasattr(self, "fov_range"):
			fov_min, fov_max = map(int, self.fov_range.split(':'))
	
		matches = list()
		for root in self.hyb_folders:
			if not os.path.exists(root):
				continue
			try:
				with os.scandir(root) as entries:
					for sub in entries:
						if sub.is_dir(follow_symlinks=False) and folder_filter.isin(sub.name):
							try:
								with os.scandir(sub.path) as items:
									for item in items:
										if item.is_dir(follow_symlinks=False) and '.zarr' in item.name:
											ifov = get_ifov(str(item.name))
											if fov_min <= ifov <= fov_max:
												matches.append(item.path)
							except PermissionError:
								continue
			except PermissionError:
				continue
		
		self.files = iter(sorted(matches))

		# Preload the first valid image
		self._first_image = self._load_first_valid_image()
		self.shape = self._first_image.shape
		self.dtype = self._first_image.dtype
		
		# Only redo analysis if it is true
		if hasattr(self, "redo") and not self.redo:
			# Filter out already processed files
			filtered = [f for f in sorted(matches) if not self._is_done(f)]
			# Reload first valid image from sorted list
			self.files = iter(filtered)
			self._first_image = self._load_first_valid_image()

		# Start prefetching the next image
		self.future = None
		self._prefetch_next_image()

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

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()

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


	def close(self):
		self.executor.shutdown(wait=True)

	def path_parts(self, path):
		path_obj = Path(path)
		fov = path_obj.stem  # The filename without extension
		tag = path_obj.parent.name  # The parent directory name (which you seem to want)
		return fov, tag

	def save_hyb(self, path, icol, Xhf, attempt=1, max_attempts=3):
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
			args = namespace_to_array(self.args.settings)
			xp.savez_compressed(filepath, Xh=Xhf, version=__version__, args=args)
			#  Optional integrity check after saving
			try:
				with np.load(filepath) as dat:
					_ = dat["Xh"].shape  # Try accessing a key
			except Exception as e:
				os.remove(filepath)
				if attempt < max_attempts:
					return self.save_hyb(path, icol, Xhf, attempt=attempt+1, max_attempts=max_attempts)
				else:
					raise RuntimeError(f"Failed saving xfit file after {max_attempts} attempts: {filepath}")
		del Xhf
		if xp == cp:
			xp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool

	def save_dapi(self, path, icol, Xh_plus, Xh_minus, attempt=1, max_attempts=3):
		fov, tag = self.path_parts(path)
		filename = self.dapi_save.format(fov=fov, tag=tag, icol=icol)
		filepath = os.path.join(self.output_folder, filename)
	
		xp = cp.get_array_module(Xh_plus)
		if not os.path.exists(filepath) or (hasattr(self, "redo") and self.redo):
			args = namespace_to_array(self.args.settings)
			xp.savez_compressed(filepath, Xh_plus=Xh_plus, Xh_minus=Xh_minus, version=__version__, args=args)
			#  Optional integrity check after saving
			try:
				with np.load(filepath) as dat:
					_ = dat["Xh_minus"].shape  # Try accessing a key
			except Exception as e:
				os.remove(filepath)
				if attempt < max_attempts:
					return self.save_dapi(path, icol, Xh_plus, Xh_minus, attempt=attempt+1, max_attempts=max_attempts)
				else:
					raise RuntimeError(f"Failed saving xfit file after {max_attempts} attempts: {filepath}")
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

def dict_to_namespace(d):
	"""Recursively convert dictionary into SimpleNamespace."""
	for key, value in d.items():
		if isinstance(value, dict):
			value = dict_to_namespace(value)
		elif isinstance(value, list):
			value = [dict_to_namespace(i) if isinstance(i, dict) else i for i in value]
		d[key] = value
	return SimpleNamespace(**d)
def namespace_to_dict(obj):
	"""Recursively convert namespace objects to dictionaries"""
	if isinstance(obj, argparse.Namespace):
		return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
	elif isinstance(obj, list):
		return [namespace_to_dict(item) for item in obj]
	elif isinstance(obj, dict):
		return {k: namespace_to_dict(v) for k, v in obj.items()}
	else:
		return obj

def namespace_to_array(obj, prefix=''):
	"""
	Recursively convert Namespace or dict to list of (block, key, value) tuples.
	prefix is the accumulated parent keys joined by dots.
	"""
	rows = []
	if isinstance(obj, (Namespace, SimpleNamespace)):
		obj = vars(obj)
	if isinstance(obj, dict):
		for k, v in obj.items():
			full_key = f"{prefix}.{k}" if prefix else k
			if isinstance(v, (Namespace, SimpleNamespace, dict)):
				rows.extend(namespace_to_array(v, prefix=full_key))
			else:
				rows.append((prefix, k, str(v)))
	else:
		# For other types just append
		rows.append((prefix, '', str(obj)))
	return rows
