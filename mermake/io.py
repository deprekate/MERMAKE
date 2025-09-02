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
import functools
import concurrent.futures
import queue
import threading
from itertools import chain

import xml.etree.ElementTree as ET
import zarr
import cupy as cp
import numpy as np

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

class Container:
	def __init__(self, data, **kwargs):
		# Store the data and any additional metadata
		self.data = data
		self.metadata = kwargs
	def __getitem__(self, item):
		# Allow indexing into the container
		return self.data[item]
	def __array__(self):
		# Return the underlying array
		return self.data
	def __repr__(self):
		return f"Container(shape={self.data.shape}, dtype={self.data.dtype}, metadata={self.metadata})"
	def __getattr__(self, name):
		if name in self.metadata:
			return self.metadata[name]
		if hasattr(self.data, name):
			return getattr(self.data, name)
		raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def containerize(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		result = func(*args, **kwargs)
		metadata = dict(kwargs)
		if args:
			metadata['path'] = args[0]
		if isinstance(result, tuple):
			arr, *rest = result
			if len(rest) == 2:
				metadata['x'], metadata['y'] = rest
			return Container(arr, **metadata)
		else:
			return Container(result, **metadata)
	return wrapper

@containerize
def read_im(path, return_pos=False):
	dirname = os.path.dirname(path)
	fov = os.path.basename(path).split('_')[-1].split('.')[0]
	file_ = os.path.join(dirname, fov, 'data')

	z = zarr.open(file_, mode='r')
	image = np.array(z[1:])
	#from dask import array as da
	#image = da.from_zarr(file_)[1:]

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
def get_iset(zarr_file_path):
	"""Extract iset from filename - finds last digits after the word set"""
	filename = Path(zarr_file_path).name  # Keep full filename with extension
	match = re.search(r'_set([0-9]+)', filename)
	if match:
		return int(match.group(1))
	raise ValueError(f"No digits found after the word _set in filename: {filename}")

class FolderFilter:
	def __init__(self, hyb_range: str, regex_pattern: str, fov_min: float, fov_max: float):
		self.hyb_range = hyb_range
		self.regex = re.compile(regex_pattern)
		self.start_pattern, self.end_pattern = self.hyb_range.split(':')	
		self.fov_min = fov_min
		self.fov_max = fov_max
		
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

	def get_matches(self, folders):
		matches = dict()
		for root in folders:
			if not os.path.exists(root):
				continue
			try:
				with os.scandir(root) as entries:
					for sub in entries:
						if sub.is_dir(follow_symlinks=False) and self.isin(sub.name):
							try:
								with os.scandir(sub.path) as items:
									# we might need other ways to determine set
									iset = get_iset(str(sub.name))
									for item in items:
										if item.is_dir(follow_symlinks=False) and '.zarr' in item.name:
											ifov = get_ifov(str(item.name))
											if self.fov_min <= ifov <= self.fov_max:
												matches.setdefault((iset,ifov), []).append(item.path)

							except PermissionError:
								continue
			except PermissionError:
				continue
		return matches
class Block(list):
	def __init__(self, items=None):
		self.background = None
		if isinstance(items, (list, tuple)):
			for item in items:
				self.append(item)
		elif items:
			self.append(items)

class ImageQueue:
	__version__ = __version__
	def __init__(self, args, prefetch_count=6):
		self.args = args
		self.args_array = namespace_to_array(self.args.settings)
		self.__dict__.update(vars(args.paths))

		os.makedirs(self.output_folder, exist_ok = True)
		
		fov_min, fov_max = (-float('inf'), float('inf'))
		if hasattr(self, "fov_range"):
			fov_min, fov_max = map(float, self.fov_range.split(':'))
		matches = FolderFilter(self.hyb_range, self.regex, fov_min, fov_max).get_matches(self.hyb_folders)
		background = None
		if hasattr(self, "background_range"):
			background = FolderFilter(self.background_range, self.regex, fov_min, fov_max).get_matches(self.hyb_folders)
			self.background = True

		# Peek at first image to set shape/dtype
		for path in chain.from_iterable(matches.values()):
			try:
				first_image = read_im(path)
				break
			except:
				continue
		if first_image is None:
			raise RuntimeError("No valid images found.")
		self.shape = first_image.shape
		self.dtype = first_image.dtype
		
		# Filter out already processed files
		if hasattr(self, "redo") and not self.redo:
			new_matches = {}
			for key, files in matches.items():
				filtered = [f for f in files if not self._is_done(f)]
				if filtered:
					new_matches[key] = filtered
			matches = new_matches
		
		# interlace the background with the regular images
		shared = set(matches.keys()).intersection(background.keys()) if background else matches.keys()
		#matches = [item for key in shared for item in matches[key]]
		#background = [item for key in shared for item in background[key]] if background else None
		interlaced = []
		for key in shared:
			if background and key in background:
				interlaced.extend(background[key])  # put background first
			interlaced.extend(matches[key])		 # then all matches for that key
		
		#self.files = iter(sorted(matches))
		self.files = iter(interlaced)

		self.block = Block()
		# Start worker thread(s)
		self.queue = queue.Queue(maxsize=prefetch_count)
		self.stop_event = threading.Event()
		self.thread = threading.Thread(target=self._worker, daemon=True)
		self.thread.start()

	def _worker(self):
		"""Continuously read images and put them in the queue."""
		for path in self.files:
			if self.stop_event.is_set():
				break
			try:
				im = read_im(path)
				self.queue.put(im)  # Blocks if queue is full
			except Exception as e:
				print(f"Warning: failed to read {path}: {e}")
				#dummy = lambda : None
				#dummy.path = path
				#self.queue.put(dummy)
				self.queue.put(False)
				continue
		# Signal no more images
		self.queue.put(None)

	def __iter__(self):
		return self

	'''
	def __next__(self):
		img = self.queue.get()
		if img is None:
			raise StopIteration
		return img
	'''
	
	def __next__(self):
		"""Return the next block of images (same FOV)."""
		block = self.block
		first_item = self.queue.get()
		if first_item is None:
			raise StopIteration
		
		if hasattr(self, "background_range"):
			self.block.background = first_item
		else:
			block.append(first_item)
		ifov = None if first_item == False else get_ifov(first_item.path)

		# Keep consuming queue until FOV changes or None
		while True:
			item = self.queue.get()
			if item == False:
				break
			if item is None:
				self.queue.put(None)
				break
			if get_ifov(item.path) != ifov:
				if hasattr(self, "background_range"):
					self.block.background = item
				else:
					self.block = Block(item)
				break
			block.append(item)
		if hasattr(self, "background_range") and first_item == False:
			block.clear()	
		return block

	'''
	def __next__(self):
		"""Return the next block of images (same FOV)."""
		logger.debug(f"__next__ called, queue size: {getattr(self.queue, 'qsize', lambda: 'unknown')()}")
		
		block = Block()  # Fresh block each time
		
		# Get first item with debugging
		logger.debug("Getting first item from queue...")
		start_time = time.time()
		first_item = self.queue.get()
		get_time = time.time() - start_time
		logger.debug(f"Got first item in {get_time:.3f}s: {type(first_item).__name__}")
		
		if first_item is None:
			logger.debug("First item is None, raising StopIteration")
			raise StopIteration
		
		block.append(first_item)
		
		# Handle FOV logic with debugging
		if first_item == False:
			ifov = None
			logger.debug("First item is False, ifov = None")
		else:
			logger.debug(f"Getting ifov for: {getattr(first_item, 'path', 'NO_PATH_ATTR')}")
			ifov = get_ifov(first_item.path)
			logger.debug(f"ifov = {ifov}")
		
		# Keep consuming queue until FOV changes or None
		item_count = 1
		while True:
			logger.debug(f"Loop iteration {item_count}, getting next item...")
			start_time = time.time()
			item = self.queue.get()
			get_time = time.time() - start_time
			logger.debug(f"Got item {item_count} in {get_time:.3f}s: {type(item).__name__}")
			
			if item == False:
				logger.debug("Got False, breaking")
				break
			if item is None:
				logger.debug("Got None, putting back and breaking")
				self.queue.put(None)
				break
			
			logger.debug(f"Getting ifov for item {item_count}: {getattr(item, 'path', 'NO_PATH_ATTR')}")
			item_ifov = get_ifov(item.path)
			logger.debug(f"Item ifov: {item_ifov}, current ifov: {ifov}")
			
			if item_ifov != ifov:
				logger.debug("FOV changed, putting item back and breaking")
				self.queue.put(item)  # Put back the item with different FOV
				break
			
			block.append(item)
			item_count += 1
			logger.debug(f"Added item to block, block size now: {len(block)}")
		
		logger.debug(f"Returning block with {len(block)} items")
		return block
	'''

	'''
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
	'''

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.close()

	def close(self):
		self.stop_event.set()
		self.thread.join()

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
			xp.savez_compressed(filepath, Xh=Xhf, version=__version__, args=self.args_array)
			#  Optional integrity check after saving
			# this seems to greatly slow everything down
			#try:
			#	with np.load(filepath) as dat:
			#		_ = dat["Xh"].shape  # Try accessing a key
			#except Exception as e:
			#	os.remove(filepath)
			#	if attempt < max_attempts:
			#		return self.save_hyb(path, icol, Xhf, attempt=attempt+1, max_attempts=max_attempts)
			#	else:
			#		raise RuntimeError(f"Failed saving xfit file after {max_attempts} attempts: {filepath}")
		del Xhf
		if xp == cp:
			xp._default_memory_pool.free_all_blocks()  # Free standard GPU memory pool

	def save_dapi(self, path, icol, Xh_plus, Xh_minus, attempt=1, max_attempts=3):
		fov, tag = self.path_parts(path)
		filename = self.dapi_save.format(fov=fov, tag=tag, icol=icol)
		filepath = os.path.join(self.output_folder, filename)
	
		xp = cp.get_array_module(Xh_plus)
		if not os.path.exists(filepath) or (hasattr(self, "redo") and self.redo):
			xp.savez_compressed(filepath, Xh_plus=Xh_plus, Xh_minus=Xh_minus, version=__version__, args=self.args_array)
			#  Optional integrity check after saving
			# this seems to greatly slow everything down
			#try:
			#	with np.load(filepath) as dat:
			#		_ = dat["Xh_minus"].shape  # Try accessing a key
			#except Exception as e:
			#	os.remove(filepath)
			#	if attempt < max_attempts:
			#		return self.save_dapi(path, icol, Xh_plus, Xh_minus, attempt=attempt+1, max_attempts=max_attempts)
			#	else:
			#		raise RuntimeError(f"Failed saving xfit file after {max_attempts} attempts: {filepath}")
		del Xh_plus, Xh_minus
		if xp == cp:
			xp._default_memory_pool.free_all_blocks()

# Debugging wrapper for your main processing loop
def debug_processing_loop(queue):
	"""Wrapper to add debugging around your processing loop"""
	logger.info("Starting processing loop")
	block_count = 0
	
	try:
		for block in queue:
			block_count += 1
			logger.info(f"Processing block {block_count} with {len(block)} images")
			
			for i, image in enumerate(block):
				logger.debug(f"Processing image {i+1}/{len(block)} in block {block_count}")
				
				# Time the GPU operations
				start_time = time.time()
				cim = cp.asarray(image)  # This might be where it stalls
				asarray_time = time.time() - start_time
				logger.debug(f"cp.asarray took {asarray_time:.3f}s")
				
				start_time = time.time()
				gpu_compute(cim)  # Or this might be where it stalls
				compute_time = time.time() - start_time
				logger.debug(f"gpu_compute took {compute_time:.3f}s")
				
			logger.info(f"Completed block {block_count}")
			
	except Exception as e:
		logger.error(f"Error in processing loop at block {block_count}: {e}", exc_info=True)
	logger.info(f"Processing loop completed, processed {block_count} blocks")


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
