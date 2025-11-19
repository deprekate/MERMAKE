import os
import re
import gc
import glob
from pathlib import Path
from fnmatch import fnmatch
from types import SimpleNamespace
from typing import List, Tuple, Optional, Union
import json
import argparse
from argparse import Namespace
import functools
import concurrent.futures
import queue
import threading
from itertools import chain
from collections import defaultdict

import xml.etree.ElementTree as ET
import dask.array as da
import cupy as cp
import numpy as np

from . import blur
from . import __version__

import logging
logging.basicConfig(
	format="{asctime} - {levelname} - {message}",
	style="{",
	datefmt="%Y-%m-%d %H:%M",
)

def center_crop(A, shape):
	"""Crop numpy array to (h, w) from center."""
	h, w = shape[-2:]
	H, W = A.shape[-2:]
	top = (H - h) // 2
	left = (W - w) // 2
	return A[top:top+h, left:left+w]

def load_flats(flat_field_tag, shape=None, **kwargs):
	stack = list()
	files = sorted(glob.glob(flat_field_tag + '*'))
	for file in files:
		im = np.load(file)['im']
		assert im.shape[-1] >= shape[-1] and im.shape[-2] >= shape[-2], 'flat field smaller than image'
		if shape is not None:
			im = center_crop(im, shape)
		cim = cp.array(im,dtype=cp.float32)
		blurred = blur.box(cim, (20,20), axes=(-1,-2))
		blurred = blurred / cp.median(blurred)
		stack.append(blurred)
	return cp.stack(stack)

class Container:
	def __init__(self, path_or_array):
		if isinstance(path_or_array, str):
			# Load from file
			im = read_im(path_or_array)
			# Always split first axis into channels
			self.data = [Container(im[i]) for i in range(im.shape[0])]
			self.path = path_or_array
		else:
			# Already an array, just store it
			self.data = path_or_array
			self.path = None

	def __getitem__(self, idx):
		return self.data[idx]

	def __repr__(self):
		if isinstance(self.data, list):
			shapes = [getattr(ch.data, "shape", None) if isinstance(ch, Container) else getattr(ch, "shape", None)
					  for ch in self.data]
			return f"Container(path={self.path}, shapes={shapes})"
		else:
			return f"Container(shape={getattr(self.data, 'shape', None)})"

	def __array__(self, dtype=None):
		"""
		Return the underlying array for NumPy/CuPy interop.
		Called automatically when passed to np.array() or cupy.set().
		"""
		if isinstance(self.data, list):
			raise ValueError("Cannot convert a multi-channel Container to a single array")
		arr = self.data
		if dtype is not None:
			arr = arr.astype(dtype)
		return arr

	def __eq__(self, other):
		if isinstance(other, Container):
			return self.path == other.path
		elif isinstance(other, str):
			return self.path == other
		return False

	def __hash__(self):
		return hash(self.path)


	def compute(self):
		"""Compute only the current container (recursively if channels)."""
		if isinstance(self.data, list):
			for i, ch in enumerate(self.data):
				self.data[i] = ch.compute()  # recurse into sub-containers
			return self
		elif isinstance(self.data, da.Array):
			self.data = self.data.compute()
			return self
		else:
			return self


def read_im(path, return_pos=False):
	dirname = os.path.dirname(path)
	fov = os.path.basename(path).split('_')[-1].split('.')[0]
	file_ = os.path.join(dirname, fov, 'data')

	#z = zarr.open(file_, mode='r')
	#image = np.array(z[1:])
	from dask import array as da
	image = da.from_zarr(file_)[1:]

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

def get_ih(file_path):
	basename = os.path.basename(file_path)
	m = re.search(r'\d+', basename)  # first contiguous digits
	if m:
		return int(m.group())
	return 10**100
	#raise ValueError(f"No number found in {basename}")

def get_ifov(file_path):
	"""Extract ifov from filename - finds last digits before .zarr"""
	filename = Path(file_path).name  # Keep full filename with extension
	match = re.search(r'([0-9]+)[^0-9]*\.zarr', filename)
	if match:
		return int(match.group(1))
	raise ValueError(f"No digits found before .zarr in filename: {filename}")

def get_iset(file_path):
	"""
	Recursively extract 'iset' from filename or parent directories.
	Finds last digits after the word '_set'.
	"""
	path = Path(file_path)  # ensure Path object
	filename = path.name
	match = re.search(r'_set([0-9]+)', filename)
	if match:
		return int(match.group(1))

	# Base case: reached root
	if path.parent == path:
		raise ValueError(f"No digits found after the word _set in {file_path}")

	# Recurse up
	return get_iset(path.parent)

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
		
	def _parse_pattern(self, haystack: str) -> Optional[Tuple]:
		"""Parse a pattern using the regex to extract components"""
		match = self.regex.match(haystack)
		if match:
			return match.groups()
		return None

	def _undo_regex(self, groups: Union[Tuple, re.Match, None]) -> Optional[str]:
		"""
		Reconstruct a string from captured groups based on the stored regex pattern.
		If `groups` is a re.Match it will use match.groups().
		If you pass fewer groups than the regex has, reconstruction stops right
		after the last provided group (no extra literal text appended).
		"""
		if groups is None:
			return None

		# Accept either a tuple-of-groups or a Match object.
		if hasattr(groups, "groups"):
			groups = groups.groups()
		groups = tuple(groups)

		pattern = self.regex.pattern
		parts = re.split(r'\([^)]*\)', pattern)  # literal chunks between capture groups

		rebuilt = []
		for i, g in enumerate(groups):
			literal = parts[i] if i < len(parts) else ''
			rebuilt.append(literal)
			rebuilt.append('' if g is None else str(g))

		# Append the trailing literal only if all groups were provided.
		if len(groups) == len(parts) - 1:
			rebuilt.append(parts[len(groups)])

		return ''.join(rebuilt)

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
	def add(self, image):
		ifov = image.ifov
		if not self or self.ifov() == ifov:
			if self:
				del self[-1].data
			self.append(image)
			return True
		else:
			return False
	def iset(self):
		return self[0].iset
	def ifov(self):
		return self[0].ifov
	def __repr__(self):
		paths = [image.path for image in self]
		return f"Block({paths})"
	
class ImageQueue:
	__version__ = __version__
	def __init__(self, args, prefetch_count=5):
		self.args = args
		self.args_array = namespace_to_array(self.args.settings)
		self.__dict__.update(vars(args.paths))

		os.makedirs(self.output_folder, exist_ok = True)
		
		fov_min, fov_max = (-float('inf'), float('inf'))
		if hasattr(self, "fov_range"):
			fov_min, fov_max = map(float, self.fov_range.split(':'))
		
		self.background_files = set()
		background = None
		if hasattr(self, "background_range"):
			filtered = FolderFilter(self.background_range, self.regex, fov_min, fov_max)
			background = filtered.get_matches(self.hyb_folders)
			self.background_files = set(chain.from_iterable(background.values()))
		filtered = FolderFilter(self.hyb_range, self.regex, fov_min, fov_max)
		matches = filtered.get_matches(self.hyb_folders)

		# do output summary, eventually replace with terminal gui
		counts = defaultdict(lambda: defaultdict(int))
		rounds = dict()
		for iset,ifov in matches:
			for path in matches[(iset,ifov)]:
				base = os.path.basename(os.path.dirname(path))
				parts = list(filtered._parse_pattern(base))
				parts = parts[:3]
				undone = filtered._undo_regex(parts)
				counts[iset][undone] += 1
				rounds[undone] = True
		rounds = sorted(rounds)
		width = max(len(r) for r in rounds) + 1
		summary = ''
		summary += 'SUMMARY'.center(len(rounds) * width + width, '-')	
		summary += '\n'
		summary += 'set'.ljust(width)
		for undone in rounds:
			summary += undone.rjust(width)
		summary += '\n'
		for iset in counts:
			summary += str(iset).ljust(width)
			for undone in rounds:
				summary += str(counts[iset][undone]).rjust(width)
			summary += '\n'
		summary += ''.center(len(rounds) * width + width, '-')
		self.summary = summary
			
		# Peek at first image to set shape/dtype
		first_image = None
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
	
		# interlace the background with the regular images
		shared = set(matches.keys()).intersection(background.keys()) if background else matches.keys()

		interlaced = []
		for key in shared:
			if background and key in background:
				interlaced.extend(background[key])  # put background first
			hsorted = self.hsorted(matches[key])
			interlaced.extend(hsorted)				# then all matches for that iset,ifov

		self.files = iter(interlaced)

		# Start worker thread(s)
		self.queue = queue.Queue(maxsize=prefetch_count)
		self.stop_event = threading.Event()
		self.thread = threading.Thread(target=self._worker, daemon=True)
		self.thread.start()

	def hsorted(self, files):
		return sorted(files, key=lambda f: get_ih(os.path.dirname(f)))

	def containerize(self, path):
		# everythign in this method is done async and prefetched

		container = Container(path)

		# check if the fov has been fitted
		# we should test whether the load worked!!!!!!!!!!
		for icol in range(self.shape[0] - 1):
			filename = self.get_name(path, icol)
			filepath = os.path.join(self.output_folder, filename)
			# only compute if the image has not been processed or if it is background
			if not os.path.exists(filepath) or container in self.background_files:
				container[icol].compute()
			else:
				Xh = cp.load(filepath)['Xh']
				setattr(container, f'col{icol}', Xh)
		icol = -1
		filename = self.get_name(path, icol)
		filepath = os.path.join(self.output_folder, filename)
		if not os.path.exists(filepath):
			container[icol].compute()
		else:
			container.Xh_plus = cp.load(filepath)['Xh_plus']
			container.Xh_minus = cp.load(filepath)['Xh_minus']
		# attach drift file to container
		iset = get_iset(path)
		ifov = get_ifov(path)
		#filename = self.drift_save.format(ifov=ifov, iset=iset)
		filename = self.get_name(path)
		filepath = os.path.join(self.output_folder, filename)
		if os.path.exists(filepath):
			drifts, files, ref, ref_path = cp.load(filepath, allow_pickle=True)
			container.drifts = drifts
			container.drift_files = files
			container.drift_ref = ref
			container.drift_ref_path = ref_path
		container.iset = iset
		container.ifov = ifov
		return container

	def _worker(self):
		"""Continuously read images and put them in the queue."""
		for path in self.files:
			if self.stop_event.is_set():
				break
			try:
				container = self.containerize(path)
				self.queue.put(container)
			except Exception as e:
				print(f"Warning: failed to read {path}: {e}")
				#dummy = lambda : None
				#dummy.path = path
				#self.queue.put(dummy)
				#self.queue.put(False)
				continue
		# Signal no more images
		container.last = True
		self.queue.put(None)

	def __iter__(self):
		return self

	def __next__(self):
		img = self.queue.get()
		if img is None:
			raise StopIteration
		return img
	
	'''
		# this code is so we can eventually add a --watch parameter so mermake keeps
		# runnning after processing all fovs and then processes any new fovs that appear
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

	'''
	def close(self):
		self.stop_event.set()
		self.thread.join()
	'''
	def close(self):
		self.stop_event.set()
    
		# Drain the queue so the worker thread can finish
		try:
			while True:
				self.queue.get_nowait()
		except queue.Empty:
			pass
    
		# Use a timeout to avoid hanging indefinitely
		self.thread.join(timeout=2.0)
    
		if self.thread.is_alive():
			print("Warning: worker thread did not terminate", flush=True)

	def get_name(self, path, icol=None):
		path_obj = Path(path)
		fov = path_obj.stem	        # the filename without extension
		tag = path_obj.parent.name  # the parent directory name
		ifov = get_ifov(path)
		iset = get_iset(path)
		if icol is None:
			return self.drift_save.format(ifov=ifov, iset=iset)
		elif icol == -1:
			return self.dapi_save.format(fov=fov, tag=tag)
		else:
			return self.hyb_save.format(fov=fov, tag=tag, icol=icol)


	def _is_fitted(self, path):
		for icol in range(self.shape[0] - 1):
			#filename = self.hyb_save.format(fov=fov, tag=tag, icol=icol)
			filename = self.get_name(path, icol)
			filepath = os.path.join(self.output_folder, filename)
			if not os.path.exists(filepath):
				return False
		filename = self.get_name(path, -1)
		filepath = os.path.join(self.output_folder, filename)
		if not os.path.exists(filepath):
			return False
		return True

	def save_xfits(self, image, icol=-1, attempt=1, max_attempts=3):
		
		# determine filename and data based on icol
		if icol == -1:
			data = {'Xh_plus': image.Xh_plus,'Xh_minus': image.Xh_minus,}
		else:
			Xh = getattr(image, f'col{icol}')
			data = {'Xh': Xh}
		
		# save if file doesn't exist
		path = image.path
		filename = self.get_name(path, icol)
		filepath = os.path.join(self.output_folder, filename)
		if not os.path.exists(filepath):
			cp.savez_compressed(filepath, version=__version__, args=self.args_array, **data)
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

def read_xml(path):
	# Open and parse the XML file
	tree = None
	with open(path, "r", encoding="ISO-8859-1") as f:
		tree = ET.parse(f)
	return tree.getroot()

def get_xml_field(file, field):
	xml = read_xml(file)
	return xml.find(f".//{field}").text

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
