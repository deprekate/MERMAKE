import os
import re
import glob
from pathlib import Path
from collections import Counter

import numpy as np
import cupy as cp

def fftconvolve(a, b, mode='full'):
	# Compute the size for FFT in each dimension
	shape = [sa + sb - 1 for sa, sb in zip(a.shape, b.shape)]

	# Perform FFTs (n-dimensional)
	fa = cp.fft.fftn(a, shape)
	fb = cp.fft.fftn(b, shape)

	# Multiply in frequency domain and inverse FFT
	out = cp.fft.ifftn(fa * fb).real

	# Handle cropping modes
	if mode == 'full':
		return out
	elif mode == 'same':
		start = [(o - s) // 2 for o, s in zip(out.shape, a.shape)]
		end = [start[i] + a.shape[i] for i in range(len(start))]
		slices = tuple(slice(start[i], end[i]) for i in range(len(start)))
		return out[slices]
	elif mode == 'valid':
		valid_shape = [sa - sb + 1 for sa, sb in zip(a.shape, b.shape)]
		start = [sb - 1 for sb in b.shape]
		end = [start[i] + valid_shape[i] for i in range(len(start))]
		slices = tuple(slice(start[i], end[i]) for i in range(len(start)))
		return out[slices]
	else:
		raise ValueError("mode must be 'full', 'same', or 'valid'")


def profile():
	import gc
	mempool = cp.get_default_memory_pool()
	# Loop through all objects in the garbage collector
	for obj in gc.get_objects():
		if isinstance(obj, cp.ndarray):
			# Check if it's a view (not a direct memory allocation)
			if obj.base is not None:
				# Skip views as they do not allocate new memory
				continue
			print(f"CuPy array with shape {obj.shape} and dtype {obj.dtype}")
			print(f"Memory usage: {obj.nbytes / 1024**2:.2f} MB")  # Convert to MB
	print(f"Used memory after: {mempool.used_bytes() / 1024**2:.2f} MB")

class Config:
	def __init__(self, args):
		self.args = args
		self.some_data = 'foo'

def find_two_means(values):
	from sklearn.cluster import KMeans
	values = np.abs(values).reshape(-1, 1)  # Reshape for clustering
	kmeans = KMeans(n_clusters=3, n_init="auto").fit(values)
	cluster_centers = kmeans.cluster_centers_
	return sorted(cluster_centers.flatten())[:2]

def estimate_step_size(points):
	points = np.array(points)
	# Build a KD-tree for efficient nearest neighbor search
	tree = KDTree(points)
	# Find the distance to the nearest neighbor for each point
	distances, _ = tree.query(points, k=2)  # k=2 because first result is the point itself
	nearest_dists = distances[:, 1]  # Extract nearest neighbor distances (skip self-distance)
	# Use the median to ignore outliers (or mode if step size is very regular)
	step_size = np.median(nearest_dists)  # More robust than mean
	return step_size

def points_to_coords(points):
	'convert xy point locations to integer grid coordinates'
	points = np.array(points)
	points -= np.min(points, axis=0)
	#_,mean = find_two_means(shifts)
	mean = estimate_step_size(points)
	coords = np.round(points / mean).astype(int)
	return coords


def count_bits(args):
	group = args.config['codebooks'][0]
	with open(group['codebook_path'], 'r') as fp:
		return next(fp).rstrip().count('bit')

def count_colors(args):
	batch = args.batch
	sset = next(iter(batch))
	fov = next(iter(batch[sset]))
	hyb = next(iter(batch[sset][fov]))
	dic = batch[sset][fov][hyb]

	path = dic['zarr']
	dirname = os.path.dirname(path)
	basename = os.path.basename(path)
	file = glob.glob(os.path.join(dirname,'*' + basename + '.xml'))[0]
	colors = get_xml_field(file, 'z_offsets').split(':')[-1]
	return int(colors)

def count_hybs(args):
	bits = count_bits(args)
	colors = count_colors(args) - 1
	num = bits / colors
	print(num)
	print(args)

if __name__ == "__main__":
	# wcmatch requires ?*+@ before the () group pattern 
	print(regex_path)
	print(files)




