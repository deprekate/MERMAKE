import os
import re
import glob
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import cupy as cp

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
	from scipy.spatial import KDTree
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
	group = args.config['codebooks'][0]
	pattern = group['hyb_pattern']
	batch = dict()
	files = list()
	# parse hybrid folders
	for folder in group['hyb_folders']:
		regex_path = os.path.join(folder, pattern, '[0-9][0-9][0-9]').replace('(','@(')
		files.extend(wc.glob(regex_path, flags = wc.EXTGLOB))
	for file in files:
		sset = re.search('_set[0-9]*', file).group()
		hyb = re.search(pattern, file).group()
		if sset and hyb:
			batch.setdefault(sset, {}).setdefault(os.path.basename(file), {})[hyb] = {'zarr' : file}
	# parse xml files
	points = list()
	for sset in sorted(batch):
		for fov in sorted(batch[sset]):
			point = list()
			for hyb,dic in natsorted(batch[sset][fov].items()):
				path = dic['zarr']
				dirname = os.path.dirname(path)
				basename = os.path.basename(path)
				file = glob.glob(os.path.join(dirname,'*' + basename + '.xml'))[0]
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

def norm_image(image, ksize=30):
	xp = cp.get_array_module(image)
	if xp == np:
		from scipy.ndimage import convolve
	else:
		from cupyx.scipy.ndimage import convolve

	image = image.astype(xp.float32)  # Ensure correct type
	kernel = xp.ones((ksize, ksize), dtype=xp.float32) / (ksize * ksize)

	# Apply blur to each slice in parallel without looping in Python
	#padded = np.pad(image, ((1,0) ,(1, 0), (1, 0)), mode='reflect')
	#blurred = convolve(padded, kernel[None, :, :], mode="constant")
	#return image - blurred [0:image.shape[0], 0:image.shape[1], 0:image.shape[2]]
	blurred = convolve(image, kernel[None, :, :], mode="reflect")
	return image - blurred

# Create kernel once to reuse across norm_image calls
class Utils:
	def __init__(self, ksize=30, xp=cp):
		self.ksize = ksize
		self.kernel = cp.ones((ksize, ksize), dtype=cp.float32) / (ksize * ksize)
		if xp == np:
			from scipy.ndimage import convolve
		else:
			from cupyx.scipy.ndimage import convolve
		self.convolve = convolve	
	def norm_image(self, image):
		xp = cp.get_array_module(image)
		image = image.astype(xp.float32)  # Ensure correct type
		blurred = self.convolve(image, self.kernel[None, :, :], mode="reflect")
		return image - blurred


if __name__ == "__main__":
	# wcmatch requires ?*+@ before the () group pattern 
	print(regex_path)
	print(files)




