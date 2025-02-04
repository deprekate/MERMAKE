import os
import re
from pathlib import Path
import glob
from wcmatch import glob as wc
from collections import Counter
from natsort import natsorted
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans

import numpy as np
from sklearn.cluster import KMeans

def find_two_means(values):
	values = np.abs(values).reshape(-1, 1)  # Reshape for clustering
	kmeans = KMeans(n_clusters=3, n_init="auto").fit(values)
	cluster_centers = kmeans.cluster_centers_
	return sorted(cluster_centers.flatten())[:2]


from scipy.spatial import KDTree
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

def read_xml(path):
	# Open and parse the XML file
	tree = None
	with open(path, "r", encoding="ISO-8859-1") as f:
		tree = ET.parse(f)
	return tree.getroot()

def get_xml_field(file, field):
	xml = read_xml(file)
	return xml.find(f".//{field}").text

def get_xy(args):
	group = args.config['codebooks'][0]
	pattern = group['hyb_pattern']
	files = list()
	for folder in group['hyb_folders']:
		#regex_path = os.path.join(folder, 'H([0-9]|1[0-6])_AER*', '*xml').replace('(','@(')
		regex_path = os.path.join(folder, pattern, '*xml').replace('(','@(')
		files.extend(wc.glob(regex_path, flags = wc.EXTGLOB))
	counts = Counter(re.search(pattern, file).group().split('_set')[0] for file in files if re.search(pattern, file))
	hybrid_count = {key: counts[key] for key in natsorted(counts)}
	hybrid,_ = max(hybrid_count.items(), key=lambda x: x[1])
	xmls = [file for file in files if hybrid in file]
	xmls = sorted(xmls, key=lambda x: os.path.basename(x))
	points = [tuple(map(float, get_xml_field(xml, 'stage_position').split(','))) for xml in xmls]
	#print(np.array(points))
	coords = points_to_coords(points)
	return coords


def set_data(args):
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
			for hyb,data in natsorted(batch[sset][fov].items()):
				path = data['zarr']
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

if __name__ == "__main__":
	# wcmatch requires ?*+@ before the () group pattern 
	print(regex_path)
	print(files)




