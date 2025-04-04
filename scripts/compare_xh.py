import sys

import numpy as np
from scipy.spatial import cKDTree

def count_shared_points(array1, array2, tolerance=1e-5):
    tree = cKDTree(array2)  # Build a KD-tree for array2
    count = np.sum(tree.query(array1, distance_upper_bound=tolerance)[0] < tolerance)
    return count

if len(sys.argv) < 3:
    raise ValueError("usage: python compare_xh.py FILE1 FILE2")

xh0 = np.load(sys.argv[1])['Xh']
print(xh0.shape)
print(xh0)
xh1 = np.load(sys.argv[2])['Xh']
print(xh1.shape)

shared = count_shared_points(xh0[:,:3], xh1[:,:3], tolerance=3)
print(shared)






