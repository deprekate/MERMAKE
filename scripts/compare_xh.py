import sys

import numpy as np
from scipy.spatial import cKDTree

def count_shared_points(array1, array2, tolerance=1e-5):
    tree = cKDTree(array2)  # Build a KD-tree for array2
    count = np.sum(tree.query(array1, distance_upper_bound=tolerance)[0] < tolerance)
    return count

def sort_Xh(xh):
    """
    Sort a 2D array by the first three columns.
    """
    # Use lexsort, which sorts by last key first, so reverse the order
    sort_idx = np.lexsort((xh[:,2], xh[:,1], xh[:,0]))
    return xh[sort_idx]

if len(sys.argv) < 3:
    raise ValueError("usage: python compare_xh.py FILE1 FILE2")

xh0 = np.load(sys.argv[1])
#print(xh0.shape)
#print(xh0)
xh1 = np.load(sys.argv[2])
#print(xh1.shape)
#print(xh1)


#shared = count_shared_points(xh0[:,:3], xh1[:,:3], tolerance=3)
#print(shared)
key = 'Xh' if 'Xh' in xh0 else 'Xh_plus'
# sort the Xh arrays since the rows are unordered from non deterministic gpu
ref_sorted = sort_Xh(xh0[key])
print(ref_sorted.shape)
temp_sorted = sort_Xh(xh1[key])
print(temp_sorted.shape)
# now compare
np.testing.assert_allclose(temp_sorted, ref_sorted, rtol=1e-5, atol=1e-8)
if key == 'Xh_plus':
	key = 'Xh_minus'
	ref_sorted = sort_Xh(ref_data[key])
	temp_sorted = sort_Xh(temp_data[key])
	# now compare
	np.testing.assert_allclose(temp_sorted, ref_sorted, rtol=1e-5, atol=1e-8)


