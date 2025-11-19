import os
import pickle
from concurrent.futures import ThreadPoolExecutor

from .utils import fftconvolve

import cupy as cp
from cupyx.scipy.spatial import KDTree
#from scipy.signal import fftconvolve
#from scipy.spatial import KDTree

class Aligner:
	def __init__(self, X_ref, resc=5, trim=10, th=None):
		self.xp = cp.get_array_module(X_ref)
		self.resc = resc
		self.trim = trim
		self.th = th
		
		# do thresholding of the maximas
		if th:
			X_ref = self.threshold(X_ref)

		# unlike scipy the cupy KDTree crashes if given no points, so catch it here
		if X_ref.size == 0:
			raise ValueError(
				"No points left after thresholding; cannot build KDTree. "
				"Check your data and/or threshold value."
			)

		self.X_ref = X_ref[:,:3]
		# Build KDTree once
		self.tree = KDTree(self.X_ref)

		# Precompute reference image for FFT correlation
		self.im_ref, self.Xm_ref = self.get_im_from_Xh(self.X_ref)

	def get_im_from_Xh(self, Xh):
		xp = self.xp
		resc = self.resc
		trim = self.trim

		X = xp.round(Xh / resc).astype(xp.int32)
		Xm = xp.min(X, axis=0)
		XM = xp.max(X, axis=0)
		n = xp.asarray([0, trim, trim], dtype=xp.int32)
		keep = xp.all((X <= (XM - n)) & (X >= (Xm + n)), axis=-1)
		X = X[keep]
		# if trim is too extreme gracefully exit
		if X.shape[0] == 0:
			return xp.zeros((1, 1, 1), dtype=xp.float32), xp.array([0, 0, 0], dtype=xp.int32)

		Xm = xp.array([0, 0, 0], dtype=xp.int32)
		sz = xp.max(X, axis=0) + 1
		imf = xp.zeros(sz.tolist(), dtype=xp.float32)
		imf[tuple(X.T)] = 1
		return imf, Xm

	def get_Xtzxy(self, X, tzxy0, target=3):
		xp = self.xp
		tzxy = tzxy0
		Npts = 0

		for dist_th in xp.linspace(self.resc, target, 5):
			XT = X - tzxy
			ds, inds = self.tree.query(XT)
			keep = ds < dist_th
			X_ref_ = self.X_ref[inds[keep]]
			X_ = X[keep]
			if X_.shape[0] == 0:
				# no points kept, skip update
				continue
			tzxy = xp.mean(X_ - X_ref_, axis=0)
		Npts = int(xp.sum(keep))
		return tzxy, Npts

	def get_shifted_slices(self, X, shape):
		"""Compute slices for signal and background images given integer shift."""
		slices_sig = []
		slices_bk = []
		tzxy = self.get_best_translation_points(X)
		tzxy = tzxy.round().astype(int).tolist()
		for t, sh in zip(tzxy, shape):
			if t > 0:
				slices_sig.append(slice(t, sh))
				slices_bk.append(slice(0, sh - t))
			else:
				slices_sig.append(slice(0, sh + t))
				slices_bk.append(slice(-t, sh))
		return tuple(slices_bk) , tuple(slices_sig)

	def get_best_translation_points(self, X, target=3, return_counts=False):
		xp = self.xp
	
		# do thresholding of the maximas
		if self.th:
			X = self.threshold(X)

		X = X[:,:3]

		# Image for FFT correlation
		im, Xm = self.get_im_from_Xh(X)

		# Pure-CuPy FFT correlation
		im_cor = fftconvolve(im, self.im_ref[::-1, ::-1, ::-1])
		im_cor_shape = xp.asarray(im_cor.shape)
		im_ref_shape = xp.asarray(self.im_ref.shape)
		tzxy = xp.array(xp.unravel_index(xp.argmax(im_cor), im_cor_shape)) - im_ref_shape + 1 + Xm - self.Xm_ref
		tzxy = tzxy * self.resc

		# Refinement
		tzxy, Npts = self.get_Xtzxy(X, tzxy, target=target)

		if return_counts:
			return tzxy, Npts
		return tzxy


	def threshold(self, X):
		xp = cp.get_array_module(X)
		th = self.th
		dim = X.shape[1]
		if X.size:
			X = X[X[:,-1]>th]
			if X.size:
				return X
		return xp.zeros([0,dim])

class DualAligner:
	def __init__(self, ref, th=None):
		self.xp = cp.get_array_module(ref)
		self.plus = Aligner(ref.Xh_plus[:,:3], th = th)
		self.minus = Aligner(ref.Xh_minus[:,:3], th = th)
		self.th = th
	
	def get_best_translation_pointsV2(self, obj):
		xp = self.xp

		Xh_plus = obj.Xh_plus
		Xh_minus = obj.Xh_minus
		
		tzxy_plus,N_plus = self.plus.get_best_translation_points(Xh_plus, return_counts = True) 
		tzxy_minus,N_minus = self.minus.get_best_translation_points(Xh_minus, return_counts = True) 
	
		if N_plus + N_minus == 0:
			tzxyf = xp.zeros(3)
		elif xp.max(xp.abs(tzxy_minus - tzxy_plus)) <= 2:
			# weighted average
			tzxyf = -(tzxy_plus * N_plus + tzxy_minus * N_minus) / (N_plus + N_minus)
		else:
			# pick the stronger match
			tzxyf = -tzxy_plus if N_plus >= N_minus else -tzxy_minus

		return tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus

def drift_save(data, filepath):
	with open(filepath, 'wb') as f:
		pickle.dump(data, f)

def drift(block, **kwargs):
	output_folder = kwargs['output_folder']
	drift_save = kwargs['drift_save']
	ifov = block.ifov()
	iset = block.iset()
	filename = drift_save.format(ifov=ifov, iset=iset)
	filepath = os.path.join(output_folder, filename)
	if not os.path.exists(filepath):
		ref = block[len(block)//2]
		dual = DualAligner(ref, th=4)
		drifts = []
		files = []
		for image in block:
			drift = dual.get_best_translation_pointsV2(image)
			drifts.append(drift)
			files.append(os.path.dirname(image.path))
		return [drifts, files, ifov, ref.path], filepath
		#pickle.dump([drifts, files, fov, ref.path], open(filepath,'wb'))

