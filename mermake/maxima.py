import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change "1" to the desired GPU ID
import time
import gc

import numpy as np
import cupy as cp

import cupy as cp
import cupyx
from cupyx.scipy.ndimage import convolve

import cupy as cp


class Maxima:
	def __init__(self, threshold=2500,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5, xp=cp):
		"""
		Detects local maxima in 3D image data and optionally refines their positions.
	
		Parameters:
		- im_dif_npy (ndarray): 3D difference image.
		- th_fit (float): Threshold for detecting peaks.
		- im_raw (ndarray, optional): Raw image for intensity reference.
		- dic_psf (dict, optional): Predefined PSF dictionary for fitting (not used here).
		- delta (int): Neighborhood size for peak suppression.
		- delta_fit (int): Neighborhood size for peak fitting.
		- sigmaZ, sigmaXY (float): Standard deviations for Gaussian fitting.
		- gpu (bool): Use CuPy for GPU acceleration.
	
		Returns:
		- ndarray: Array of detected peaks with refined positions.
		"""
		self.threshold = threshold
		# Vectorized offsets
		d1, d2, d3 = xp.meshgrid(
			xp.arange(-delta, delta + 1),
			xp.arange(-delta, delta + 1),
			xp.arange(-delta, delta + 1),
			indexing='ij'
		)
		mask = (d1 ** 2 + d2 ** 2 + d3 ** 2) <= (delta ** 2)
		self.d1 = d1[mask]
		self.d2 = d2[mask]
		self.d3 = d3[mask]
	
		if delta_fit > 0:
			d1, d2, d3 = xp.meshgrid(
					xp.arange(-delta_fit, delta_fit + 1),
					xp.arange(-delta_fit, delta_fit + 1),
					xp.arange(-delta_fit, delta_fit + 1),
					indexing='ij'
				)
			mask = (d1 ** 2 + d2 ** 2 + d3 ** 2) <= (delta_fit ** 2)
			self.dd1 = d1[mask]
			self.dd2 = d2[mask]
			self.dd3 = d3[mask]
			sigma = xp.array([sigmaZ, sigmaXY, sigmaXY], dtype=xp.float32)[xp.newaxis]
			Xft = xp.stack([self.dd1, self.dd2, self.dd3]).T / sigma
	
			norm_G = xp.exp(-xp.sum(Xft * Xft, -1) / 2.)
			norm_G = (norm_G - xp.mean(norm_G)) / xp.std(norm_G)
			self.norm_G = norm_G.reshape(-1,1)
		self.delta_fit = delta_fit

	def get_ind(self, a, amax):
		# modify a_ to be within image
		a_ = a.copy()
		bad = a_ >= amax
		a_[bad] = amax - a_[bad] - 1
		bad = a_ < 0
		a_[bad] = -a_[bad]
		return a_

	def apply(self, im_dif, im_raw=None):
		xp = cp.get_array_module(im_dif)
		im_dif = im_dif.astype(xp.float32)

		z,x,y = xp.where(im_dif > self.threshold)

		if len(z) == 0:
			return xp.empty((0, 8), dtype=xp.float32)  # Return empty CuPy array

		zmax, xmax, ymax = im_dif.shape

		# Compute indices for neighborhood check
		z_ = self.get_ind(z[:, None] + self.d1, zmax)
		x_ = self.get_ind(x[:, None] + self.d2, xmax)
		y_ = self.get_ind(y[:, None] + self.d3, ymax)
		
		# Get local maxima condition
		keep = im_dif[z, x, y][:, None] >= im_dif[z_, x_, y_]
		keep = keep.all(axis=1)  # Keep only points that are local maxima
		z, x, y = z[keep], x[keep], y[keep]
		
		if len(z) == 0:
			return xp.empty((0, 8), dtype=xp.float32)
	
		h = im_dif[z, x, y]

		if self.delta_fit > 0:
			im_centers0 = (z[:, None] + self.dd1).T
			im_centers1 = (x[:, None] + self.dd2).T
			im_centers2 = (y[:, None] + self.dd3).T
	
			z_ = self.get_ind(im_centers0, zmax)
			x_ = self.get_ind(im_centers1, xmax)
			y_ = self.get_ind(im_centers2, ymax)
	
			im_centers3 = im_dif[z_, x_, y_]
	
			if im_raw is not None:
				im_centers4 = im_raw[z_, x_, y_]
				habs = im_raw[z, x, y]
			else:
				im_centers4 = im_dif[z_, x_, y_]
				habs = xp.zeros_like(x)
	
			bk = xp.min(im_centers3,0)
	
			im_centers3 = im_centers3 - bk
			im_centers3 = im_centers3 / xp.sum(im_centers3,0)
	
			hn = xp.mean(((im_centers3-im_centers3.mean(0))/im_centers3.std(0))*self.norm_G, 0)
			a = xp.mean(((im_centers4-im_centers4.mean(0))/im_centers4.std(0))*self.norm_G, 0)

			# Compute weighted centroids
			zc = xp.sum(im_centers0 * im_centers3, axis=0)
			xc = xp.sum(im_centers1 * im_centers3, axis=0)
			yc = xp.sum(im_centers2 * im_centers3, axis=0)
			bk = xp.min(im_centers3, axis=0)
	
			Xh = xp.stack([zc, xc, yc, bk, a, habs, hn, h]).T
		else:
			Xh = xp.stack([z, x, y, h]).T
		return Xh

