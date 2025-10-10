import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import gc

try:
	import cupy as cp
	CUPY_AVAILABLE = True
except ImportError:
	import numpy as cp
	CUPY_AVAILABLE = False

from mermake.io import read_im

from mermake.deconvolver import (
	laplacian_3d, 
	batch_laplacian_fft, 
	repeat_last, 
	Deconvolver, 
	full_deconv
)


class TestUtilityFunctions:
	"""Test utility functions."""
	
	def test_repeat_last_basic(self):
		"""Test repeat_last with basic iterable."""
		result = list(zip(repeat_last([1, 2, 3]), range(6)))
		expected = [(1, 0), (2, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
		assert result == expected
	
	def test_repeat_last_empty(self):
		"""Test repeat_last with empty iterable."""
		result = list(repeat_last([]))
		assert result == []
	
	def test_repeat_last_single_element(self):
		"""Test repeat_last with single element."""
		result = list(zip(repeat_last([42]), range(5)))
		expected = [(42, 0), (42, 1), (42, 2), (42, 3), (42, 4)]
		assert result == expected


class TestLaplacianFunctions:
	"""Test Laplacian-related functions."""
	
	def test_batch_laplacian_fft(self):
		"""Test batch_laplacian_fft function."""
		batch_size = 2
		shape = (5, 5, 5)
		
		result = batch_laplacian_fft(batch_size, shape)
		
		# Check output shape (should have batch dimension)
		assert result.shape == (1, *shape)
		
		# Check it's in frequency domain (complex values)
		if CUPY_AVAILABLE:
			assert result.dtype == cp.complex64 or result.dtype == cp.complex128
		else:
			assert result.dtype == np.complex64 or result.dtype == np.complex128


class TestDeconvolverInitialization:
	"""Test Deconvolver class initialization."""
	
	@pytest.fixture
	def sample_psf(self):
		"""Create a sample PSF for testing."""
		# Create a small Gaussian-like PSF
		psf = np.zeros((10, 10, 10))
		psf[5, 5, 5] = 1.0
		# Add some spread
		psf[4:7, 4:7, 4:7] += 0.1
		return psf
	
	@pytest.fixture
	def sample_image_shape(self):
		"""Sample image shape for testing."""
		return (20, 100, 100)  # batch, z, x, y
	
	def test_deconvolver_init_single_psf(self, sample_psf, sample_image_shape):
		"""Test Deconvolver initialization with single PSF."""
		deconv = Deconvolver(
			psfs=sample_psf,
			channel_shape=sample_image_shape,
			tile_size=50,
			zpad=5,
			overlap=10,
			beta=0.001
		)
		
		assert deconv.tile_size == 50
		assert deconv.overlap == 10
		assert deconv.zpad == 5
		assert deconv.psf_fft is not None
	
	def test_deconvolver_init_multiple_psfs(self, sample_psf, sample_image_shape):
		"""Test Deconvolver initialization with multiple PSFs."""
		psfs_dict = {
			'center': sample_psf,
			'corner1': sample_psf * 0.9,
			'corner2': sample_psf * 1.1
		}
		
		deconv = Deconvolver(
			psfs=psfs_dict,
			channel_shape=sample_image_shape,
			tile_size=50,
			overlap=10,
			beta=0.001
		)
		
		assert len(deconv.psf_fft) == 3
	
	def test_deconvolver_memory_allocation(self, sample_psf, sample_image_shape):
		"""Test that Deconvolver properly allocates memory for tiles."""
		tile_size = 50
		overlap = 10
		zpad = 5
		
		deconv = Deconvolver(
			psfs=sample_psf,
			channel_shape=sample_image_shape,
			tile_size=tile_size,
			zpad=zpad,
			overlap=overlap
		)
		
		expected_tile_shape = (tile_size + 2*overlap, tile_size + 2*overlap)
		expected_pad_shape = (2*zpad + sample_image_shape[0], *expected_tile_shape)
		expected_res_shape = (sample_image_shape[0], *expected_tile_shape)
		
		assert deconv.tile_pad.shape == expected_pad_shape
		assert deconv.tile_res.shape == expected_res_shape


class TestDeconvolverMethods:
	"""Test Deconvolver methods."""
	
	@pytest.fixture
	def deconv_instance(self):
		"""Create a Deconvolver instance for testing."""
		psf = np.zeros((10, 10, 10))
		psf[5, 5, 5] = 1.0
		channel_shape = (20, 100, 100)
		
		return Deconvolver(
			psfs=psf,
			channel_shape=channel_shape,
			tile_size=50,
			zpad=5,
			overlap=10,
			beta=0.001
		)
	
	def test_center_psf_same_size(self, deconv_instance):
		"""Test center_psf with PSF same size as target."""
		psf = np.ones((20, 50, 50))  # Same as target
		result = deconv_instance.center_psf(psf)
		
		assert result.shape == (20, 50, 50)
		assert np.allclose(result.sum(), 1.0)  # Should be normalized
	
	def test_center_psf_smaller(self, deconv_instance):
		"""Test center_psf with smaller PSF."""
		psf = np.ones((10, 25, 25))  # Smaller than target
		result = deconv_instance.center_psf(psf)
		
		assert result.shape == (20, 50, 50)
		assert np.allclose(result.sum(), 1.0)  # Should be normalized
		# Check that PSF is centered
		assert result[10, 25, 25] > 0  # Should have non-zero values in center
	
	def test_center_psf_larger(self, deconv_instance):
		"""Test center_psf with larger PSF."""
		psf = np.ones((30, 75, 75))  # Larger than target
		result = deconv_instance.center_psf(psf)
		
		assert result.shape == (20, 50, 50)
		assert np.allclose(result.sum(), 1.0)  # Should be normalized
	
	def test_tiled_basic(self, deconv_instance):
		"""Test basic tiling functionality."""
		image = np.random.random((20, 100, 100))
		
		tiles = list(deconv_instance.tiled(image))
		
		# Should have 4 tiles (2x2 grid for 100x100 image with tile_size=50)
		assert len(tiles) == 4
		
		# Check tile positions
		positions = [(x, y) for x, y, _ in tiles]
		expected_positions = [(0, 0), (0, 50), (50, 0), (50, 50)]
		assert positions == expected_positions
	
	def test_tiled_with_overlap(self, deconv_instance):
		"""Test that tiles include proper overlap."""
		image = np.random.random((20, 100, 100))
		
		overlap = deconv_instance.overlap
		tile_size = deconv_instance.tile_size
		for x, y, tile in deconv_instance.tiled(image):
			zdim,xdim,ydim = tile.shape
			xstart = 0 if x == 0 else overlap
			ystart = 0 if y == 0 else overlap
			xend = xdim if xdim < tile_size else tile_size
			yend = ydim if ydim < tile_size else tile_size
			# First tile should include overlap on right and bottom
			if x == 0 and y == 0:
				assert tile.shape == (20, 60, 60)  # 50 + 10 overlap
			# Check that tile data matches original image
			#assert np.array_equal(tile[:,overlap:-overlap,overlap:-overlap], image[:, x:x+tile_size, y:y+tile_size]), f"x={x},y={y}"
			sub = tile[:,xstart:xstart+xend, ystart:ystart+yend]
			ref = image[:, x:x+tile_size, y:y+tile_size]
			assert sub.shape == ref.shape
			assert np.array_equal(sub, ref), f"x={x},y={y}"


class TestDeconvolverProcessing:
	"""Test actual deconvolution processing."""
	
	@pytest.fixture
	def simple_test_data(self):
		"""Create simple test data."""
		# Create a simple test image with a point source
		image = cp.zeros((10, 50, 50))
		image[5, 25, 25] = 100  # Point source in center
		
		# Create a simple PSF
		psf = np.zeros((10, 10, 10))
		psf[5, 5, 5] = 1.0
		
		return image, psf
	
	def test_apply_basic(self, simple_test_data):
		"""Test basic apply functionality."""
		image, psf = simple_test_data
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			tile_size=30,
			overlap=5,
			zpad=2,
			beta=0.001
		)
		result = deconv.apply(image)
		
		assert result.shape == image.shape, 1
		assert result.dtype == np.float32, 2
		# Result should be finite (no NaN or inf)
		assert np.all(np.isfinite(result)), 3
	
	@pytest.mark.parametrize("blur_radius", [None, 2, 5])
	def test_apply_with_blur_subtraction(self, simple_test_data, blur_radius):
		"""Test apply with different blur subtraction settings."""
		image, psf = simple_test_data
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			tile_size=30,
			overlap=5,
			zpad=2
		)
		
		result = deconv.apply(image, blur_radius=blur_radius)
		
		assert result.shape == image.shape
		assert np.all(np.isfinite(result))
	
	def test_apply_with_flat_field(self, simple_test_data):
		"""Test apply with flat field correction."""
		image, psf = simple_test_data
		# Create a simple flat field (slight gradient)
		flat_field = cp.ones(( 50, 50))
		flat_field[:, :] *= cp.linspace(0.8, 1.2, 50)[:, None]
		
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			tile_size=30,
			overlap=5,
			zpad=2
		)
		
		result = deconv.apply(image, flat_field=flat_field)
		
		assert result.shape == image.shape
		assert np.all(np.isfinite(result))


class TestDeconvolverEdgeCases:
	"""Test edge cases and error conditions."""
	
	def test_small_image(self):
		"""Test with very small image."""
		image = cp.random.random((5, 20, 20))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			tile_size=15,  # Smaller than image
			overlap=2,
			zpad=1
		)
		
		result = deconv.apply(image)
		assert result.shape == image.shape
	
	def test_single_tile_image(self):
		"""Test with image that fits in a single tile."""
		image = cp.random.random((10, 30, 30))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			tile_size=50,  # Larger than image
			overlap=5,
			zpad=2
		)
		
		result = deconv.apply(image)
		assert result.shape == image.shape


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestCuPySpecific:
	"""Test CuPy-specific functionality."""
	
	def test_cupy_memory_management(self):
		"""Test that CuPy memory is properly managed."""
		image = cp.random.random((10, 50, 50))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		# Monitor memory before
		mempool = cp.get_default_memory_pool()
		initial_memory = mempool.used_bytes()
		
		deconv = Deconvolver(
			psfs=psf,
			channel_shape=image.shape,
			tile_size=30,
			overlap=5,
			zpad=2,
			xp=cp
		)
		
		result = deconv.apply(image)
		
		# Clean up
		del deconv, result
		cp._default_memory_pool.free_all_blocks()
		
		# Memory should be cleaned up
		final_memory = mempool.used_bytes()
		assert final_memory <= initial_memory + 1024  # Allow small overhead


class TestFullDeconv:
	"""Test the full_deconv convenience function."""
	
	def test_full_deconv_basic(self):
		"""Test basic full_deconv functionality."""
		image = cp.random.random((10, 50, 50))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		result = full_deconv(
			image=image,
			psfs=psf,
			tile_size=30,
			overlap=5,
			beta=0.001
		)
		
		assert result.shape == image.shape
		assert np.all(np.isfinite(result))
	
	def test_full_deconv_with_all_options(self):
		"""Test full_deconv with all options."""
		image = cp.random.random((10, 50, 50))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		flat_field = cp.ones(( 50, 50))
		
		result = full_deconv(
			image=image,
			psfs=psf,
			flat_field=flat_field,
			tile_size=30,
			zpad=5,
			overlap=5,
			beta=0.01
		)
		
		assert result.shape == image.shape
		assert np.all(np.isfinite(result))
	
	def test_full_deconv_memory_cleanup(self):
		"""Test that full_deconv cleans up memory properly."""
		image = cp.random.random((10, 50, 50))
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		# This should not raise memory errors and should clean up
		result = full_deconv(image=image, psfs=psf, tile_size=30)
		
		assert result is not None
		assert result.shape == image.shape

	def test_full_deconv_full(self):
		image = read_im('tests/examples/single/H1_TEST_set1/Conv_zscan__020.zarr')
		cim = cp.asarray(image[0])

		psfs = np.load('tests/examples/psf_750_Scope0_final.npy')
		result = full_deconv(image=cim, psfs=psfs, tile_size=300, overlap=89)

		expected = cp.zeros([35,1200,1200], dtype=cp.float16)
		for z in range(35):
			zslice = cp.load(f'tests/examples/single/out/deconv.300.{z}.npz')
			expected[z] = zslice['arr_0']
	
		cp.testing.assert_allclose(result.astype(cp.float16), expected, rtol=1e-6)


class TestIntegrationAndPerformance:
	"""Integration tests and performance checks."""
	
	def test_deconvolution_preserves_intensity(self):
		"""Test that deconvolution roughly preserves total intensity."""
		# Create image with known total intensity
		image = cp.zeros((10, 50, 50))
		image[5, 25, 25] = 100
		total_input = np.sum(image)
		
		psf = np.zeros((5, 5, 5))
		psf[2, 2, 2] = 1.0
		
		result = full_deconv(
			image=image,
			psfs=psf,
			tile_size=30,
			beta=0.001  # Low regularization
		)
		
		total_output = np.sum(result)
		
		# Should preserve intensity within reasonable bounds
		# (exact preservation depends on regularization and boundary effects)
		assert abs(total_output - total_input) / total_input < 0.5
	
	def test_different_psf_formats(self):
		"""Test that different PSF input formats work."""
		image = cp.random.random((10, 50, 50))
		
		# Single PSF
		psf_single = np.zeros((5, 5, 5))
		psf_single[2, 2, 2] = 1.0
		
		result1 = full_deconv(image=image, psfs=psf_single, tile_size=30)
		
		# Dictionary of PSFs
		psfs_dict = {
			'center': psf_single,
			'corner': psf_single * 0.9
		}
		
		result2 = full_deconv(image=image, psfs=psfs_dict, tile_size=30)
		
		assert result1.shape == result2.shape == image.shape
		# Results should be different due to different PSF handling
		assert not np.allclose(result1, result2)

'''
# Mock tests for dependencies that might not be available
class TestMockDependencies:
	"""Test with mocked dependencies."""
	
	@patch('mermake.deconvolver.cp', new=np)
	def test_fallback_to_numpy(self):
		"""Test that code works when CuPy is mocked to NumPy."""
		image = np.random.random((5, 20, 20))
		psf = np.zeros((3, 3, 3))
		psf[1, 1, 1] = 1.0
		
		# This should work even with mocked CuPy
		result = full_deconv(image=image, psfs=psf, tile_size=15)
		assert result.shape == image.shape
	
	@patch('mermake.blur.box_1d')
	def test_with_mocked_blur(self, mock_blur):
		"""Test with mocked blur function."""
		mock_blur.return_value = None  # blur modifies in-place
		
		image = cp.random.random((5, 20, 20))
		psf = np.zeros((3, 3, 3))
		psf[1, 1, 1] = 1.0
		
		result = full_deconv(
			image=image, 
			psfs=psf, 
			tile_size=15,
			blur_radius=2  # This should trigger the blur calls
		)
		
		# Blur should have been called
		assert mock_blur.called
		assert result.shape == image.shape
'''

if __name__ == "__main__":
	pytest.main([__file__, "-v"])
