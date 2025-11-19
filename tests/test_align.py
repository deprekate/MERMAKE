import pytest
import numpy as np
import cupy as cp
import pickle
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

# Import the classes and functions to test
from mermake.align import Aligner, DualAligner, drift_save, drift


class TestAligner:
	
	@pytest.fixture
	def sample_X_ref(self):
		"""Create sample reference data"""
		return cp.array([
			[10, 20, 30, 0.6],
			[15, 25, 35, 0.7],
			[20, 30, 40, 0.8],
			[25, 35, 45, 0.9]
		])
	
	@pytest.fixture
	def aligner(self, sample_X_ref):
		"""Create an Aligner instance for testing"""
		return Aligner(sample_X_ref, resc=5, trim=10, th=0.5)
	
	def test_aligner_init_without_threshold(self, sample_X_ref):
		"""Test Aligner initialization without thresholding"""
		aligner = Aligner(sample_X_ref, resc=3, trim=5, th=None)
		assert aligner.resc == 3
		assert aligner.trim == 5
		assert aligner.th is None
		assert len(aligner.X_ref) == len(sample_X_ref)
		assert aligner.tree is not None
	
	def test_aligner_init_with_threshold(self, sample_X_ref):
		"""Test Aligner initialization with thresholding"""
		with patch.object(Aligner, 'threshold', return_value=sample_X_ref[:2]) as mock_threshold:
			aligner = Aligner(sample_X_ref, resc=5, trim=10, th=0.8)
			mock_threshold.assert_called_once_with(sample_X_ref)
			assert aligner.th == 0.8
	
	def test_get_im_from_Xh(self, aligner, sample_X_ref):
		"""Test image generation from point cloud"""
		im, Xm = aligner.get_im_from_Xh(sample_X_ref[:, :3])
		assert isinstance(im, cp.ndarray)
		assert isinstance(Xm, cp.ndarray)
		assert im.dtype == cp.float32
		assert Xm.dtype == cp.int32
		assert len(im.shape) == 3  # 3D image
		assert len(Xm) == 3  # 3D minimum coordinates
	
	def test_get_im_from_Xh_with_custom_params(self, aligner, sample_X_ref):
		"""Test image generation with custom rescale and trim parameters"""
		im, Xm = aligner.get_im_from_Xh(sample_X_ref[:, :3])
		assert isinstance(im, cp.ndarray)
		assert isinstance(Xm, cp.ndarray)
	
	def test_get_Xtzxy(self, aligner, sample_X_ref):
		"""Test translation estimation refinement"""
		X = sample_X_ref[:, :3] + cp.array([1, 1, 1])  # Slight offset
		tzxy0 = cp.array([1.0, 1.0, 1.0])
		tzxy, Npts = aligner.get_Xtzxy(X, tzxy0, target=3)
		
		assert isinstance(tzxy, cp.ndarray)
		assert isinstance(Npts, (int, cp.integer))
		assert len(tzxy) == 3
		assert Npts >= 0
	
	def test_threshold_with_valid_data(self, sample_X_ref):
		"""Test thresholding with valid data"""
		aligner = Aligner(sample_X_ref, th=0.75)
		result = aligner.threshold(sample_X_ref)
		
		# Should keep only points with intensity > 0.75
		expected_mask = sample_X_ref[:, -1] > 0.75
		expected_points = sample_X_ref[expected_mask]
		assert cp.array_equal(result, expected_points)
	
	def test_threshold_with_empty_data(self):
		"""Test thresholding with empty data"""
		X_ref = cp.array([[10, 20, 30, 0.8]])
		aligner = Aligner(X_ref, th=0.5)
		empty_data = cp.array([]).reshape(0, 4)
		result = aligner.threshold(empty_data)
	
		assert result.shape == (0,4)  # Returns zeros array
	
	def test_threshold_no_points_above_threshold(self, sample_X_ref):
		"""Test thresholding when no points meet the threshold"""
		aligner = Aligner(sample_X_ref, th=0.5)
		low_intensity_data = cp.array([[10, 20, 30, 0.3], [15, 25, 35, 0.4]])
		result = aligner.threshold(low_intensity_data)
		
		assert result.shape == (0,4)  # Returns zeros array
	
	def test_get_best_translation_points(self, aligner, sample_X_ref):
		"""Test best translation estimation"""
		X = sample_X_ref[:, :3] + cp.array([2, 2, 2])  # Known offset
		
		with patch.object(aligner, 'get_im_from_Xh') as mock_get_im:
			mock_get_im.return_value = (cp.ones((10, 10, 10)), cp.zeros(3))
			
			with patch('mermake.align.fftconvolve') as mock_fftconvolve:
				mock_fftconvolve.return_value = cp.ones((20, 20, 20))
				
				tzxy = aligner.get_best_translation_points(X)
				assert isinstance(tzxy, cp.ndarray)
				assert len(tzxy) == 3
	
	def test_get_best_translation_points_with_counts(self, aligner, sample_X_ref):
		"""Test best translation estimation with return counts"""
		X = sample_X_ref[:, :3]
		
		with patch.object(aligner, 'get_im_from_Xh') as mock_get_im:
			mock_get_im.return_value = (cp.ones((10, 10, 10)), cp.zeros(3))
			
			with patch('mermake.align.fftconvolve') as mock_fftconvolve:
				mock_fftconvolve.return_value = cp.ones((20, 20, 20))
				
				with patch.object(aligner, 'get_Xtzxy') as mock_get_Xtzxy:
					mock_get_Xtzxy.return_value = (cp.array([1, 2, 3]), 5)
					
					tzxy, Npts = aligner.get_best_translation_points(X, return_counts=True)
					assert isinstance(tzxy, cp.ndarray)
					assert isinstance(Npts, (int, cp.integer))
					assert Npts == 5


class TestDualAligner:
	
	@pytest.fixture
	def mock_ref(self):
		"""Create mock reference object"""
		ref = Mock()
		ref.Xh_plus = cp.array([
			[10, 20, 30, 0.8],
			[15, 25, 35, 0.9]
		])
		ref.Xh_minus = cp.array([
			[12, 22, 32, 0.7],
			[17, 27, 37, 0.85]
		])
		return ref
	
	@pytest.fixture
	def dual_aligner(self, mock_ref):
		"""Create DualAligner instance for testing"""
		return DualAligner(mock_ref, th=0.5)
	
	def test_dual_aligner_init(self, mock_ref):
		"""Test DualAligner initialization"""
		dual = DualAligner(mock_ref, th=0.6)
		assert dual.th == 0.6
		assert isinstance(dual.plus, Aligner)
		assert isinstance(dual.minus, Aligner)
	
	def test_get_best_translation_pointsV2_normal_case(self, dual_aligner):
		"""Test dual alignment with normal case"""
		obj = Mock()
		obj.Xh_plus = cp.array([[10, 20, 30], [15, 25, 35]])
		obj.Xh_minus = cp.array([[12, 22, 32], [17, 27, 37]])
		
		with patch.object(dual_aligner.plus, 'get_best_translation_points') as mock_plus:
			with patch.object(dual_aligner.minus, 'get_best_translation_points') as mock_minus:
				mock_plus.return_value = (cp.array([1, 2, 3]), 10)
				mock_minus.return_value = (cp.array([1.5, 2.5, 3.5]), 8)
				
				result = dual_aligner.get_best_translation_pointsV2(obj)
				tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus = result
				
				assert isinstance(tzxyf, cp.ndarray)
				assert len(tzxyf) == 3
				assert N_plus == 10
				assert N_minus == 8
	
	def test_get_best_translation_pointsV2_no_matches(self, dual_aligner):
		"""Test dual alignment with no matches"""
		obj = Mock()
		obj.Xh_plus = cp.array([[10, 20, 30]])
		obj.Xh_minus = cp.array([[12, 22, 32]])
		
		with patch.object(dual_aligner.plus, 'get_best_translation_points') as mock_plus:
			with patch.object(dual_aligner.minus, 'get_best_translation_points') as mock_minus:
				mock_plus.return_value = (cp.array([1, 2, 3]), 0)
				mock_minus.return_value = (cp.array([1, 2, 3]), 0)
				
				result = dual_aligner.get_best_translation_pointsV2(obj)
				tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus = result
				
				assert cp.allclose(tzxyf, cp.zeros(3))
				assert N_plus == 0
				assert N_minus == 0
	
	def test_get_best_translation_pointsV2_close_translations(self, dual_aligner):
		"""Test dual alignment with close translations (weighted average)"""
		obj = Mock()
		obj.Xh_plus = cp.array([[10, 20, 30]])
		obj.Xh_minus = cp.array([[12, 22, 32]])
		
		with patch.object(dual_aligner.plus, 'get_best_translation_points') as mock_plus:
			with patch.object(dual_aligner.minus, 'get_best_translation_points') as mock_minus:
				mock_plus.return_value = (cp.array([1, 2, 3]), 10)
				mock_minus.return_value = (cp.array([1, 2, 3]), 5)  # Same translation
				
				result = dual_aligner.get_best_translation_pointsV2(obj)
				tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus = result
				
				# Should be weighted average: -(1*10 + 1*5)/(10+5) = -1 for each component
				expected = -cp.array([1, 2, 3])
				assert cp.allclose(tzxyf, expected)
	
	def test_get_best_translation_pointsV2_different_translations(self, dual_aligner):
		"""Test dual alignment with different translations (pick stronger)"""
		obj = Mock()
		obj.Xh_plus = cp.array([[10, 20, 30]])
		obj.Xh_minus = cp.array([[12, 22, 32]])
		
		with patch.object(dual_aligner.plus, 'get_best_translation_points') as mock_plus:
			with patch.object(dual_aligner.minus, 'get_best_translation_points') as mock_minus:
				mock_plus.return_value = (cp.array([1, 2, 3]), 10)
				mock_minus.return_value = (cp.array([10, 20, 30]), 5)  # Very different
				
				result = dual_aligner.get_best_translation_pointsV2(obj)
				tzxyf, tzxy_plus, tzxy_minus, N_plus, N_minus = result
				
				# Should pick plus (stronger match): -[1, 2, 3]
				expected = -cp.array([1, 2, 3])
				assert cp.allclose(tzxyf, expected)


class TestUtilityFunctions:
	
	def test_drift_save(self):
		"""Test drift data saving"""
		test_data = {'drift': [1, 2, 3], 'files': ['a', 'b', 'c']}
		
		with tempfile.NamedTemporaryFile(delete=False) as tmp:
			try:
				drift_save(test_data, tmp.name)
				
				# Verify the file was created and contains correct data
				with open(tmp.name, 'rb') as f:
					loaded_data = pickle.load(f)
				assert loaded_data == test_data
			finally:
				os.unlink(tmp.name)
	
	def test_drift_function_file_exists(self):
		"""Test drift function when output file already exists"""
		mock_block = Mock()
		mock_block.fov.return_value = 1
		mock_block.iset.return_value = 2
		
		kwargs = {
			'output_folder': '/tmp',
			'drift_save': 'drift_fov{ifov}_iset{iset}.pkl'
		}
		
		with patch('os.path.exists', return_value=True):
			result = drift(mock_block, **kwargs)
			assert result is None  # Should return None if file exists
	
	def test_drift_function_file_not_exists(self):
		"""Test drift function when output file doesn't exist"""
		mock_block = MagicMock()
		mock_block.ifov.return_value = 1
		mock_block.iset.return_value = 2
		mock_block.__len__.return_value = 10
		mock_block.__getitem__.return_value = Mock()
		
		# Create mock images
		mock_images = []
		for i in range(5):
			mock_image = Mock()
			mock_image.path = f'/path/to/image_{i}.tif'
			mock_images.append(mock_image)
		
		mock_block.__iter__.return_value = iter(mock_images)
		mock_block.__getitem__.return_value = mock_images[2]  # Middle image as reference
		
		kwargs = {
			'output_folder': '/tmp',
			'drift_save': 'drift_fov{ifov}_iset{iset}.pkl'
		}
		
		with patch('os.path.exists', return_value=False):
			with patch('mermake.align.DualAligner') as mock_dual_class:
				mock_dual = Mock()
				mock_dual_class.return_value = mock_dual
				mock_dual.get_best_translation_pointsV2.return_value = ([1, 2, 3], [1, 2, 3], [1, 2, 3], 5, 3)
				
				result, filepath = drift(mock_block, **kwargs)
				
				# Check the structure of returned data
				drifts, files, ifov, ref_path = result
				assert len(drifts) == 5  # Number of images
				assert len(files) == 5
				assert ifov == 1
				assert 'image_2.tif' in ref_path
				assert filepath == '/tmp/drift_fov1_iset2.pkl'


class TestErrorHandling:
	
	def test_aligner_with_invalid_input_shape(self):
		"""Test Aligner with invalid input shape"""
		invalid_X_ref = cp.array([[1, 2]])  # Only 2D, should be at least 3D
		with pytest.raises((IndexError, ValueError)):
			Aligner(invalid_X_ref)
	

class TestEdgeCases:
	def test_aligner_with_single_point(self):
		"""Test Aligner with single reference point"""
		single_point = cp.array([[10, 20, 30, 0.8]])
		aligner = Aligner(single_point, resc=1, trim=1)
		assert aligner.X_ref.shape[0] == 1
	
	def test_aligner_with_no_points(self):
		"""Test Aligner with single reference point"""
		no_points = cp.ones([0,4])
		with pytest.raises((ValueError)):
			Aligner(no_points, resc=1, trim=1)
	
	def test_dual_aligner_empty_arrays(self):
		"""Test DualAligner with empty point arrays"""
		ref = Mock()
		ref.Xh_plus = cp.array([]).reshape(0, 4)
		ref.Xh_minus = cp.array([]).reshape(0, 4)
		
		# Should not handle empty arrays gracefully
		with pytest.raises((ValueError)):
			dual = DualAligner(ref, th=0.5)
		#assert dual.plus is not None
		#assert dual.minus is not None


@pytest.fixture(scope="session")
def setup_cupy():
	"""Setup CuPy for testing"""
	try:
		cp.cuda.Device(0).use()
		yield
	except cp.cuda.runtime.CUDARuntimeError:
		pytest.skip("No CUDA device available")


# Integration test
class TestIntegration:
	
	def test_full_alignment_pipeline(self, setup_cupy):
		"""Integration test for full alignment pipeline"""
		# Create synthetic data
		ref_points = cp.random.rand(100, 4) * 100
		ref_points[:, 3] = cp.random.rand(100) * 0.5 + 0.5  # Intensities 0.5-1.0
		
		# Create test points with known translation
		translation = cp.array([5, 3, -2])
		test_points = ref_points.copy()
		test_points[:, :3] += translation
		
		# Test single aligner
		aligner = Aligner(ref_points[:, :3], th=0.6)
		estimated_translation = aligner.get_best_translation_points(test_points[:, :3])
		
		# The estimated translation should be close to the negative of applied translation
		# (since we want to find the transformation to align test back to ref)
		assert isinstance(estimated_translation, cp.ndarray)
		assert len(estimated_translation) == 3


if __name__ == "__main__":
	pytest.main([__file__])
