import pytest
import cupy as cp
import numpy as np

from mermake.fill import reflect, repeat


@pytest.fixture
def test_data_2d():
	"""Provide a reusable 2D test array."""
	return cp.array([
		[ 1.0, 2.0, 3.0, 4.0, 5.0],
		[ 6.0, 7.0, 8.0, 9.0,10.0],
		[11.0,12.0,13.0,14.0,15.0]
	], dtype=cp.float32)


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestReflectFunction:
	"""Tests for the reflect function."""

	def test_reflect_1d_mode_out_non_inplace(self):
		"""Test non-inplace behavior (default)."""
		arr = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
		original = arr.copy()
		result = reflect(arr, i=2, axis=0, mode="out")
		
		# Original should be unchanged
		cp.testing.assert_array_equal(arr, original)
		# Result should be different from input
		assert result is not arr
		# Result should have expected values
		expected = cp.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

	def test_reflect_1d_mode_out_inplace(self):
		"""Test in-place behavior using out parameter."""
		arr = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
		result = reflect(arr, i=2, axis=0, mode="out", out=arr)
		
		# Should return same array
		assert result is arr
		# Should have expected values
		expected = cp.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=cp.float32)
		cp.testing.assert_array_equal(arr, expected)

	def test_reflect_1d_mode_in_non_inplace(self):
		"""Test non-inplace mode='in'."""
		arr = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
		original = arr.copy()
		result = reflect(arr, i=2, axis=0, mode="in")
		
		# Original should be unchanged
		cp.testing.assert_array_equal(arr, original)
		# Result should have expected values
		expected = cp.array([5.0, 4.0, 3.0, 4.0, 5.0], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

	def test_reflect_1d_mode_in_inplace(self):
		"""Test in-place mode='in'."""
		arr = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
		reflect(arr, i=2, axis=0, mode="in", out=arr)
		expected = cp.array([5.0, 4.0, 3.0, 4.0, 5.0], dtype=cp.float32)
		cp.testing.assert_array_equal(arr, expected)

	def test_reflect_2d_mode_in(self, test_data_2d):
		# edge, nothing reflected
		arr = test_data_2d.copy()
		result = reflect(arr, i=2, axis=0, mode="in")
		expected = cp.array([
			[ 1.0, 2.0, 3.0, 4.0, 5.0],
			[ 6.0, 7.0, 8.0, 9.0,10.0],
			[11.0,12.0,13.0,14.0,15.0]
		], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)
		# Original unchanged
		cp.testing.assert_array_equal(arr, test_data_2d)

		# reflect row 1 along axis 0
		arr = test_data_2d.copy()
		result = reflect(arr, i=1, axis=0, mode="in")
		expected = cp.array([
			[11.0,12.0,13.0,14.0,15.0],
			[ 6.0, 7.0, 8.0, 9.0,10.0],
			[11.0,12.0,13.0,14.0,15.0]
		], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)
		# Original unchanged
		cp.testing.assert_array_equal(arr, test_data_2d)

		# reflect col 2 along axis 1
		arr = test_data_2d.copy()
		result = reflect(arr, i=2, axis=1, mode="in")
		expected = cp.array([
			[ 5.0, 4.0, 3.0, 4.0, 5.0],
			[10.0, 9.0, 8.0, 9.0,10.0],
			[15.0,14.0,13.0,14.0,15.0]
		], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

		# weird edge case
		arr = test_data_2d.copy()
		result = reflect(arr, i=3, axis=1, mode="in")
		expected = cp.array([
			[ 1.0, 2.0, 5.0, 4.0, 5.0],
			[ 6.0, 7.0,10.0, 9.0,10.0],
			[11.0,12.0,15.0,14.0,15.0]
		], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

	def test_reflect_2d_mode_out(self, test_data_2d):
		arr = test_data_2d.copy()
		result = reflect(arr, i=0, axis=0, mode="out")
		expected = test_data_2d.copy()
		cp.testing.assert_array_equal(result, expected)

		arr = test_data_2d.copy()
		result = reflect(arr, i=1, axis=0, mode="out")
		expected = cp.array([
			[ 1.0, 2.0, 3.0, 4.0, 5.0],
			[ 6.0, 7.0, 8.0, 9.0,10.0],
			[ 1.0, 2.0, 3.0, 4.0, 5.0]
		], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

		arr = test_data_2d.copy()
		result = reflect(arr, i=2, axis=1, mode="out")
		expected = cp.array([
			[ 1.0, 2.0, 3.0, 2.0, 1.0],
			[ 6.0, 7.0, 8.0, 7.0, 6.0],
			[11.0,12.0,13.0,12.0,11.0]
		], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

	def test_reflect_out_parameter_validation(self):
		"""Test validation of out parameter."""
		arr = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
		
		# Wrong shape
		wrong_shape = cp.zeros((4,), dtype=cp.float32)
		with pytest.raises(ValueError, match="output array shape"):
			reflect(arr, i=1, out=wrong_shape)
		
		# Wrong dtype
		wrong_dtype = cp.zeros((3,), dtype=cp.int32)
		with pytest.raises(ValueError, match="output array dtype"):
			reflect(arr, i=1, out=wrong_dtype)
		
		# Correct usage with pre-allocated array
		out = cp.zeros_like(arr)
		result = reflect(arr, i=1, out=out)
		assert result is out

	def test_basic_functionality(self):
		arr1d = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
		reflect(arr1d, i=1, axis=0, mode="out", out=arr1d)
		reflect(arr1d, i=1, axis=0, mode="in", out=arr1d)

		arr2d = cp.random.random((4, 5)).astype(cp.float32)
		reflect(arr2d, i=2, axis=0, mode="out", out=arr2d)
		reflect(arr2d, i=2, axis=1, mode="in", out=arr2d)

		arr3d = cp.random.random((3, 4, 5)).astype(cp.float32)
		reflect(arr3d, i=1, axis=0, mode="out", out=arr3d)
		reflect(arr3d, i=2, axis=1, mode="in", out=arr3d)
		reflect(arr3d, i=1, axis=2, mode="out", out=arr3d)

	def test_error_handling(self):
		arr = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
		with pytest.raises(IndexError):
			reflect(arr, i=5, axis=0)
		with pytest.raises(ValueError):
			reflect(arr, i=1, mode="invalid")
		with pytest.raises(IndexError):
			reflect(arr, i=1, axis=1)
		with pytest.raises(IndexError):
			reflect(arr, i=3, axis=0)

	def test_default_parameters(self):
		"""Test that defaults still work as expected (non-inplace now)."""
		arr1 = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
		arr2 = arr1.copy()
		
		result1 = reflect(arr1, i=1)  # Non-inplace by default now
		result2 = reflect(arr2, i=1, axis=0, mode="out")  # Explicit parameters
		
		# Both originals should be unchanged
		cp.testing.assert_array_equal(arr1, arr2)
		# Results should be the same
		cp.testing.assert_array_equal(result1, result2)


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestRepeatFunction:
	"""Tests for the repeat (fill) function."""
	
	def test_repeat_1d_mode_in_basic(self):
		"""Test basic 1D repeat functionality from your example."""
		arr = cp.array([1,2,3,4,5,6,7], dtype=cp.float32)
		result = repeat(arr, 3, axis=0, mode='in')
		expected = cp.array([4,4,4,4,5,6,7], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)
		
		# Original should be unchanged
		original = cp.array([1,2,3,4,5,6,7], dtype=cp.float32)
		cp.testing.assert_array_equal(arr, original)

	def test_repeat_1d_mode_out(self):
		"""Test 1D repeat with mode='out'."""
		arr = cp.array([1,2,3,4,5,6,7], dtype=cp.float32)
		result = repeat(arr, 3, axis=0, mode='out')
		expected = cp.array([1,2,3,4,4,4,4], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

	def test_repeat_1d_inplace(self):
		"""Test in-place repeat operation."""
		arr = cp.array([1,2,3,4,5,6,7], dtype=cp.float32)
		result = repeat(arr, 3, axis=0, mode='in', out=arr)
		expected = cp.array([4,4,4,4,5,6,7], dtype=cp.float32)
		
		assert result is arr
		cp.testing.assert_array_equal(arr, expected)

	def test_repeat_2d_axis0(self, test_data_2d):
		"""Test 2D repeat along axis 0."""
		arr = test_data_2d.copy()
		result = repeat(arr, 1, axis=0, mode='in')
		
		# Row 0 should be filled with row 1 values
		expected = cp.array([
			[ 6.0, 7.0, 8.0, 9.0,10.0],  # Row 1 values
			[ 6.0, 7.0, 8.0, 9.0,10.0],  # Row 1 values (unchanged)
			[11.0,12.0,13.0,14.0,15.0]   # Row 2 values (unchanged)
		], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

	def test_repeat_2d_axis1(self, test_data_2d):
		"""Test 2D repeat along axis 1."""
		arr = test_data_2d.copy()
		result = repeat(arr, 2, axis=1, mode='out')
		
		# Columns 3,4 should be filled with column 2 values
		expected = cp.array([
			[ 1.0, 2.0, 3.0, 3.0, 3.0],
			[ 6.0, 7.0, 8.0, 8.0, 8.0],
			[11.0,12.0,13.0,13.0,13.0]
		], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)
	
	def test_repeat_twice_serially(self, test_data_2d):
		"""Test 2D repeat along axis 1."""
		arr = test_data_2d.copy()
		repeat(arr, 1, axis=0, mode='out', out=arr)
		repeat(arr, 1, axis=1, mode='out', out=arr)
		
		# Columns 3,4 should be filled with column 2 values
		expected = cp.array([
			[ 1.0, 2.0, 2.0, 2.0, 2.0],
			[ 6.0, 7.0, 7.0, 7.0, 7.0],
			[ 6.0, 7.0, 7.0, 7.0, 7.0]
		], dtype=cp.float32)
		cp.testing.assert_array_equal(arr, expected)

	def test_repeat_edge_cases(self):
		"""Test edge cases for repeat function."""
		# Fill at beginning
		arr = cp.array([1,2,3,4,5], dtype=cp.float32)
		result = repeat(arr, 0, axis=0, mode='out')
		expected = cp.array([1,1,1,1,1], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)
		
		# Fill at end
		arr = cp.array([1,2,3,4,5], dtype=cp.float32)
		result = repeat(arr, 4, axis=0, mode='in')
		expected = cp.array([5,5,5,5,5], dtype=cp.float32)
		cp.testing.assert_array_equal(result, expected)

	def test_repeat_out_parameter_validation(self):
		"""Test validation of out parameter for repeat."""
		arr = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
		
		# Wrong shape
		wrong_shape = cp.zeros((4,), dtype=cp.float32)
		with pytest.raises(ValueError, match="output array shape"):
			repeat(arr, i=1, out=wrong_shape)
		
		# Wrong dtype
		wrong_dtype = cp.zeros((3,), dtype=cp.int32)
		with pytest.raises(ValueError, match="output array dtype"):
			repeat(arr, i=1, out=wrong_dtype)
		
		# Correct usage with pre-allocated array
		out = cp.zeros_like(arr)
		result = repeat(arr, i=1, out=out)
		assert result is out

	def test_repeat_error_handling(self):
		"""Test error handling for repeat function."""
		arr = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
		
		# Invalid index
		with pytest.raises(IndexError):
			repeat(arr, i=5, axis=0)
		
		# Invalid mode
		with pytest.raises(ValueError):
			repeat(arr, i=1, mode="invalid")
		
		# Invalid axis
		with pytest.raises(IndexError):
			repeat(arr, i=1, axis=1)

	def test_repeat_3d(self):
		"""Test repeat function on 3D arrays."""
		arr = cp.random.random((3, 4, 5)).astype(cp.float32)
		original = arr.copy()
		
		# Test different axes
		result0 = repeat(arr, i=1, axis=0, mode="out")
		result1 = repeat(arr, i=2, axis=1, mode="in")
		result2 = repeat(arr, i=1, axis=2, mode="out")
		
		# Original should be unchanged
		cp.testing.assert_array_equal(arr, original)
		
		# Results should have correct shapes
		assert result0.shape == arr.shape
		assert result1.shape == arr.shape
		assert result2.shape == arr.shape

	def test_repeat_default_parameters(self):
		"""Test default parameters for repeat function."""
		arr = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
		
		# Test with minimal parameters (should default to axis=0, mode="out")
		result1 = repeat(arr, i=1)
		result2 = repeat(arr, i=1, axis=0, mode="out")
		
		cp.testing.assert_array_equal(result1, result2)


if __name__ == "__main__":
	pytest.main([__file__, "-v"])
