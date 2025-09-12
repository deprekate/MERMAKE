import pytest
import cupy as cp
import numpy as np

from mermake.fill import reflect


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

    def test_reflect_1d_mode_out(self):
        arr = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
        result = reflect(arr, i=2, axis=0, mode="out")
        assert result is arr
        expected = cp.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=cp.float32)
        cp.testing.assert_array_equal(arr, expected)

    def test_reflect_1d_mode_in(self):
        arr = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
        reflect(arr, i=2, axis=0, mode="in")
        expected = cp.array([5.0, 4.0, 3.0, 4.0, 5.0], dtype=cp.float32)
        cp.testing.assert_array_equal(arr, expected)

    def test_reflect_2d_mode_in(self, test_data_2d):
        # edge, nothing reflected
        arr = test_data_2d.copy()
        reflect(arr, i=2, axis=0, mode="in")
        expected = cp.array([
            [ 1.0, 2.0, 3.0, 4.0, 5.0],
            [ 6.0, 7.0, 8.0, 9.0,10.0],
            [11.0,12.0,13.0,14.0,15.0]
        ], dtype=cp.float32)
        cp.testing.assert_array_equal(arr, expected)

        # reflect row 1 along axis 0
        arr = test_data_2d.copy()
        reflect(arr, i=1, axis=0, mode="in")
        expected = cp.array([
            [11.0,12.0,13.0,14.0,15.0],
            [ 6.0, 7.0, 8.0, 9.0,10.0],
            [11.0,12.0,13.0,14.0,15.0]
        ], dtype=cp.float32)
        cp.testing.assert_array_equal(arr, expected)

        # reflect col 2 along axis 1
        arr = test_data_2d.copy()
        reflect(arr, i=2, axis=1, mode="in")
        expected = cp.array([
            [ 5.0, 4.0, 3.0, 4.0, 5.0],
            [10.0, 9.0, 8.0, 9.0,10.0],
            [15.0,14.0,13.0,14.0,15.0]
        ], dtype=cp.float32)
        cp.testing.assert_array_equal(arr, expected)

        # weird edge case
        arr = test_data_2d.copy()
        reflect(arr, i=3, axis=1, mode="in")
        expected = cp.array([
            [ 1.0, 2.0, 5.0, 4.0, 5.0],
            [ 6.0, 7.0,10.0, 9.0,10.0],
            [11.0,12.0,15.0,14.0,15.0]
        ], dtype=cp.float32)
        cp.testing.assert_array_equal(arr, expected)

    def test_reflect_2d_mode_out(self, test_data_2d):
        arr = test_data_2d.copy()
        reflect(arr, i=0, axis=0, mode="out")
        expected = test_data_2d.copy()
        cp.testing.assert_array_equal(arr, expected)

        arr = test_data_2d.copy()
        reflect(arr, i=1, axis=0, mode="out")
        expected = cp.array([
            [ 1.0, 2.0, 3.0, 4.0, 5.0],
            [ 6.0, 7.0, 8.0, 9.0,10.0],
            [ 1.0, 2.0, 3.0, 4.0, 5.0]
        ], dtype=cp.float32)
        cp.testing.assert_array_equal(arr, expected)

        arr = test_data_2d.copy()
        reflect(arr, i=2, axis=1, mode="out")
        expected = cp.array([
            [ 1.0, 2.0, 3.0, 2.0, 1.0],
            [ 6.0, 7.0, 8.0, 7.0, 6.0],
            [11.0,12.0,13.0,12.0,11.0]
        ], dtype=cp.float32)
        cp.testing.assert_array_equal(arr, expected)

    def test_basic_functionality(self):
        arr1d = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        reflect(arr1d, i=1, axis=0, mode="out")
        reflect(arr1d, i=1, axis=0, mode="in")

        arr2d = cp.random.random((4, 5)).astype(cp.float32)
        reflect(arr2d, i=2, axis=0, mode="out")
        reflect(arr2d, i=2, axis=1, mode="in")

        arr3d = cp.random.random((3, 4, 5)).astype(cp.float32)
        reflect(arr3d, i=1, axis=0, mode="out")
        reflect(arr3d, i=2, axis=1, mode="in")
        reflect(arr3d, i=1, axis=2, mode="out")

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
        arr1 = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        arr2 = arr1.copy()
        reflect(arr1, i=1)
        reflect(arr2, i=1, axis=0, mode="out")
        cp.testing.assert_array_equal(arr1, arr2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

