# tests/test_blur.py
import pytest
import cupy as cp
import numpy as np

import mermake.blur as blur


def numpy_box_1d(arr, size, axis=0):
	"""Reference implementation of 1D box blur using NumPy."""
	arr = np.asarray(arr, dtype=np.float32)
	out = np.empty_like(arr)
	delta = size // 2
	it = np.nditer(arr, flags=["multi_index"])
	while not it.finished:
		idx = list(it.multi_index)
		start = max(0, idx[axis] - delta)
		end = min(arr.shape[axis], idx[axis] + delta + 1)
		slicer = [slice(i, i + 1) if ax != axis else slice(start, end)
				  for ax, i in enumerate(idx)]
		out[tuple(idx)] = arr[tuple(slicer)].mean()
		it.iternext()
	return out


@pytest.mark.parametrize("shape,axis,size", [
	((5, 5), 0, 3),
	((5, 5), 1, 3),
	((4, 4, 4), 0, 3),
	((4, 4, 4), 1, 5),
	((4, 4, 4), 2, 3),
])
def test_box_1d_matches_numpy(shape, axis, size):
	arr = cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape)
	got = blur.box_1d(arr, size, axis=axis)
	want = numpy_box_1d(cp.asnumpy(arr), size, axis=axis)
	cp.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.parametrize("axes,size", [
	((0, 1), 3),
	((1, 2), (3, 5)),
])
def test_box_and_box_2d_match_numpy(axes, size):
	arr = cp.arange(4*4*4, dtype=cp.float32).reshape((4, 4, 4))
	got = blur.box(arr, size, axes=axes)
	want = cp.asnumpy(arr)
	# Apply reference blur sequentially along axes
	if isinstance(size, int):
		sizes = [size] * len(axes)
	else:
		sizes = size
	for ax, s in zip(axes, sizes):
		want = numpy_box_1d(want, s, axis=ax)
	cp.testing.assert_allclose(got, want, rtol=1e-6)

	# box_2d convenience should match box with 2 axes
	if len(axes) == 2:
		got2d = blur.box_2d(arr, size, axes=axes)
		cp.testing.assert_allclose(got2d, got, rtol=1e-6)

def test_box_1d():
	arr = cp.zeros((3, 3, 3), dtype=cp.float32)
	arr[0,0,0] = 1
	want = cp.array([[[0.5       , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]],
					 [[0.33333334, 0.        , 0.        ],
					  [0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]],
					 [[0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]]], dtype=cp.float32)

	got = blur.box_1d(arr, 2, axis=0)
	cp.testing.assert_allclose(want, got, rtol=1e-6)

	want = cp.array([[[0.5       , 0.        , 0.        ],
					  [0.33333334, 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]],
					 [[0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]],
					 [[0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]]], dtype=cp.float32)
	got = blur.box_1d(arr, 2, axis=1)
	cp.testing.assert_allclose(want, got, rtol=1e-6)

	want = cp.array([[[0.5       , 0.33333334, 0.        ],
					  [0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]],
					 [[0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]],
					 [[0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ],
					  [0.        , 0.        , 0.        ]]], dtype=cp.float32)
	got = blur.box_1d(arr, 2, axis=2)
	cp.testing.assert_allclose(want, got, rtol=1e-6)

def test_inplace():
	arr = cp.zeros((3, 3, 3), dtype=cp.float32)
	arr[1,1] = 1
	with pytest.raises(ValueError):
		blur.box_1d(arr, 2, axis=0, out = arr)  # mismatch size/axes
	

def test_box_invalid_inputs():
	arr = cp.ones((3, 3, 3), dtype=cp.float32)
	with pytest.raises(ValueError):
		blur.box(arr, 3, axes=(0, 1, 2, 3))  # invalid axis
	with pytest.raises(TypeError):
		blur.box(arr, 3, axes="bad")  # wrong type
	with pytest.raises(ValueError):
		blur.box(arr, (3, 3), axes=(0, 1, 2))  # mismatch size/axes

