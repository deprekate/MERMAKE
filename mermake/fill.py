import cupy as cp


import cupy as cp

reflect_kernel = cp.RawKernel(r'''
extern "C" __global__
void mirror(float* arr, const int ndim, const long long* shape,
			const long long* strides, const int axis, const int i, const int mode) {
	long idx = blockDim.x * blockIdx.x + threadIdx.x;
	long size = 1;
	for (int d=0; d<ndim; d++) size *= shape[d];
	if (idx >= size) return;
	// Convert flat idx -> multi-index
	long coords[16];  // supports up to 16D
	long tmp = idx;
	for (int d=ndim-1; d>=0; d--) {
		coords[d] = tmp % shape[d];
		tmp /= shape[d];
	}
	int j = coords[axis];
	int n = shape[axis];
	int dist = j - i;
	if (mode == 0 && dist > 0) {  // "out": reflect right/down
		int mirror_j = i - dist;
		if (mirror_j >= 0) {
			coords[axis] = mirror_j;
		} else {
			return;
		}
	} else if (mode == 1 && dist < 0) {  // "in": reflect left/up
		int mirror_j = i - dist;
		if (mirror_j < n) {
			coords[axis] = mirror_j;
		} else {
			return;
		}
	} else {
		return;
	}
	long offset_src = 0;
	for (int d=0; d<ndim; d++) offset_src += coords[d]*strides[d];
	long offset_dst = 0;
	tmp = idx;
	for (int d=ndim-1; d>=0; d--) {
		int c = tmp % shape[d];
		tmp /= shape[d];
		offset_dst += c*strides[d];
	}
	arr[offset_dst/sizeof(float)] = arr[offset_src/sizeof(float)];
}
''', 'mirror')

def reflect(arr: cp.ndarray, i: int, axis: int=0, mode: str="out", out: cp.ndarray=None):
	"""
	Reflect array elements around index i along the specified axis.
	
	Parameters:
	- arr: input array
	- i: index along axis to reflect around
	- axis: axis along which to perform the reflection
	- mode: 'out' reflects elements after index i, 'in' reflects elements before index i
	- out: output array. If None, creates a copy. If provided, must have same shape as arr.
	
	Returns:
	- reflected array (either new array or the out parameter)
	"""
	# Handle output array
	if out is None:
		result = arr.copy()
	else:
		if out.shape != arr.shape:
			raise ValueError(f"output array shape {out.shape} does not match input shape {arr.shape}")
		if out.dtype != arr.dtype:
			raise ValueError(f"output array dtype {out.dtype} does not match input dtype {arr.dtype}")
		result = out
		# Copy input to output if they're different arrays
		if result is not arr:
			cp.copyto(result, arr)
	
	# Handle negative axis
	if axis < 0:
		axis += result.ndim
	if axis < 0 or axis >= result.ndim:
		raise IndexError(f"axis {axis} is out of bounds for array of dimension {result.ndim}")
	
	n = result.shape[axis]
	
	# Handle negative index
	if i < 0: 
		i += n
	if i < 0 or i >= n:
		raise IndexError(f"index {i} is out of bounds for axis {axis} with size {n}")
	
	size = result.size
	threads = 256
	blocks = (size + threads - 1) // threads
	
	shape = cp.array(result.shape, dtype=cp.int64)
	strides = cp.array(result.strides, dtype=cp.int64)
	
	if mode == "out":
		mode_flag = 0
	elif mode == "in":
		mode_flag = 1
	else:
		raise ValueError("mode must be 'out' or 'in'")
	
	reflect_kernel((blocks,), (threads,),
				   (result, result.ndim, shape.data.ptr, strides.data.ptr, axis, i, mode_flag))
	return result

repeat_kernel = cp.RawKernel(r'''
extern "C" __global__
void fill_simple(float* arr, const int ndim, const long* shape,
				 const long* strides, const int axis, const int i, const int mode) {
	long idx = blockDim.x * blockIdx.x + threadIdx.x;
	long size = 1;
	for (int d=0; d<ndim; d++) size *= shape[d];
	if (idx >= size) return;

	// Convert flat idx -> multi-index
	long coords[16];  // supports up to 16D
	long tmp = idx;
	for (int d=ndim-1; d>=0; d--) {
		coords[d] = tmp % shape[d];
		tmp /= shape[d];
	}

	int j = coords[axis];
	int n = shape[axis];
	int dist = j - i;

	// Only fill elements that are on the specified side of index i
	if ((mode == 0 && dist > 0) || (mode == 1 && dist < 0)) {
		// Get the value at index i along this axis
		coords[axis] = i;
		long offset_src = 0;
		for (int d=0; d<ndim; d++) offset_src += coords[d]*strides[d];
		float fill_value = arr[offset_src/sizeof(float)];

		// Fill current position with that value
		coords[axis] = j;  // restore original coordinate
		long offset_dst = 0;
		for (int d=0; d<ndim; d++) offset_dst += coords[d]*strides[d];
		arr[offset_dst/sizeof(float)] = fill_value;
	}
}
''', 'fill_simple')

def repeat(arr: cp.ndarray, i: int, axis: int=0, mode: str="out", out: cp.ndarray=None):
	"""
	Fill array elements with the value at index i along the specified axis.

	Parameters:
	- arr: input array
	- i: index along axis whose value will be used for filling
	- axis: axis along which to perform the fill
	- mode: 'out' fills elements after index i, 'in' fills elements before index i
	- out: output array. If None, creates a copy. If provided, must have same shape as arr.

	Returns:
	- filled array (either new array or the out parameter)

	For mode='in' with 1D array [1,2,3,4,5,6,7] and i=3:
	Elements 0,1,2 (before index 3) get filled with value at index 3 (which is 4)
	Result: [4,4,4,4,5,6,7]
	"""
	# Handle output array
	if out is None:
		result = arr.copy()
	else:
		if out.shape != arr.shape:
			raise ValueError(f"output array shape {out.shape} does not match input shape {arr.shape}")
		if out.dtype != arr.dtype:
			raise ValueError(f"output array dtype {out.dtype} does not match input dtype {arr.dtype}")
		result = out
		# Copy input to output if they're different arrays
		if result is not arr:
			cp.copyto(result, arr)

	# Handle negative axis
	if axis < 0:
		axis += result.ndim
	if axis < 0 or axis >= result.ndim:
		raise IndexError(f"axis {axis} is out of bounds for array of dimension {result.ndim}")

	n = result.shape[axis]

	# Handle negative index
	if i < 0:
		i += n
	if i < 0 or i >= n:
		raise IndexError(f"index {i} is out of bounds for axis {axis} with size {n}")

	size = result.size
	threads = 256
	blocks = (size + threads - 1) // threads

	shape = cp.array(result.shape, dtype=cp.int64)
	strides = cp.array(result.strides, dtype=cp.int64)

	if mode == "out":
		mode_flag = 0
	elif mode == "in":
		mode_flag = 1
	else:
		raise ValueError("mode must be 'out' or 'in'")

	repeat_kernel((blocks,), (threads,),
				(result, result.ndim, shape, strides, axis, i, mode_flag))
	return result

