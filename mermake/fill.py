import cupy as cp

reflect_kernel = cp.RawKernel(r'''
extern "C" __global__
void mirror(float* arr, const int ndim, const long* shape,
            const long* strides, const int axis, const int i) {
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
    if (dist > 0) {
        int mirror_j = i - dist;
        if (mirror_j >= 0) {
            coords[axis] = mirror_j;
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
    }
}
''', 'mirror')

reflect_kernel = cp.RawKernel(r'''
extern "C" __global__
void mirror(float* arr, const int ndim, const long* shape,
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

def reflect0(arr: cp.ndarray, i: int, axis: int=0):
    axis %= arr.ndim
    n = arr.shape[axis]
    if i < 0: i += n
    if i < 0 or i >= n:
        raise IndexError("index out of bounds")

    size = arr.size
    threads = 256
    blocks = (size + threads - 1) // threads

    shape = cp.array(arr.shape, dtype=cp.int64)
    strides = cp.array(arr.strides, dtype=cp.int64)

    reflect_kernel((blocks,), (threads,),
                  (arr, arr.ndim, shape, strides, axis, i))
    return arr
def reflect(arr: cp.ndarray, i: int, axis: int=0, mode: str="out"):
    axis %= arr.ndim
    n = arr.shape[axis]
    if i < 0: i += n
    if i < 0 or i >= n:
        raise IndexError("index out of bounds")

    size = arr.size
    threads = 256
    blocks = (size + threads - 1) // threads

    shape = cp.array(arr.shape, dtype=cp.int64)
    strides = cp.array(arr.strides, dtype=cp.int64)

    if mode == "out":
        mode_flag = 0
    elif mode == "in":
        mode_flag = 1
    else:
        raise ValueError("mode must be 'out' or 'in'")

    reflect_kernel((blocks,), (threads,),
                  (arr, arr.ndim, shape, strides, axis, i, mode_flag))
    return arr

