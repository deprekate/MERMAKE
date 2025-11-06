import cupy as cp

# this one clamps the output to have a min of 0 instead of rolling over
subtract_clamp_kernel = cp.ElementwiseKernel(
    'uint16 x, uint16 y',
    'uint16 z',
    '''
    z = (x > y) ? (x - y) : 0;
    ''',
    'subtract_clamp_kernel'
)

def subtract_clamp(x, y, out=None):
    if out is None:
        return subtract_clamp_kernel(x, y)
    subtract_clamp_kernel(x, y, out)
    return out

# this one reflects the output back awway from 0 instead of rolling over
subtract_reflect_kernel = cp.ElementwiseKernel(
    'uint16 x, uint16 y',
    'uint16 z',
    '''
    z = (x >= y) ? (x - y) : (y - x);
    ''',
    'subtract_reflect_kernel'
)

def subtract_reflect(x, y, out=None):
    if out is None:
        return subtract_reflect_kernel(x, y)
    subtract_reflect_kernel(x, y, out)
    return out

