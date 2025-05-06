import os
import queue
import threading
import dask.array as da
import cupy as cp
from mermake.deconvolver import Deconvolver
from mermake.maxima import find_local_maxima
import mermake.blur as blur
from mermake.io import get_files, image_generator, Container, read_im
import concurrent.futures
import time

import numpy as np

import os
import numpy as np
import zarr
import cupy as cp
import concurrent.futures
import queue
import time
import threading
from contextlib import contextmanager
import os
import numpy as np
import zarr
import cupy as cp
import concurrent.futures
import queue
import time
import threading
from contextlib import contextmanager
class Container:
    def __init__(self, data, **kwargs):
        """Store the array and metadata"""
        self.data = data
        self.metadata = kwargs
        
    def __getitem__(self, item):
        """Allow indexing into the container"""
        return self.data[item]
        
    def __array__(self):
        """Return the underlying array"""
        return self.data
        
    def __repr__(self):
        """Custom string representation"""
        return f"Container(shape={self.data.shape}, dtype={self.data.dtype}, metadata={self.metadata})"
        
    def __getattr__(self, name):
        """Delegate attribute access to the underlying data"""
        if hasattr(self.data, name):
            return getattr(self.data, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def clear(self):
        """Explicitly delete the CuPy array and synchronize"""
        if hasattr(self, 'data') and self.data is not None:
            # Ensure we're synchronized before deletion
            cp.cuda.runtime.deviceSynchronize()
            del self.data
            self.data = None
            # Try to free memory using the appropriate method
            try:
                cp._default_memory_pool.free_all_blocks()
            except:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass

def read_cim(path):
    """Read image and transfer to GPU"""
    im = read_im(path)
    
    # Ensure we're using the default device
    with cp.cuda.Device(0):
        # Force a memory cleanup before loading a new image
        try:
            cp._default_memory_pool.free_all_blocks()
        except:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
        
        # Transfer to GPU
        cim = cp.asarray(im)
        
        # Make sure the transfer is complete
        cp.cuda.runtime.deviceSynchronize()
    
    container = Container(cim)
    container.path = path
    return container

class ImageQueue:
    """Simple but effective asynchronous image queue"""
    
    def __init__(self, files, max_workers=2, retry_count=3):
        """
        Initialize the image queue with files to process.
        
        Args:
            files: Iterable of file paths
            max_workers: Number of worker threads
            retry_count: Number of times to retry loading an image if it fails
        """
        self.files = list(files) if not isinstance(files, list) else files
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.retry_count = retry_count
        
        # Preload the first image with retries
        try:
            first_file = self.files.pop(0)
        except IndexError:
            raise ValueError("No image files provided.")
        
        # Try loading with retries
        first_image = None
        error = None
        
        for attempt in range(self.retry_count):
            try:
                future = self.executor.submit(read_cim, first_file)
                first_image = future.result()
                error = None
                break
            except Exception as e:
                error = e
                print(f"Attempt {attempt+1} failed for {first_file}: {str(e)}")
                sleep(1)  # Small delay before retry
        
        if first_image is None:
            raise ValueError(f"Failed to load first image after {self.retry_count} attempts: {str(error)}")
        
        self._first_image = first_image
        self.shape = self._first_image.shape
        self.dtype = self._first_image.dtype
        
        # Start prefetching the next image if there are more files
        self.future = None
        if self.files:
            next_file = self.files.pop(0)
            self.future = self.executor.submit(read_cim, next_file)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Get the next image, with prefetching of the subsequent one"""
        # Return the preloaded first image if it's available
        if self._first_image is not None:
            image = self._first_image
            self._first_image = None
            return image
        
        # If no future is pending, we're done
        if self.future is None:
            raise StopIteration
        
        # Try to get the current future's result
        try:
            image = self.future.result()
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            # If there are more files, skip this one and try another
            if self.files:
                next_file = self.files.pop(0)
                self.future = self.executor.submit(read_cim, next_file)
                return self.__next__()  # Recursive call to try again
            else:
                self.future = None
                raise StopIteration
        
        # Prefetch the next image in the background
        self.future = None
        if self.files:
            next_file = self.files.pop(0)
            self.future = self.executor.submit(read_cim, next_file)
        
        return image
    
    def close(self):
        """Clean up resources"""
        if self._first_image is not None:
            self._first_image.clear()
            self._first_image = None
        
        # Cancel any pending future
        if self.future is not None and not self.future.done():
            self.future.cancel()
        
        self.executor.shutdown(wait=False)
        
        # Final memory cleanup
        try:
            cp._default_memory_pool.free_all_blocks()
        except:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

def buffered_gpu_loader(hybs, fovs):
	"""
	Use pre-allocated GPU buffers to avoid allocation/deallocation stalls
	"""
	# Build list of files
	file_list = []
	for all_flds, fov in zip(hybs, fovs):
		for hyb in all_flds:
			file = os.path.join(hyb, fov)
			file_list.append(file)

	if not file_list:
		return

	# Preload first file to determine shapes
	sample_im = read_im(file_list[0])
	sample_im = sample_im.compute()

	# Create fixed GPU buffers (one for each channel)
	n_channels = sample_im.shape[0]
	buffer_shape = sample_im.shape[1:]  # z, y, x

	# Pre-allocate two sets of buffers for double-buffering
	gpu_buffers = []
	for i in range(2):
		channel_buffers = []
		for j in range(n_channels):
			buffer = cp.empty(buffer_shape, dtype=sample_im.dtype)
			channel_buffers.append(buffer)
		gpu_buffers.append(channel_buffers)

	# Create worker pool for loading images
	with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
		# Start loading the first file
		future = executor.submit(read_im, file_list[0])

		for i in range(len(file_list)):
			# Current buffer set index
			buf_idx = i % 2

			# Get the CPU image data
			if i == 0:
				cpu_im = sample_im  # Reuse the sample we already loaded
			else:
				cpu_im = future.result().compute()

			# Start loading the next file if available
			if i + 1 < len(file_list):
				future = executor.submit(read_im, file_list[i + 1])

			# Copy each channel to its pre-allocated GPU buffer
			containers = []
			for j in range(n_channels):
				# Copy to GPU buffer without allocating new memory
				gpu_buffers[buf_idx][j].set(cpu_im[j])

				# Create a container with metadata but using the pre-allocated buffer
				container = Container(gpu_buffers[buf_idx][j])
				container.path = file_list[i]
				container.channel = j
				containers.append(container)

			# Yield the containers
			yield containers
def stream_based_prefetcher(files):
	"""
	Use dedicated CUDA streams to achieve truly asynchronous operations
	"""
	# Create two streams for alternating operations
	stream1 = cp.cuda.Stream(non_blocking=True)
	stream2 = cp.cuda.Stream(non_blocking=True)
	streams = [stream1, stream2]
	
	# Preload first file (blocking)
	im0 = read_im(files[0])
	
	# Function to create containers using specific stream
	def make_containers(im, path, stream_idx):
		results = []
		with streams[stream_idx]:
			channels = cp.asarray(im)
			container = Container(channels)
			container.path = path
			results = container
		return results
	
	# Start the first transfer on stream1
	containers0 = make_containers(im0, files[0], 0)
	
	# For remaining files
	for i in range(1, len(files)):
		# Determine which stream to use for current and next operations
		current_stream_idx = (i-1) % 2
		next_stream_idx = i % 2
		
		# Start loading next file while current file is being processed
		import threading
		next_im = [None]
		next_path = files[i]
		
		def load_next():
			next_im[0] = read_im(next_path)
		
		# Start loading next file in background
		load_thread = threading.Thread(target=load_next)
		load_thread.start()
		
		# Yield current containers (previous iteration's results)
		yield containers0
		
		# Wait for next image to be loaded to RAM
		load_thread.join()
		
		# Start transfer of next image on the alternate stream
		# This will overlap with processing of current image
		containers0 = make_containers(next_im[0], next_path, next_stream_idx)
		
		# Clear reference to free CPU memory
		next_im[0] = None

def async_gpu_prefetcher(hybs, fovs):
	"""
	Asynchronously prefetch images to GPU without blocking
	"""
	# Build list of files
	file_list = []
	for all_flds, fov in zip(hybs, fovs):
		for hyb in all_flds:
			file = os.path.join(hyb, fov)
			file_list.append(file)
	
	if not file_list:
		return
		
	# Split the file loading and GPU transfer into separate steps
	def load_to_ram(file_path):
		"""Load image to RAM only"""
		im = read_im(file_path)
		im = im.compute()  # This is CPU-bound and can run in a thread
		return (im, file_path)
	
	def transfer_to_gpu(ram_data):
		"""Transfer from RAM to GPU without synchronizing"""
		im, path = ram_data
		channel_containers = []
		
		for icol in range(im.shape[0]):
			# Transfer to GPU without synchronizing
			chan = cp.asarray(im[icol])
			container = Container(chan)
			container.path = path
			container.channel = icol
			channel_containers.append(container)
		
		return channel_containers
	
	with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
		# Start RAM loaders for first two files
		ram_futures = []
		for i in range(min(2, len(file_list))):
			ram_futures.append(executor.submit(load_to_ram, file_list[i]))
		
		# Start GPU transfer for first file
		if ram_futures:
			gpu_future = executor.submit(transfer_to_gpu, ram_futures[0].result())
			ram_futures.pop(0)
		else:
			gpu_future = None
			
		# Process remaining files
		for i in range(len(file_list)):
			# Get current GPU result
			result = gpu_future.result() if gpu_future else None
			
			# Start next RAM loader if needed
			if i + 2 < len(file_list):
				ram_futures.append(executor.submit(load_to_ram, file_list[i + 2]))
			
			# Start next GPU transfer if RAM data is ready
			if ram_futures:
				gpu_future = executor.submit(transfer_to_gpu, ram_futures[0].result())
				ram_futures.pop(0)
			else:
				gpu_future = None
			
			# Yield current result
			if result:
				yield result

import concurrent.futures
def image_generator00(hybs, fovs):
	"""Generator that prefetches the next image while processing the current one."""
	with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
		future = None
		all_files = [os.path.join(hyb, fov) for all_flds, fov in zip(hybs, fovs) for hyb in all_flds]
		if not all_files:
			return

		# Submit the first task
		future = executor.submit(read_im, all_files[0])

		for file in all_files[1:]:
			next_future = executor.submit(read_im, file)
			result = future.result()
			if isinstance(result, tuple):
				yield (result[0].compute(), *result[1:])
			else:
				yield result.compute()
			future = next_future

		# Yield the last pending future
		result = future.result()
		if isinstance(result, tuple):
			yield (result[0].compute(), *result[1:])
		else:
			yield result.compute()


if __name__ == "__main__":
	# Define your hybrid (hybs) and field of view (fovs) directories
	master_data_folders = ['/data/07_22_2024__PFF_PTBP1']
	iHm = 1
	iHM = 16
	shape = (4, 40, 3000, 3000)
	items = [(set_, ifov) for set_ in ['_set1'] for ifov in range(1, 5)]
	
	hybs = list()
	fovs = list()
	for item in items[:4]:
		all_flds, fov = get_files(master_data_folders, item, iHm=iHm, iHM=iHM)
		hybs.append(all_flds)
		fovs.append(fov)

	paths = list()
	for all_flds, fov in zip(hybs, fovs):
		for hyb in all_flds:
			file = os.path.join(hyb, fov)
			paths.append(file)

	#paths = ['/data/07_22_2024__PFF_PTBP1/H2_AER_set1/Conv_zscan1_002.zarr'] * 20
	
	psf_file = '../psfs/dic_psf_60X_cy5_Scope5.pkl'
	psfs = np.load(psf_file, allow_pickle=True)
	key = (0,1500,1500)
	psfs = { key : psfs[key] }
	deconvolver = Deconvolver(psfs, shape[1:], tile_size=300, overlap=89, zpad=39, beta=0.0001)
				
	import time
	print('start')	
	start = time.time()
	for cim in stream_based_prefetcher(hybs, fovs):
		end = time.time()
		#print(f"Processing image: {cim.path}", flush=True)
		print()
		print(f"time: {end - start:.6f} seconds")
		# Your minimal processing code
		for icol in [0,1,2,3]:
			print(icol, end=' ', flush=True)
			view = cim[icol]
			for tile in deconvolver.tile_wise(view):
				pass
		print('', flush=True)
		start = time.time()
