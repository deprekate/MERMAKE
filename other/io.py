import os
import queue
import threading
import dask.array as da
import cupy as cp
from mermake.io import *
# Sentinel object for stopping threads
import concurrent.futures
import time

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
def stream_based_prefetcher(hybs, fovs):
    """
    Use dedicated CUDA streams to achieve truly asynchronous operations
    """
    # Create two streams for alternating operations
    stream1 = cp.cuda.Stream(non_blocking=True)
    stream2 = cp.cuda.Stream(non_blocking=True)
    streams = [stream1, stream2]
    
    # Build list of files
    file_list = []
    for all_flds, fov in zip(hybs, fovs):
        for hyb in all_flds:
            file = os.path.join(hyb, fov)
            file_list.append(file)
    
    if not file_list:
        return
    
    # Preload first file (blocking)
    print("Preloading first file...", flush=True)
    im0 = read_im(file_list[0])
    im0 = im0.compute()
    
    # Function to create containers using specific stream
    def make_containers(im, path, stream_idx):
        results = []
        with streams[stream_idx]:
            for icol in range(im.shape[0]):
                chan = cp.asarray(im[icol])
                container = Container(chan)
                container.path = path
                container.channel = icol
                results.append(container)
        return results
    
    # Start the first transfer on stream1
    containers0 = make_containers(im0, file_list[0], 0)
    
    # For remaining files
    for i in range(1, len(file_list)):
        # Determine which stream to use for current and next operations
        current_stream_idx = (i-1) % 2
        next_stream_idx = i % 2
        
        # Start loading next file while current file is being processed
        import threading
        next_im = [None]
        next_path = file_list[i]
        
        def load_next():
            next_im[0] = read_im(next_path)
            next_im[0] = next_im[0].compute()
        
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
import queue
import time
# Now, a simple but effective prefetching mechanism
def efficient_image_generator(hybs, fovs):
    """Efficient generator that prefetches just one image ahead"""
    # Create a list of all files to process
    all_files = []
    for all_flds, fov in zip(hybs, fovs):
        for hyb in all_flds:
            file = os.path.join(hyb, fov)
            all_files.append(file)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Start loading the first file
        future = executor.submit(read_cim, all_files[0]) if all_files else None
        
        # Process all remaining files
        for i in range(1, len(all_files)):
            # Start loading the next file
            next_future = executor.submit(read_cim, all_files[i])
            
            # Get the result of the current file
            result = future.result()
            yield result
            
            # Move to the next file
            future = next_future
        
        # Don't forget the last file
        if future:
            yield future.result()



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
	
	for cim in efficient_image_generator(hybs, fovs):
		print(f"Processing image: {cim[0].path}", flush=True)
	
		# Your minimal processing code
		for icol in [0,1,2]:
			pass
