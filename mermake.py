import os
import sys
import argparse
import contextlib
from time import sleep,time
from typing import Generator
# Try to import the appropriate TOML library
if sys.version_info >= (3, 11):
	import tomllib  # Python 3.11+ standard library
else:
	import tomli as tomllib  # Backport for older Python versions
from types import SimpleNamespace

import psutil
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

import numpy as np
from blessed import Terminal
from dashing import HSplit,VSplit,Text,Log,HGauge,Grext
from graphic import Graphic  # Assuming the class is saved as graphic.py

from coords import points
from coords import points_to_coords

from mermake.io import set_data
sys.path.pop(0)
#sys.path.append('mermake')

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
def worker(procnum):
	sleep(2)
	return 'str',procnum


def dict_to_namespace(d):
	"""Recursively convert dictionary into SimpleNamespace."""
	for key, value in d.items():
		if isinstance(value, dict):
			value = dict_to_namespace(value)  # Recursively convert nested dictionaries
		elif isinstance(value, list):
			value = [dict_to_namespace(item) if isinstance(item, dict) else item for item in value]  # Handle lists of dicts
		d[key] = value
	return SimpleNamespace(**d)

toml_text = "
[paths]
codebook = '/home/katelyn/develop/MERMAKE/codebooks/codebook_code_color2__ExtraAaron_8_6_blank.csv' ### 
psf_file = '/home/katelyn/develop/MERMAKE/psfs/dic_psf_60X_cy5_Scope5.pkl'  ### Scope5 psf
flat_field_tag = '/home/katelyn/develop/MERMAKE/flat_field/Scope5_'
hyb_range = 'H1_AER_set1:H16_AER_set1'
hyb_folders = [
                '/data/07_22_2024__PFF_PTBP1',
                ]
output_folder = '/home/katelyn/develop/MERMAKE/MERFISH_Analysis_AER'

#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#           you probably dont have to change any of the settings below                  #
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
[hybs]
tile_size = 300
overlap = 89
beta = 0.0001
threshold = 3600
blur_radius = 30
delta = 1
delta_fit = 3
sigmaZ = 1
sigmaXY = 1.5

[dapi]
tile_size = 300
overlap = 89
beta = 0.01
threshold = 3.0
blur_radius = 50
delta = 5
delta_fit = 5
sigmaZ = 1
sigmaXY = 1.5
"

# Validator and loader for the TOML file
def is_valid_file(path):
	if not os.path.exists(path):
		raise argparse.ArgumentTypeError(f"{path} does not exist.")
	try:
		with open(path, "rb") as f:
			config = tomllib.load(f)
			return config
	except Exception as e:
		raise argparse.ArgumentTypeError(f"Error loading TOML file {path}: {e}")

class CustomArgumentParser(argparse.ArgumentParser):
	def error(self, message):
		# Customizing the error message
		if "the following arguments are required: config" in message:
			message = message.replace("config", "config.toml")
		message += toml_text
		super().error(message)


@contextlib.contextmanager
def open_terminal() -> Generator:
	"""
	Helper function that creates a Blessed terminal session to restore the screen after
	the UI closes.
	"""
	t = Terminal()

	with t.fullscreen(), t.hidden_cursor():
		yield t

class Grid(list):
	def __init__(self, rows, cols):
		self.flip = False
		if rows > cols:
			self.flip = True
		# Define grid dimensions
		self.cell_width = 2
		self.cell_height = 1
		# Define grid characters
		self.char_blank = "░"
		self.path_char = "■"  # Character to highlight the path
		for row in range(rows):  # Include last line for grid border
			line = []
			for col in range(cols):  # Include last column for grid border
				line.append(self.char_blank)
				line.append(self.char_blank)
			line.append('\n')
			self.append(line)
	def set(self, row, col, char):
		self[row * self.cell_height + 1][ col * self.cell_width + 1] = char
	def __repr__(self):
		if self.flip:
			# Transpose the grid (excluding newline characters)
			transposed = list(map(list, zip(*[row[:-1] for row in self])))
			return '\n'.join(''.join(row) for row in transposed)
		else:
			return ''.join(''.join(item) for item in self)


class ColorLog(Log):
	def __init__(self, title="", border_color=7, text_color=None, *args, **kwargs):
		super().__init__(title=title, border_color=border_color, *args, **kwargs)
		self.term = Terminal()
		self.text_color = text_color or self.term.white  # Default text color is white

	def append(self, message):
		"""Append a message with the specified text color."""
		colored_message = self.text_color(message)
		super().append(colored_message)
class SmallHGauge(HGauge):
    def draw(self, stdscr, colors, x, y, w, h):
        # Only use a small portion of the height
        small_h = 3  # Just enough for the gauge and borders
        super().draw(stdscr, colors, x, y, small_h, w)

if __name__ == "__main__":
	usage = '%s [-opt1, [-opt2, ...]] settings.toml' % __file__
	#parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter, usage=usage)
	parser = CustomArgumentParser(description='',formatter_class=argparse.RawTextHelpFormatter,usage=usage)
	parser.add_argument('settings', type=is_valid_file, help='settings file')
	#parser.add_argument('-c', '--check', action="store_true", help="Check a single zarr")
	args = parser.parse_args()
	# Convert settings to namespace and attach each top-level section to args
	for key, value in vars(dict_to_namespace(args.settings)).items():
		setattr(args, key, value)
	#----------------------------------------------------------------------------#
	set_data(args)
	
	maxx = max(grid[0] for sset in args.batch.values() for fov in sset.values() for grid in [fov['grid_position']])
	maxy = max(grid[1] for sset in args.batch.values() for fov in sset.values() for grid in [fov['grid_position']])

	grid = Grid(maxx+2, maxy+2)
	# Create the terminal layout
	ui = HSplit(
			Grext(str(grid), title='Grext', color=1, border_color=1),
			VSplit(
				SmallHGauge(label="cpu usage ", val=20, border_color=5),
				SmallHGauge(label="gpu ram usage ", val=20, border_color=5),
				SmallHGauge(label="gpu utilization ", val=20, border_color=5),
				ColorLog(title="logs", border_color=5, text_color=Terminal().white),
			)
			#title='Dashing',
	)

	# Access the Graphic tile
	graphic_tile = ui.items[0]
	hgauge0 = ui.items[1].items[0]
	hgauge1 = ui.items[1].items[1]
	hgauge2 = ui.items[1].items[2]
	log_tile = ui.items[1].items[3]
	log_tile.append("Logs")


	prev_time = time()
	terminal = Terminal()
	with terminal.fullscreen(), terminal.hidden_cursor():
		log_tile.append("Checking xml data....")
		for sset in sorted(args.batch):
			for fov in sorted(args.batch[sset]):
				coord = args.batch[sset][fov]['grid_position']
				point = args.batch[sset][fov]['stage_position']
				grid.set(coord[0], coord[1],  terminal.yellow('■'))
				graphic_tile.text = str(grid)
				log_tile.append('Processing: fov' + str(coord))
				hgauge0.value = psutil.cpu_percent(interval=1)
				mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
				mem_total = mem_info.total # / (1024 ** 3)
				mem_used = mem_info.used #/ (1024 ** 3)
				mem_free = mem_info.free #/ (1024 ** 3)
				gpu = pynvml.nvmlDeviceGetUtilizationRates(handle)
				hgauge1.value = 100 * mem_used / mem_total
				hgauge2.value = gpu.gpu
				
				ui.display()
				sleep(1.0/10)
		log_tile.append("Checking hyb data....")
		with ProcessPoolExecutor(max_workers=5) as executor:
			future_to_task = dict()
			i = 0
			for sset in sorted(args.batch):
				for fov in sorted(args.batch[sset]):
					block = args.batch[sset][fov]['grid_position']
					future_to_task[executor.submit(worker, coord)] = i
					i += 1

			for future in as_completed(future_to_task):  # Process results as they complete
				i, coord = future.result()
				log_tile.append(f"Task {i} completed with result: {coord}")
				grid.set(coord[0], coord[1],  terminal.green('■'))
				graphic_tile.text = str(grid)
				ui.display()


			log_tile.append(f"Done!")
			ui.display()
			while True:
				sleep(1)


