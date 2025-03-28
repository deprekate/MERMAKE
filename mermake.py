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


import numpy as np
from blessed import Terminal
from dashing import HSplit,VSplit,Log,Grext
from graphic import Graphic  # Assuming the class is saved as graphic.py

from coords import points
from coords import points_to_coords

sys.path.pop(0)
sys.path.append('mermake')
from utils import *

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


if __name__ == "__main__":
	usage = '%s [-opt1, [-opt2, ...]] config.toml' % __file__
	parser = CustomArgumentParser(description='',formatter_class=argparse.RawTextHelpFormatter,usage=usage)
	parser.add_argument('config', type=is_valid_file, help='config file')
	#parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	args = parser.parse_args()

	set_data(args)
	'''
	for sset in sorted(args.batch):
		for fov in sorted(args.batch[sset]):
			block = args.batch[sset][fov]
			print(block)
			exit()
	#count_hybs(args)
	'''
	maxx = max(grid[0] for sset in args.batch.values() for fov in sset.values() for grid in [fov['grid_position']])
	maxy = max(grid[1] for sset in args.batch.values() for fov in sset.values() for grid in [fov['grid_position']])

	grid = Grid(maxx+2, maxy+2)
	# Create the terminal layout
	ui = HSplit(
			#HSplit(
				#Graphic(title="Graphic Example", border_color=5, height=maxs[0], width=(2*maxs[1])),
				Grext(str(grid), title='Grext', color=1, border_color=1),
				ColorLog(title="logs", border_color=5, text_color=Terminal().red)
			#),
			#title='Dashing',
	)

	# Access the Graphic tile
	graphic_tile = ui.items[0]
	log_tile = ui.items[1]
	log_tile.append("Logs")

	#input("Press Enter to clear the graphic...")

	# Clear the graphic and render again
	#graphic_tile.clear()
	#ui.display()

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
				log_tile.append(str(coord))
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


