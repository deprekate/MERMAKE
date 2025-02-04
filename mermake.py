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

class Grid

class ColorLog(Log):
	def __init__(self, title="", border_color=7, text_color=None, *args, **kwargs):
		super().__init__(title=title, border_color=border_color, *args, **kwargs)
		self.term = Terminal()
		self.text_color = text_color or self.term.white  # Default text color is white

	def append(self, message):
		"""Append a message with the specified text color."""
		colored_message = self.text_color(message)
		super().append(colored_message)

def joined(lol):
	return ''.join(''.join(l) for l in lol)

# Define grid dimensions
cell_width = 2
cell_height = 1
# Define grid characters
corner_char = "C"
horizontal_char = "H"
vertical_char = "V"
char_blank = "░"
path_char = "■"  # Character to highlight the path

def make_grid(rows, cols):
	#print(term.clear)
	grid = []
	for row in range(rows):  # Include last line for grid border
		line = []
		for col in range(cols):  # Include last column for grid border
			line.append(char_blank)
			line.append(char_blank)
		line.append('\n')
		grid.append(line)
	return grid

if __name__ == "__main__":
	usage = '%s [-opt1, [-opt2, ...]] config.toml' % __file__
	parser = CustomArgumentParser(description='',formatter_class=argparse.RawTextHelpFormatter,usage=usage)
	parser.add_argument('config', type=is_valid_file, help='config file')
	#parser.add_argument('-o', '--outfile', action="store", default=sys.stdout, type=argparse.FileType('w'), help='where to write output [stdout]')
	args = parser.parse_args()
	
	set_data(args)

	maxx = max(grid[0] for sset in args.batch.values() for fov in sset.values() for grid in [fov['grid_position']])
	maxy = max(grid[1] for sset in args.batch.values() for fov in sset.values() for grid in [fov['grid_position']])

	'''
	for sset in args.batch:
		for fov in args.batch[sset]:
			point = args.batch[sset][fov]['stage_position']
			coord = args.batch[sset][fov]['grid_position']
			print(sset, fov, coord[0], coord[1], point[0], point[1], sep='\t')
	exit()
	'''
	#coords = get_coords(args)
	grid = make_grid(maxx+2, maxy+2)
	# Create the terminal layout
	ui = HSplit(
			#HSplit(
				#Graphic(title="Graphic Example", border_color=5, height=maxs[0], width=(2*maxs[1])),
				Grext(joined(grid), title='Grext', color=1, border_color=1),
				ColorLog(title="logs", border_color=5, text_color=Terminal().red)
			#),
			#title='Dashing',
	)

	# Access the Graphic tile
	graphic_tile = ui.items[0]
	log_tile = ui.items[1]
	log_tile.append("Logs")

	# Draw some characters and strings
	#graphic_tile.draw_char(2, 2, "@")
	#graphic_tile.draw_char(3, 3, "#")
	#graphic_tile.draw_string(5, 5, "Hello, Dashing!")

	#input("Press Enter to clear the graphic...")

	# Clear the graphic and render again
	#graphic_tile.clear()
	#ui.display()

	#draw_grid(*maxs, graphic_tile)

	prev_time = time()
	terminal = Terminal()
	with terminal.fullscreen(), terminal.hidden_cursor():
		for sset in args.batch:
			for fov in args.batch[sset]:
				coord = args.batch[sset][fov]['grid_position']
				ui.display()
				grid[coord[0] * cell_height + 1][ coord[1] * cell_width + 1] = terminal.green(path_char)
				graphic_tile.text = joined(grid)
				sleep(1.0/5)
				'''
				while True:
					ui.display()
					sleep(1.0 / 25)
					t = int(time())
					if t != prev_time:
						log_tile.append("%s" % t)
						prev_time = t
				'''
		while True:
			sleep(1)


