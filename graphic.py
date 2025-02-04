from dashing import Tile
from blessed import Terminal
from typing import Literal, Optional, Tuple, Generator, List, Union
from collections import namedtuple, deque
TBox = namedtuple("TBox", "t x y w h")
class RGB:
    """An RGB color, stored as 3 integers"""

    __slots__ = ["r", "g", "b"]

    def __init__(self, r: int, g: int, b: int) -> None:
        self.r = r
        self.g = g
        self.b = b

    def __iter__(self):
        yield self.r
        yield self.g
        yield self.b

    @classmethod
    def parse(cls, color: str):
        """Parse color expressed in different formats and return an RGB object
        Formats:
            color("#RRGGBB") RGB in hex
            color("*HHSSVV") HSV in hex with values ranging 00 to FF
        """
        if color.startswith("#"):
            return cls(int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))

        if color.startswith("*"):
            h = int(color[1:3], 16) / 255.0
            s = int(color[3:5], 16) / 255.0
            v = int(color[5:7], 16) / 255.0
            return cls(*(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)))

        raise ValueError("Invalid color")

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float):
        """Create an RGB instance from HSV values"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return cls(int(r * 255), int(g * 255), int(b * 255))

    def to_hls(self) -> Tuple[float, float, float]:
        """Convert to HLS"""
        return colorsys.rgb_to_hls(self.r, self.g, self.b)

    def __eq__(self, other) -> bool:
        """Compares 2 colors for equality"""
        if isinstance(other, self.__class__):
            return (self.r, self.g, self.b) == (other.r, other.g, other.b)
        return False

    def __ne__(self, other) -> bool:
        """Compares 2 colors for not-equality"""
        return not self.__eq__(other)

    def pr(self, term) -> str:
        """Returns a printable string element"""
        return term.color_rgb(self.r, self.g, self.b)

class _Buf:
    """Output buffer for internal use"""

    __slots__ = ["_buf"]

    def __init__(self) -> None:
        self._buf: List[str] = []

    def add(self, *i: str) -> None:
        """Append string to buffer"""
        self._buf.extend(i)

    def print(self) -> None:
        """Render buffer on screen"""
        print("".join(self._buf))
Color = Union[int, str, None, RGB]
class Grext(Tile):
	"""
	A multi-line text box. Example::

	   Text('Hello World, this is dashing.', border_color=2),

	"""

	def __init__(self, text: str, color: Color = None, **kw) -> None:
		super().__init__(**kw)
		self.text: str = text

	def _display(self, buf: _Buf, tbox: TBox) -> None:
		tbox = self._draw_borders_and_title(buf, tbox)
		for dx, line in _pad(self.text.splitlines()[-(tbox.h) :], tbox.h):
			buf.add(
				self.text_color.pr(tbox.t),
				tbox.t.move(tbox.x + dx, tbox.y),
				line,
				" " * (tbox.w - len(line)),
			)
	def draw_char(self, x, y, char):
		"""Draw a single character at (x, y) within the graphic window."""
		#if 0 <= x < self.width and 0 <= y < self.height:
		#	self.buffer[y][x] = self.term.white(char)
		pass


class Graphic(Tile):
	def __init__(self, title="", width=30, height=30, *args, **kwargs):
		super().__init__(title=title, *args, **kwargs)
		self.term = Terminal()
		self.width = width
		self.height = height
		self.buffer = [[" " for _ in range(self.width)] for _ in range(self.height)]

	def draw_char(self, x, y, char):
		"""Draw a single character at (x, y) within the graphic window."""
		if 0 <= x < self.width and 0 <= y < self.height:
			self.buffer[y][x] = self.term.white(char)

	def draw_string(self, x, y, string):
		"""Draw a string starting at (x, y) within the graphic window."""
		for i, char in enumerate(string):
			if 0 <= x + i < self.width and 0 <= y < self.height:
				self.buffer[y][x + i] = char

	def clear(self):
		"""Clear the buffer."""
		self.buffer = [[" " for _ in range(self.width)] for _ in range(self.height)]

	def _display(self, tbox, parent):
		"""Render the content of the buffer into the terminal."""
		for row_index, row in enumerate(self.buffer[:tbox.h]):
			# Safely format the row to fit within TBox width
			content = "".join(row[:tbox.w])
			# Position the cursor and write the content
			print(self.term.move_xy(tbox.x, tbox.y + row_index) + content)

