#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
#from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils import sysconfig
import subprocess

def readme():
	with open("README.md", "r") as fh:
		long_desc = fh.read()
	return long_desc

def get_version():
	with open("VERSION", 'r') as f:
		v = f.readline().strip()
		return v

def cupy_package():
	cuda_version = os.getenv("CUDA_VERSION", "")  # Optionally set this before installing
	if not cuda_version:
		try:
			import subprocess
			result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
			if "CUDA Version" in result.stdout:
				cuda_version = result.stdout.split("CUDA Version: ")[1].split()[0]
		except:
			pass  # Assume no CUDA installed if nvidia-smi fails

	if cuda_version.startswith("12"):
		return "cupy-cuda12x"
	elif cuda_version.startswith("11"):
		return "cupy-cuda11x"
	elif cuda_version.startswith("10.2"):
		return "cupy-cuda102"
	return "cupy"  # Fallback to CPU-only


def main():
	setup (
		name = 'mermake',
		version = get_version(),
		author = "Katelyn McNair",
		author_email = "deprekate@gmail.com",
		description = 'Code to process merfish data',
		long_description = readme(),
		long_description_content_type="text/markdown",
		url =  "https://github.com/deprekate/mermake",
		#scripts=['mermake.py'],
		entry_points={'console_scripts': ['mermake = mermake.__main__:main']},
		classifiers=[
			"Programming Language :: Python :: 3",
			"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
			"Operating System :: OS Independent",
		],
		python_requires='>3.5.2',
		packages=find_packages(),
		include_package_data=True,
		package_data={"mermake": ["VERSION", "*.cu"]},
		install_requires=['dask','zarr','numpy', cupy_package()],

		#ext_modules = [module],
	)


if __name__ == "__main__":
	main()

