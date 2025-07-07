import subprocess
import sys
import warnings
from pathlib import Path

def _check_cuda():
    message = (
        "\n" + "=" * 80 + "\n"
        + "=" * 80 + "\n"
        + "=" * 80 + "\n"
        "WARNING: This package uses CuPy, which requires the CUDA Toolkit.\n"
        "Make sure you have installed the correct version of CUDA for your GPU.\n"
        "See: https://docs.cupy.dev/en/stable/install.html\n"
        + "=" * 80 + "\n"
        + "=" * 80 + "\n"
        + "=" * 80
    )

    try:
        # Using nvvcc to force an error
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception("NVCC check failed")
    except Exception:
        print(message, file=sys.stderr)
        sys.exit(1)  # Exit installation with error

# Run the check at import time
_check_cuda()

def _get_version():
    # Try to find VERSION file relative to this module
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()

    # Fall back to looking in parent directory (for development)
    version_file = Path(__file__).parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()

    return "unknown"

__version__ = _get_version()
