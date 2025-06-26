import subprocess
import sys
import warnings

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
