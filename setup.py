"""Build helper for the optional _usb_accel C extension.

If libusb-1.0-dev is installed, the extension compiles automatically
during ``pip install``. If not, the package installs as pure Python
and falls back to pyusb for USB communication.
"""

import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """build_ext that treats all extensions as optional.

    If compilation fails (e.g., missing libusb-1.0-dev), the install
    proceeds without the C extension.
    """

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(
                f"\nWARNING: Failed to build {ext.name}: {e}\n"
                "Falling back to pure-Python pyusb transport.\n"
                "Install libusb-1.0-dev to enable the C acceleration extension.\n",
                file=sys.stderr,
            )


def _get_libusb_flags():
    """Get compiler/linker flags for libusb-1.0 via pkg-config."""
    try:
        cflags = subprocess.check_output(
            ["pkg-config", "--cflags", "libusb-1.0"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split()
        libs = subprocess.check_output(
            ["pkg-config", "--libs", "libusb-1.0"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split()
        return cflags, libs
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def _build_ext_modules():
    """Return ext_modules list, or empty list if libusb not available."""
    # Source path must be relative to setup.py (setuptools requirement).
    # setup.py is at the repo root; _usb_accel.c is in libredgetpu/.
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    c_source_abs = os.path.join(setup_dir, "libredgetpu", "_usb_accel.c")
    if not os.path.isfile(c_source_abs):
        return []

    cflags, libs = _get_libusb_flags()
    if cflags is None:
        print(
            "NOTE: pkg-config cannot find libusb-1.0; "
            "_usb_accel C extension will not be built.\n"
            "Install libusb-1.0-dev to enable it.",
            file=sys.stderr,
        )
        return []

    c_source_rel = os.path.join("libredgetpu", "_usb_accel.c")
    return [Extension(
        "libredgetpu._usb_accel",
        sources=[c_source_rel],
        extra_compile_args=cflags + ["-O2"],
        extra_link_args=libs,
    )]


setup(
    ext_modules=_build_ext_modules(),
    cmdclass={"build_ext": OptionalBuildExt},
)
