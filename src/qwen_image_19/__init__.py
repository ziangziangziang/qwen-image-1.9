"""Qwen-Image 1.9 — unified checkpoint pipeline."""

__all__ = ["__version__"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qwen-image-1-9")
except PackageNotFoundError:
    __version__ = "0.3.0"  # fallback for uninstalled runs
