"""AlayaRenderer-Flash: real-time G-buffer-conditioned generative rendering."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("alayarenderer-flash")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
