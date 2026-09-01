"""Public API for DeepVarLog."""

from .core import configure, record
from .threadlog import monitor

__all__ = ["configure", "monitor", "record"]
