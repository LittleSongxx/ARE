"""AC-PBGRL research package.

Heavy dependencies are intentionally not imported at package import time so that
configuration, GPU scheduling, and offline deployment tools remain usable before
PyTorch is installed.
"""

__version__ = "0.1.0"
