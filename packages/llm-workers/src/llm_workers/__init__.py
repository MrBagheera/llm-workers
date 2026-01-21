"""LLM Workers Core Library"""

import importlib.metadata

try:
    # Fetch version from the installed package metadata
    __version__ = importlib.metadata.version("my_package")
except importlib.metadata.PackageNotFoundError:
    # Handle cases where package is not installed (e.g. local dev)
    __version__ = "unknown"

