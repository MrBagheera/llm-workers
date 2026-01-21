"""Console UI components for LLM Workers - interactive chat and rich terminal output."""

import importlib.metadata

from .chat import ChatSession, chat_with_llm_script

try:
    # Fetch version from the installed package metadata
    __version__ = importlib.metadata.version("my_package")
except importlib.metadata.PackageNotFoundError:
    # Handle cases where package is not installed (e.g. local dev)
    __version__ = "unknown"
