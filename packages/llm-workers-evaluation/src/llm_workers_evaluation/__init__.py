"""Evaluation framework for LLM Workers - run evaluation suites and report scores."""

import importlib.metadata

from .evaluation import (
    TestResult,
    EvaluationResults,
    run_evaluation,
    format_results,
)

try:
    # Fetch version from the installed package metadata
    __version__ = importlib.metadata.version("my_package")
except importlib.metadata.PackageNotFoundError:
    # Handle cases where package is not installed (e.g. local dev)
    __version__ = "unknown"
