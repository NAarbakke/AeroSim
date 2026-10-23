"""Errors and the optional-dependency guard."""

from __future__ import annotations

import importlib
from types import ModuleType


class AerosimError(Exception):
    """Base class for aerosim errors."""


class OptionalDependencyError(AerosimError, ImportError):
    """Raised when a backend needs a library that isn't installed."""


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or raise with the pip extra that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise OptionalDependencyError(
            f"'{module}' is required for this feature. Install it with: "
            f"pip install -e '.[{extra}]' (from the AeroSim repo root)"
        ) from e
