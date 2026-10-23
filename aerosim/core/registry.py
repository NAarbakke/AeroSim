"""Plugin registry: maps (category, name) -> factory.

Backends register lazily, so importing aerosim never imports RocketPy, JSBSim,
pyCycle, etc. A factory is only called (and its heavy import only happens) when
a mission config actually asks for it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

CATEGORIES = ("combustion", "rocket_engine", "airbreathing", "atmosphere", "flight")

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {c: {} for c in CATEGORIES}


def register(category: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if category not in _REGISTRY:
        raise KeyError(f"unknown category {category!r}; expected one of {CATEGORIES}")

    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY[category]:
            raise KeyError(f"{category}/{name} already registered")
        _REGISTRY[category][name] = factory
        return factory

    return deco


def create(category: str, name: str, /, **kwargs: Any) -> Any:
    try:
        factory = _REGISTRY[category][name]
    except KeyError:
        avail = ", ".join(sorted(_REGISTRY.get(category, {}))) or "none"
        raise KeyError(f"no backend {category}/{name}; available: {avail}") from None
    return factory(**kwargs)


def available() -> dict[str, list[str]]:
    return {c: sorted(v) for c, v in _REGISTRY.items()}
