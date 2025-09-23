"""Utilities for accessing bundled JSON schemas."""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import BinaryIO, Iterator


def iter_schema_paths() -> Iterator[Path]:
    """Yield packaged schema file paths."""
    package = __name__
    for name in resources.contents(package):
        if not name.endswith(".json"):
            continue
        with resources.path(package, name) as path:
            yield path


def load_schema(name: str) -> BinaryIO:
    """Open a schema resource for reading."""
    return resources.open_binary(__name__, name)


__all__ = ["iter_schema_paths", "load_schema"]
