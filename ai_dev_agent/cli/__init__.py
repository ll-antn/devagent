"""CLI package exposing the DevAgent command entry points."""
from __future__ import annotations

import sys
import types

from . import commands as _commands


def _is_public(name: str) -> bool:
    return not (name.startswith("__") and name.endswith("__"))


def _export_from_commands() -> None:
    for name in dir(_commands):
        if _is_public(name):
            globals()[name] = getattr(_commands, name)


_export_from_commands()

__all__ = [
    name for name in globals()
    if _is_public(name) and name not in {"_commands", "_is_public", "_export_from_commands"}
]


class _CLIProxy(types.ModuleType):
    """Proxy module that keeps commands and cli exports in sync."""

    def __getattr__(self, name: str):  # type: ignore[override]
        if _is_public(name):
            return getattr(_commands, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:  # type: ignore[override]
        super().__setattr__(name, value)
        if _is_public(name):
            setattr(_commands, name, value)

    def __dir__(self):  # type: ignore[override]
        return sorted({*super().__dir__(), *dir(_commands)})


_proxy = _CLIProxy(__name__)
_proxy.__dict__.update(globals())
sys.modules[__name__] = _proxy

# Clean up helper symbols from the proxy
for _name in ["_proxy", "_is_public", "_export_from_commands"]:
    delattr(sys.modules[__name__], _name)
