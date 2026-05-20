"""Process-level logging bootstrap for jig consumers.

Library code in jig never calls :func:`logging.basicConfig` — that would
silently overwrite an application's own handler config the first time
the import chain ran. Instead, library modules do:

.. code-block:: python

    import logging
    logger = logging.getLogger(__name__)

and applications call :func:`configure_logging` once at startup. This
helper exists so consumers (the gecko sweep CLI, refiner scripts, the
researcher entrypoint) all get the same format and the same environment
variable for runtime level control.

The level resolution order is:

1. Explicit ``level=`` kwarg (caller knows best).
2. ``$JIG_LOG_LEVEL`` (or whatever ``env_var`` the caller passes — gecko
   uses ``GECKO_LOG_LEVEL`` so the two log domains can be tuned
   independently).
3. ``logging.INFO`` as the default.

The format defaults to ``"%(asctime)s %(levelname)s [%(name)s] %(message)s"``,
which matches the existing ``basicConfig`` calls scattered across gecko
scripts — so migrating to ``configure_logging`` is a no-op visually.
"""
from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(
    *,
    level: int | str | None = None,
    format: str | None = None,  # noqa: A002
    env_var: str = "JIG_LOG_LEVEL",
    stream: object | None = None,
) -> None:
    """Install a single ``StreamHandler`` on the root logger.

    Idempotent: re-running with a different level adjusts the existing
    handler in place rather than stacking duplicates. This matters for
    test suites and for apps that call ``configure_logging`` from both a
    library entrypoint and an outer CLI.

    Parameters
    ----------
    level:
        Explicit log level. Wins over ``env_var``. Accepts the same
        forms as :func:`logging.Logger.setLevel`.
    format:
        Format string for the handler. Defaults to a name-tagged format
        matching the legacy ``basicConfig`` setup.
    env_var:
        Environment variable consulted when ``level`` is ``None``.
        Gecko sets this to ``GECKO_LOG_LEVEL`` so jig's own debug noise
        stays togglable independently via ``JIG_LOG_LEVEL``.
    stream:
        Optional stream override for the handler (mostly for tests).
    """
    resolved = _resolve_level(level, env_var)
    fmt = format or _DEFAULT_FORMAT
    root = logging.getLogger()
    root.setLevel(resolved)

    handler = _find_jig_handler(root)
    if handler is None:
        handler = logging.StreamHandler(stream) if stream is not None else logging.StreamHandler()
        handler.set_name("jig.observability")
        root.addHandler(handler)
    handler.setLevel(resolved)
    handler.setFormatter(logging.Formatter(fmt))


def _resolve_level(level: int | str | None, env_var: str) -> int:
    if level is not None:
        return _coerce_level(level)
    env_value = os.environ.get(env_var)
    if env_value:
        return _coerce_level(env_value)
    return logging.INFO


def _coerce_level(value: int | str) -> int:
    if isinstance(value, int):
        return value
    name = value.strip().upper()
    # ``logging.getLevelName`` returns the int for known names and the
    # string back for unknowns — we want a hard fallback to INFO on
    # bogus values rather than a misleading "Level <name>" sentinel.
    resolved = logging.getLevelName(name)
    if isinstance(resolved, int):
        return resolved
    return logging.INFO


def _find_jig_handler(root: logging.Logger) -> logging.Handler | None:
    for handler in root.handlers:
        if handler.get_name() == "jig.observability":
            return handler
    return None


__all__ = ["configure_logging"]
