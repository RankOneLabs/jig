"""Observability helpers — logging bootstrap, log-format conventions.

Library modules in jig only ever do ``logger = logging.getLogger(__name__)``.
Applications opt into formatting and level control by calling
:func:`jig.observability.logging.configure_logging` at process startup.
"""
from __future__ import annotations

from jig.observability.logging import configure_logging

__all__ = ["configure_logging"]
