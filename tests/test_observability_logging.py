"""Tests for :mod:`jig.observability.logging` and library-level breadcrumbs.

These cover the bootstrap helper (handler install, idempotency, env-var
level control, fallback for bogus levels) and confirm that library
modules emit DEBUG breadcrumbs through their named loggers so consumers
can opt in via ``configure_logging(level="DEBUG")``.
"""
from __future__ import annotations

import logging

import pytest

from jig.observability.logging import configure_logging


@pytest.fixture
def reset_root_logger():
    """Restore root handlers/level after each test so cases don't leak."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)


def test_configure_logging_installs_named_handler(reset_root_logger):
    configure_logging(level=logging.DEBUG)
    root = logging.getLogger()
    handlers = [h for h in root.handlers if h.get_name() == "jig.observability"]
    assert len(handlers) == 1
    assert handlers[0].level == logging.DEBUG
    assert root.level == logging.DEBUG


def test_configure_logging_is_idempotent(reset_root_logger):
    configure_logging(level=logging.INFO)
    configure_logging(level=logging.DEBUG)
    root = logging.getLogger()
    handlers = [h for h in root.handlers if h.get_name() == "jig.observability"]
    # Re-running should adjust in place, never stack duplicates.
    assert len(handlers) == 1
    assert handlers[0].level == logging.DEBUG
    assert root.level == logging.DEBUG


def test_env_var_controls_level(monkeypatch, reset_root_logger):
    monkeypatch.setenv("JIG_LOG_LEVEL", "WARNING")
    configure_logging()
    root = logging.getLogger()
    assert root.level == logging.WARNING


def test_custom_env_var(monkeypatch, reset_root_logger):
    monkeypatch.setenv("GECKO_LOG_LEVEL", "DEBUG")
    configure_logging(env_var="GECKO_LOG_LEVEL")
    root = logging.getLogger()
    assert root.level == logging.DEBUG


def test_explicit_level_overrides_env(monkeypatch, reset_root_logger):
    monkeypatch.setenv("JIG_LOG_LEVEL", "DEBUG")
    configure_logging(level="ERROR")
    root = logging.getLogger()
    assert root.level == logging.ERROR


def test_bogus_level_falls_back_to_info(monkeypatch, reset_root_logger):
    monkeypatch.setenv("JIG_LOG_LEVEL", "NOT-A-LEVEL")
    configure_logging()
    root = logging.getLogger()
    assert root.level == logging.INFO


def test_library_modules_use_named_loggers():
    """Every module we instrumented should expose ``logger`` at module
    scope, named to its dotted path, so consumers can filter by
    ``jig.sweep``, ``jig.llm.openai`` etc. without ``LogRecord`` archaeology.
    """
    expected = [
        "jig.sweep",
        "jig.budget",
        "jig.core.pipeline",
        "jig.core.runner",
        "jig.llm.openai",
        "jig.llm.openrouter",
        "jig.llm.anthropic",
        "jig.llm.ollama",
        "jig.llm.factory",
        "jig.tools.registry",
        "jig.tools.past_results",
        "jig.tracing.sqlite",
    ]
    for name in expected:
        # Importing through getLogger and walking the existing loggers
        # would only confirm that *some* logger of this name exists.
        # Resolving the module attribute instead asserts the *module*
        # set its own logger — i.e. that future refactors don't drop
        # the breadcrumb path silently.
        module = __import__(name, fromlist=["logger"])
        assert hasattr(module, "logger"), f"{name} is missing module-level logger"
        assert module.logger.name == name, (
            f"{name}.logger has wrong name {module.logger.name!r}"
        )
