"""Contract test for jig's stable top-level public import surface.

Consumers should import everything they need from the top-level ``jig``
package rather than reaching into internal module layout (jig.core.types,
jig.feedback.loop, jig.tracing.sqlite, jig.tools.registry, jig.regression,
jig.sweep internals, ...). This test locks down every name a consumer is
expected to be able to import as ``from jig import <name>``, so an
accidental removal or module-layout refactor breaks loudly here instead of
silently in a downstream project.
"""
from __future__ import annotations

import importlib

import jig

# ``jig.replay`` the submodule is shadowed on the ``jig`` package object by
# the ``replay()`` function of the same name (both are exported from
# ``jig/__init__.py``), so submodules under ``jig.replay`` must be reached
# via ``importlib.import_module`` rather than dotted attribute access.
align = importlib.import_module("jig.replay.align")

REQUIRED_PUBLIC_NAMES = [
    # Core config / result types
    "AgentConfig",
    "AgentResult",
    "SUBMIT_OUTPUT_TOOL",
    # Feedback protocol + data types
    "EvalCase",
    "FeedbackLoop",
    "FeedbackQuery",
    "Grader",
    "Score",
    "ScoreSource",
    "ScoredResult",
    # Feedback loop implementations
    "NullFeedbackLoop",
    "SQLiteFeedbackLoop",
    # Pipeline config/result types
    "MapResult",
    "PipelineConfig",
    "PipelineResult",
    "Step",
    # Tracer protocol + implementations
    "TracingLogger",
    "SQLiteTracer",
    "StdoutTracer",
    "FederatedTracer",
    "RollupClient",
    "RollupUnreachableError",
    # Tools
    "ToolRegistry",
    # Regression + sweep
    "CompareResult",
    "CompareRun",
    "PassAtK",
    "RegressionAlert",
    "RegressionReport",
    "SweepResult",
    "SweepRun",
    "WinRate",
    "compare",
    "detect_regressions",
    "pass_at_k",
    "sweep",
    "win_rate",
    # Entry points
    "run_agent",
    "run_pipeline",
    "map_pipeline",
]


def test_every_required_public_name_is_importable_from_top_level():
    missing = [name for name in REQUIRED_PUBLIC_NAMES if not hasattr(jig, name)]
    assert missing == [], f"missing from top-level jig package: {missing}"


def test_every_required_public_name_is_in_dunder_all():
    missing = [name for name in REQUIRED_PUBLIC_NAMES if name not in jig.__all__]
    assert missing == [], f"missing from jig.__all__: {missing}"


def test_from_jig_import_star_style_access_works():
    for name in REQUIRED_PUBLIC_NAMES:
        assert getattr(jig, name) is not None


# --- identity_map convenience: supported import paths ---


def test_identity_map_importable_from_jig_and_jig_replay():
    replay_pkg = importlib.import_module("jig.replay")
    assert jig.identity_map is align.identity_map
    assert replay_pkg.identity_map is align.identity_map


def test_identity_map_in_both_dunder_all():
    replay_pkg = importlib.import_module("jig.replay")
    assert "identity_map" in jig.__all__
    assert "identity_map" in replay_pkg.__all__


# --- ToolEvent: existing public paths remain intact ---


def test_tool_event_importable_from_all_existing_paths():
    replay_pkg = importlib.import_module("jig.replay")
    diff_mod = importlib.import_module("jig.replay.diff")
    assert jig.ToolEvent is replay_pkg.ToolEvent is diff_mod.ToolEvent


# --- Alignment internals: deliberately not part of the public surface ---

_INTERNAL_ALIGNMENT_NAMES = [
    "AlignmentTier",
    "AlignedPair",
    "UnmatchedEvent",
    "Alignment",
    "Aligner",
    "OrdinalAligner",
    "IdentityAligner",
    "resolve_identity",
]


def test_alignment_internals_absent_from_jig_dunder_all():
    present = [name for name in _INTERNAL_ALIGNMENT_NAMES if name in jig.__all__]
    assert present == []


def test_alignment_internals_absent_from_jig_replay_dunder_all():
    replay_pkg = importlib.import_module("jig.replay")
    present = [name for name in _INTERNAL_ALIGNMENT_NAMES if name in replay_pkg.__all__]
    assert present == []


def test_alignment_internals_still_importable_from_align_module():
    missing = [name for name in _INTERNAL_ALIGNMENT_NAMES if not hasattr(align, name)]
    assert missing == []
