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

import jig

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
