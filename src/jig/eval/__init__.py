"""Optional eval helpers — file I/O for EvalCase, dataset utilities.

This submodule is import-stable but its contents are intentionally
thin. For richer eval tooling, consume jig outputs (traces, scores,
cases) with promptfoo, Inspect, Braintrust, etc.
"""
from jig.eval.dataset import (
    load_jsonl,
    load_promptfoo_yaml,
    write_jsonl,
)

__all__ = [
    "load_jsonl",
    "load_promptfoo_yaml",
    "write_jsonl",
]
