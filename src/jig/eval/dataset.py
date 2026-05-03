"""File I/O for ``EvalCase`` records.

JSONL is the canonical on-disk format: one JSON object per line,
``input`` required (str), ``expected`` optional (str), ``context``
and ``metadata`` optional (dict). The promptfoo-yaml reader is a
thin compat layer for borrowing test sets — it loses fidelity in
both directions, so for round-trip use jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jig.core.types import EvalCase


def load_jsonl(path: str | Path) -> list[EvalCase]:
    """Load ``EvalCase`` records from a jsonl file.

    Each line is a JSON object with keys: ``input`` (required, str),
    ``expected`` (optional, str), ``context`` (optional, dict),
    ``metadata`` (optional, dict). Blank lines and lines starting with
    ``#`` are skipped.

    Validation is strict: malformed JSON, missing ``input``, or a
    non-string ``input``/``expected`` raises ``ValueError`` with the
    file path and line number, so corrupt records fail loudly at load
    time instead of producing scores against an invalid ``EvalCase``.
    """
    cases: list[EvalCase] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{p}:{lineno}: invalid JSON — {exc.msg}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{p}:{lineno}: expected a JSON object, got "
                    f"{type(obj).__name__}"
                )
            if "input" not in obj:
                raise ValueError(
                    f"{p}:{lineno}: missing required 'input' field"
                )
            if not isinstance(obj["input"], str):
                raise ValueError(
                    f"{p}:{lineno}: 'input' must be a string, got "
                    f"{type(obj['input']).__name__}"
                )
            expected = obj.get("expected")
            if expected is not None and not isinstance(expected, str):
                raise ValueError(
                    f"{p}:{lineno}: 'expected' must be a string or null, "
                    f"got {type(expected).__name__}"
                )
            ctx = obj.get("context")
            if ctx is not None and not isinstance(ctx, dict):
                raise ValueError(
                    f"{p}:{lineno}: 'context' must be a dict or null, "
                    f"got {type(ctx).__name__}"
                )
            meta = obj.get("metadata")
            if meta is not None and not isinstance(meta, dict):
                raise ValueError(
                    f"{p}:{lineno}: 'metadata' must be a dict or null, "
                    f"got {type(meta).__name__}"
                )
            cases.append(
                EvalCase(
                    input=obj["input"],
                    expected=expected,
                    context=ctx,
                    metadata=meta,
                )
            )
    return cases


def write_jsonl(cases: list[EvalCase], path: str | Path) -> None:
    """Write ``EvalCase`` records to a jsonl file. Overwrites existing.

    Optional fields (``expected``, ``context``, ``metadata``) are
    omitted from the line when ``None`` — keeps the on-disk format
    stable for tools that care about field presence.
    """
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for c in cases:
            obj: dict[str, Any] = {"input": c.input}
            if c.expected is not None:
                obj["expected"] = c.expected
            if c.context is not None:
                obj["context"] = c.context
            if c.metadata is not None:
                obj["metadata"] = c.metadata
            f.write(json.dumps(obj) + "\n")


def load_promptfoo_yaml(path: str | Path) -> list[EvalCase]:
    """Load ``EvalCase`` records from a promptfoo-compatible yaml file.

    Reads the ``tests`` array. Maps:

    - ``vars.input`` or ``vars.prompt`` → ``EvalCase.input``
    - ``vars.expected`` → ``EvalCase.expected``
    - ``description`` → ``EvalCase.metadata["description"]``
    - ``assert`` array → ``EvalCase.metadata["assertions"]``
    - everything else under ``vars`` → ``EvalCase.context``

    Requires the ``eval`` extra for ``pyyaml``:
        pip install 'jig[eval]'
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "load_promptfoo_yaml requires pyyaml. "
            "Install with: pip install 'jig[eval]'"
        ) from exc
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "tests" not in doc:
        raise ValueError(
            f"{p}: expected a top-level mapping with a 'tests' key"
        )
    tests = doc["tests"]
    if not isinstance(tests, list):
        raise ValueError(f"{p}: 'tests' must be a list")
    cases: list[EvalCase] = []
    for i, t in enumerate(tests):
        if not isinstance(t, dict):
            raise ValueError(f"{p}: tests[{i}] must be a mapping")
        v = t.get("vars") or {}
        if not isinstance(v, dict):
            raise ValueError(f"{p}: tests[{i}].vars must be a mapping")
        # Key-presence check (not truthiness): an explicit empty
        # ``vars.input: ""`` should be preserved, not silently fall
        # through to ``vars.prompt``.
        if "input" in v:
            input_val = v["input"]
        else:
            input_val = v.get("prompt")
        if input_val is None:
            raise ValueError(
                f"{p}: tests[{i}] missing vars.input or vars.prompt"
            )
        if not isinstance(input_val, str):
            raise ValueError(
                f"{p}: tests[{i}] vars.input/prompt must be a string"
            )
        expected = v.get("expected")
        if expected is not None and not isinstance(expected, str):
            raise ValueError(
                f"{p}: tests[{i}] vars.expected must be a string"
            )
        context = {
            k: val
            for k, val in v.items()
            if k not in ("input", "prompt", "expected")
        }
        meta: dict[str, Any] = {}
        if "description" in t:
            meta["description"] = t["description"]
        if "assert" in t:
            meta["assertions"] = t["assert"]
        cases.append(
            EvalCase(
                input=input_val,
                expected=expected,
                context=context or None,
                metadata=meta or None,
            )
        )
    return cases
