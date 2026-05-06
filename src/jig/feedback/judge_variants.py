"""Calibrated LLM-judge variants — pairwise + committee.

`LLMJudge` is single-judge, fixed prompt, no bias controls.
Documented failure modes:

- **Position bias** in pairwise judging — first answer wins
  disproportionately.
- **Length bias** — longer answers score higher independent of
  quality.
- **Self-preference** — judges favor outputs from their own model
  family.
- **Calibration drift** — judge scores don't track human or
  ground-truth agreement.

Two variants in this module address position bias and judge
disagreement; calibration drift is measured by
``jig.eval.calibration.calibrate_judge`` (separate workflow).
"""
from __future__ import annotations

import asyncio
import json
import random
import warnings
from typing import Any

from jig.core.types import (
    CompletionParams,
    Grader,
    LLMClient,
    Message,
    Role,
    Score,
    ScoreSource,
)
from jig.feedback.parsing import strip_markdown_fence


_PAIRWISE_PROMPT = """You are an evaluation judge. You will be shown an
input and two candidate outputs labeled A and B. Decide which output
is better against the criteria listed.

Return ONLY valid JSON: {{"winner": "A" | "B" | "tie", "reasoning": "<brief>"}}

Criteria: {criteria}
{rubric}"""


class PairwiseLLMJudge(Grader):
    """Pairwise judge with position randomization.

    Reads ``context["compare_to"]`` for the alternate output.
    Internally randomizes which output is shown first to the judge,
    then maps the judge's verdict back to a self-relative score:
    ``1.0`` for win, ``0.0`` for loss, ``0.5`` for tie. The returned
    ``Score.dimension`` is ``"vs_<other_id>"`` so multiple pairwise
    comparisons in the same grading context don't collide.

    ``context["compare_to"]`` should be a mapping with keys
    ``output`` (str) and ``id`` (str — the dimension suffix).

    ``criteria`` shapes the prompt — the judge is asked to weigh A vs
    B against this list — but the returned Score is always a single
    holistic winner judgment. This is intentional: pairwise judging
    is "which is better overall, given these things matter," not
    "score each dimension independently." Use ``LLMJudge`` (not
    pairwise) when you want per-dimension absolute scores.

    Position bias mitigation is per-call randomization. When
    ``seed`` is provided, the flip is a deterministic function of
    ``(seed, other_id, input)`` — so two calls with the same inputs
    always produce the same A/B assignment regardless of dispatch
    order. This matters in a sweep where one judge instance is
    shared across concurrent ``run_agent`` calls; a shared mutable
    RNG would interleave under ``asyncio.gather`` and break
    reproducibility. Without a seed, the flip uses ``random`` (the
    module's process-global RNG) — randomization without
    reproducibility.
    """

    def __init__(
        self,
        llm: LLMClient,
        criteria: list[str] | None = None,
        rubric: str = "",
        seed: int | None = None,
    ):
        self._llm = llm
        # ``None`` means "unspecified — use the default." An explicit
        # empty list means "render no criteria" (caller wants the
        # judge to weigh outputs free-form). The two are distinct,
        # so don't collapse them with truthiness.
        self._criteria = (
            ["overall quality"] if criteria is None else criteria
        )
        self._rubric = rubric
        self._seed = seed

    def _flip_self_to_a(self, other_id: str, input: Any) -> bool:
        """Return True iff self's output should appear in position A.

        Deterministic when ``self._seed`` is set: a fresh
        ``random.Random`` seeded with ``(seed, other_id, input)``
        produces the same draw regardless of call order. With
        ``seed=None``, falls back to the process-random module which
        gives randomization but no cross-run reproducibility.
        """
        if self._seed is None:
            return random.random() < 0.5
        # ``random.Random`` accepts None/int/float/str/bytes — not
        # tuples — so fold the components into a single string seed.
        # The pipe separator avoids ambiguity from values that happen
        # to contain digits at component boundaries.
        local_rng = random.Random(f"{self._seed}|{other_id}|{input}")
        return local_rng.random() < 0.5

    async def grade(
        self,
        input: Any,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        if not context or "compare_to" not in context:
            return []
        compare = context["compare_to"]
        try:
            other_output = compare["output"]
            other_id = compare["id"]
        except (KeyError, TypeError):
            return []

        # Position randomization — stable under concurrency when seeded.
        if self._flip_self_to_a(other_id, input):
            a_text, b_text = output, other_output
            a_is_self = True
        else:
            a_text, b_text = other_output, output
            a_is_self = False

        system = _PAIRWISE_PROMPT.format(
            criteria=", ".join(self._criteria),
            rubric=self._rubric,
        )
        user_msg = (
            f"**Input:** {input}\n\n"
            f"**Output A:** {a_text}\n\n"
            f"**Output B:** {b_text}"
        )
        params = CompletionParams(
            messages=[Message(role=Role.USER, content=user_msg)],
            system=system,
            temperature=0.0,
        )
        response = await self._llm.complete(params)
        try:
            data = json.loads(strip_markdown_fence(response.content))
            winner = data["winner"]
        except (json.JSONDecodeError, KeyError, TypeError):
            # Malformed judge response — record as a tie rather than
            # raise so a single bad call doesn't crash a sweep.
            return [
                Score(
                    dimension=f"vs_{other_id}",
                    value=0.5,
                    source=ScoreSource.LLM_JUDGE,
                )
            ]

        if winner == "tie":
            value = 0.5
        elif winner == "A":
            value = 1.0 if a_is_self else 0.0
        elif winner == "B":
            value = 0.0 if a_is_self else 1.0
        else:
            # Valid JSON but ``winner`` isn't one of A/B/tie — treat as
            # a tie like the malformed-JSON path above. A meaningless
            # verdict is "no opinion," not a loss.
            value = 0.5
        return [
            Score(
                dimension=f"vs_{other_id}",
                value=value,
                source=ScoreSource.LLM_JUDGE,
            )
        ]


class CommitteeJudge(Grader):
    """N judges vote; score is mean across judges; agreement reported
    as a separate dimension.

    Each underlying grader runs in parallel. The final score for
    each dimension is the mean across judges. An additional Score
    with ``dimension=f"{dim}_agreement"`` reports a normalized
    inverse-stddev signal in ``[0, 1]`` (1 = unanimous, 0 = maximally
    split). The transform is ``max(0, 1 - 2 * stddev)``: stddev of
    ``[0,1]`` values is bounded by 0.5, so this maps to ``[0, 1]``
    linearly. This is a heuristic confidence signal, not a formal
    inter-rater agreement metric — for that, run the committee
    against a labeled holdout and use Fleiss' kappa offline.

    **Dimension set mismatch.** If different judges return different
    dimensions (one returns extra dims, another drops dims), the
    aggregator emits a ``RuntimeWarning`` naming the missing dims
    and averages over only the judges that reported each dimension.
    Mismatches usually indicate a misconfigured judge prompt;
    surface rather than silently down-weight.
    """

    def __init__(self, judges: list[Grader]):
        if not judges:
            raise ValueError("CommitteeJudge requires at least one judge")
        self._judges = judges

    async def grade(
        self,
        input: Any,
        output: Any,
        context: dict[str, Any] | None = None,
    ) -> list[Score]:
        per_judge = await asyncio.gather(
            *[j.grade(input, output, context) for j in self._judges]
        )

        # Detect dimension-set mismatches across judges
        dim_sets = [{s.dimension for s in scores} for scores in per_judge]
        if dim_sets and not all(d == dim_sets[0] for d in dim_sets):
            union = set().union(*dim_sets)
            missing_per_judge = [union - d for d in dim_sets]
            warnings.warn(
                f"CommitteeJudge: judges returned different dimension "
                f"sets (per-judge missing: {missing_per_judge}). "
                f"Averaging over judges that reported each dimension.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Aggregate by dimension. First collapse each judge's
        # within-call duplicates (same dimension reported twice by
        # one judge — possible via CompositeGrader or a misconfigured
        # judge) to a per-judge mean, THEN average across judges.
        # Without this collapse, one judge with two ``q`` scores
        # would silently outweigh peers reporting ``q`` once.
        by_dim: dict[str, list[float]] = {}
        for scores in per_judge:
            per_judge_dim: dict[str, list[float]] = {}
            for s in scores:
                per_judge_dim.setdefault(s.dimension, []).append(s.value)
            for dim, vals in per_judge_dim.items():
                by_dim.setdefault(dim, []).append(sum(vals) / len(vals))

        out: list[Score] = []
        for dim, values in by_dim.items():
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            stddev = variance ** 0.5
            out.append(
                Score(
                    dimension=dim,
                    value=mean,
                    source=ScoreSource.LLM_JUDGE,
                )
            )
            out.append(
                Score(
                    dimension=f"{dim}_agreement",
                    value=max(0.0, 1.0 - 2.0 * stddev),
                    source=ScoreSource.LLM_JUDGE,
                )
            )
        return out
