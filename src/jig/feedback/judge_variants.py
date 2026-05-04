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

    Position bias mitigation is per-call randomization. Over many
    calls the position averages out; for a single ``grade()`` the
    flip can change the result, so seed the judge for reproducible
    runs.
    """

    def __init__(
        self,
        llm: LLMClient,
        criteria: list[str] | None = None,
        rubric: str = "",
        seed: int | None = None,
    ):
        self._llm = llm
        self._criteria = criteria or ["overall quality"]
        self._rubric = rubric
        self._rng = random.Random(seed)

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

        # Position randomization
        if self._rng.random() < 0.5:
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
            data = json.loads(response.content)
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
        elif (winner == "A" and a_is_self) or (winner == "B" and not a_is_self):
            value = 1.0
        else:
            value = 0.0
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

        # Aggregate by dimension
        by_dim: dict[str, list[float]] = {}
        for scores in per_judge:
            for s in scores:
                by_dim.setdefault(s.dimension, []).append(s.value)

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
