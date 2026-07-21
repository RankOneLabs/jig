# Human-reviewed feedback prompt injection

`AgentConfig.human_feedback_prompt` is an opt-in policy for surfacing
task-similar, *human-graded* exemplars as labeled few-shot examples in the
agent's system prompt. It is independent of the older
`include_feedback_in_prompt` / `feedback.get_signals` path documented in
[`sqlite-feedback-maintenance.md`](sqlite-feedback-maintenance.md#effective-scores-opt-in):
that path can inject any score source above a single `min_score` threshold
and carries no safety gate. This path only ever surfaces `source="human"`
effective scores, split into clearly labeled positive/negative sections,
gated by a caller-supplied eligibility filter before a result can enter
either section. Both paths can be enabled at once — they append to
different parts of the system message.

Disabled by default: `AgentConfig().human_feedback_prompt.enabled` is
`False`, so existing agents see no behavior change until they opt in.

## Config fields

```python
from jig import EffectiveScoreFilter, HumanFeedbackPromptConfig

human_feedback_prompt = HumanFeedbackPromptConfig(
    enabled=True,
    dimensions=("plausibility", "lookahead_safety"),
    positive_threshold=0.75,
    negative_threshold=0.25,
    positive_limit=2,
    negative_limit=2,
    total_character_budget=6000,
    eligibility_filters=(
        EffectiveScoreFilter(dimension="plausibility", min_value=0.5),
    ),
)
```

- `enabled` — must be explicitly set `True`; `dimensions` must be non-empty
  when it is.
- `dimensions` — which rubric axes are considered for classification. Not
  validated against any fixed taxonomy — jig is domain-agnostic; the
  caller's rubric owns that.
- `positive_threshold` / `negative_threshold` — inclusive bounds in
  `[0.0, 1.0]`. An effective *human* score `>= positive_threshold` on a
  selected dimension qualifies it as a positive signal; `<=
  negative_threshold` qualifies it as negative. Values strictly between are
  never injected. `positive_threshold` must be `>= negative_threshold`.
- `positive_limit` / `negative_limit` — per-section cap on the number of
  rendered examples.
- `total_character_budget` — combined character budget for both sections'
  rendered example bodies (see below).
- `eligibility_filters` — a tuple of `EffectiveScoreFilter`, AND-combined,
  applied to every candidate's effective scores *before* classification. A
  result that fails this can never enter either section, regardless of how
  it would otherwise classify — a poor result cannot be repurposed as a
  "negative" exemplar to bypass a safety gate such as a plausibility floor.

## Retrieval and classification

`FeedbackLoop.get_human_examples(task_input, config)` — a concrete method
with a no-op default (returns an empty set) so backends that can't rank by
embedding similarity (e.g. `NullFeedbackLoop`) need no override.
`SQLiteFeedbackLoop` overrides it with real retrieval:

1. Rank stored results by embedding similarity to `task_input`. Results
   with no stored embedding are skipped — never substituted with a
   recency fallback.
2. Apply `eligibility_filters` on each candidate's effective scores.
3. Classify the survivors by looking only at `source="human"` effective
   scores for `config.dimensions` (a dimension resolved to a heuristic row
   — no human grade yet — never qualifies a result, even if the value
   itself would cross a threshold). Any selected dimension `<=
   negative_threshold` puts the whole result in the negative section
   (checked first — negative precedence); only when nothing crosses
   negative does a `>= positive_threshold` dimension qualify it as
   positive. A result crossing neither threshold is omitted from both
   sections. Each result appears in at most one section.
4. Ties in similarity break on the newest qualifying human score
   (`created_at`), then on `result_id`, both deterministic.
5. Each section is capped at its configured limit.

## Prompt rendering

`build_human_feedback_section` (in `jig.core.prompt`) renders the result as
two headings:

```
## Human-reviewed positive examples: imitate the cited strengths
## Human-reviewed negative examples: avoid the cited failures
```

Each example wraps its stored input and output in untrusted-data
delimiters (`<UNTRUSTED_EXAMPLE_INPUT>` / `<UNTRUSTED_EXAMPLE_OUTPUT>`) so
historical content can't be read as a new system instruction, followed by
a line naming every threshold-crossing dimension, its value, and the
human's note (from the score's `metadata["note"]`).

A section is omitted entirely — no heading — when it has no qualified
examples; when *both* are empty, the whole block is an empty string (no
placeholder text). Absence of evidence is preferable to noisy prompt text.

`total_character_budget` is allocated **round-robin** across every
selected example (both sections combined), so the first example can never
consume the entire budget in one step. An example left short of its full
length gets an explicit truncation marker appended.

## Where it runs

`run_agent` queries `feedback.get_human_examples` right after the legacy
`feedback.get_signals` call, only when `config.human_feedback_prompt.enabled`
is set, and appends the rendered section to the system message assembled by
`build_system_message`. The query is recorded as its own
`SpanKind.MEMORY_QUERY` span (`query_human_feedback`) with the count of
positive/negative examples returned.
