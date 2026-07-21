# SQLite feedback maintenance

`SQLiteFeedbackLoop` now enables SQLite foreign-key enforcement on every
connection and validates score payloads before writing them. New score rows must
reference a real feedback result created by `store_result()`, and invalid values
such as `NaN`, infinities, empty dimensions, or values outside `0.0` through
`1.0` are rejected instead of being normalized.

Older feedback databases may contain orphan score rows from versions that did
not enforce foreign keys. Check a long-lived database before using it with newer
code:

```sql
SELECT s.result_id
FROM scores AS s
LEFT JOIN results AS r ON r.id = s.result_id
WHERE r.id IS NULL
GROUP BY s.result_id;
```

If this query returns rows, those scores were never attached to queryable
feedback results. They cannot be returned by `query()`, `get_signals()`, or
`export_eval_set()`. Remove them before continuing with strict foreign-key
enforcement:

```sql
DELETE FROM scores
WHERE result_id NOT IN (
  SELECT id FROM results
);
```

Run `PRAGMA foreign_key_check;` after cleanup. It should return no rows.

Back up production feedback databases before deleting rows. If orphan scores
carry information you need to preserve, export them first and decide whether to
re-ingest them by creating explicit feedback results with the right content,
input, and metadata.

## Score metadata column

`scores` now has a nullable `metadata` JSON column for per-dimension causal
detail (e.g. an offending claim, a missing piece of context). It's added
automatically — `SQLiteFeedbackLoop` checks for it via `PRAGMA table_info`
on first use and runs `ALTER TABLE scores ADD COLUMN metadata JSON` once if
absent. No manual migration step is required, and every historical score
row is left with `metadata = NULL`, which `Score.metadata` reads back as
`None` (distinct from an explicitly-stored empty object).

## Timestamps

New `results.created_at` and `scores.created_at` values are written as
aware UTC (`datetime.now(UTC).isoformat()`, e.g. `...+00:00`). Rows written
by older versions have naive timestamps with no offset; these are not
rewritten, but `query()` and any span reads interpret them as UTC when
constructing `datetime` objects — a database is never read back as a mix of
naive and aware values from the same call. `export_eval_set(since=...)`
requires an aware `datetime` and raises `ValueError` on a naive one.

## Effective scores (opt-in)

The `scores` table is append-only: every `score()` call adds new rows, it
never updates or deletes existing ones. By default `query()` and
`get_signals()` keep returning the complete score history per result, in
`ScoredResult.scores`, exactly as before — nothing about existing callers
changes.

Consumers that need a single current judgment per dimension instead of the
full history can opt in per-`FeedbackQuery`:

- `resolve_effective=True` populates `ScoredResult.effective_scores`, a
  `dict[str, EffectiveScore]` keyed by dimension.
- `effective_filters=[EffectiveScoreFilter(dimension=..., min_value=...,
  max_value=...)]` adds an inclusive, AND-combined gate on the *effective*
  score for one or more dimensions, applied before the query's similarity
  ranking is truncated to `limit`. Setting `effective_filters` also
  populates `effective_scores` on the returned rows, even without
  separately setting `resolve_effective`.

Resolution rule, per `(result_id, dimension)`:

1. If any `source="human"` row exists for that dimension, the newest one
   wins (`created_at` descending, then SQLite `rowid` descending as a
   deterministic tie-breaker for same-timestamp rows).
2. Otherwise, the newest `source="heuristic"` row wins under the same
   ordering.
3. Otherwise, the dimension has no effective score — `effective_filters`
   treats a missing dimension as failing that filter, never as passing.

Only the exact source strings `human` and `heuristic` participate;
`llm_judge`, `ground_truth`, and any other source remain visible in the
full history but can never become an effective judgment. A human row
always outranks every heuristic row for its dimension, even a heuristic
row appended later — later automated grading can never silently overrule
a human judgment. Appending a *newer* human row is the only way to change
an effective score that already has a human row.

No score rows are ever mutated to compute an effective score; the full,
unfiltered history is always available via `ScoredResult.scores`.
