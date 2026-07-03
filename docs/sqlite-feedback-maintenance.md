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
