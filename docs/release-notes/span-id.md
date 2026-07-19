# Identity-aware trace diff

`trace_diff` can now align tool calls by tool-author-declared entity identity
instead of relying exclusively on ordinal position.

See [Replay and compare recorded traces](../../jig-usage-guide.md#replay-and-compare-recorded-traces)
for a complete declaration and comparison example.

## Public API additions

- `ToolDefinition.identity_fields` accepts an ordered list of dot-paths into
  tool-call arguments. The default is `None`, which leaves the tool undeclared.
- `identity_map(registry.list())`, exported from both `jig` and `jig.replay`,
  builds the mapping accepted by `trace_diff`.
- `trace_diff(..., identity_fields=...)` enables identity matching, patience
  anchors for the identity-less remainder, and segment-local ordinal fallback
  whenever the supplied mapping is non-empty.

## Compatibility and serialization

Passing `identity_fields=None` or `{}` preserves legacy ordinal pairing,
divergence classifications and order, event payloads, score/cost/latency
rollups, and `identical`/`fully_identical` behavior.

The serialized public `ToolDiff` schema adds three fields:

- `tier`: `"identity"`, `"anchor"`, or `"ordinal"`, describing alignment
  provenance and assertion strength.
- `index_a`: the source position in trace A's filtered tool-event list.
- `index_b`: the source position in trace B's filtered tool-event list.

Identity matching is intentionally order-insensitive. Therefore,
`identical` and `fully_identical` describe equality under the selected
alignment semantics and do not imply exact tool-call sequence equality.
Use `TrajectoryGrader` for ordering assertions.

`identity_fields` remains internal jig metadata: provider function/tool
schemas do not include it. Replay configuration snapshots are also unchanged
because they serialize configuration state, not tool definitions.
