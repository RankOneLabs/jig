# Phase 12 plan — ta capstone migration

**Goal.** Ta's researcher agents migrate from their fork of pre-phase-0
jig surfaces to post-phase-11 main. The specialist sweep runs as
nested parallelism (smithers fan-out across workers × concurrent
Optuna trials per backtest) and drops from ~83h to ~8h. Along the way,
five deprecated ta surfaces are deleted and replaced with their
post-redesign equivalents.

## Scope

**In:**

- Ta imports `jig.llm.from_model` everywhere it calls the old
  `agents.researcher.llm.create_llm`. `agents/researcher/llm.py` is
  deleted.
- `specialist_config` / `refiner_config` / `explorer_config` collapse
  into one base `AgentConfig[BacktestOutcome]` plus three
  `.with_(...)` variants. `_build_shared_infra` shrinks.
- Graders become `Grader[BacktestOutcome]` consuming the pydantic
  instance; `_extract_metrics` / `_extract_sharpe` regex helpers go.
- `RunBacktestTool(dispatch=True)` with a `dispatch_fn_ref` pointing at
  `backtester.training.generic_objective:run_signal_study`, registered
  via the `jig.smithers_fn` entry-point group.
- `agents/researcher/sweep.py`'s driver swaps manual for-loops for
  `jig.sweep(cases, [config], concurrency=N)`. Explorer → specialist →
  refiner stays three sweeps in sequence, not one.
- Worker deploy picks up the `backtester` package + its entry point so
  `run_signal_study` resolves on mcbain / frink / mcclure.
- NFS: willie exposes `/srv/ta-repo` and `/srv/ta-data` read-only
  (ro/rw respectively) over Tailscale; workers mount at `/ta-repo` and
  `/ta-data`.
- Benchmark harness: before/after wall-clock + worker utilization for
  an N=50 specialist sweep across 3 strategy types.

**Out (deferred):**

- Scout + algerknown lift (phase 13).
- Auto-persistence of `SweepResult` into `FeedbackLoop`.
- Callback-based sweep fan-out (phase 10).
- Migrating ta's own file-backed `PastResultsTool` to jig's
  `FeedbackLoop`-backed `PastResults` — different data sources.
- `WriteStrategyTool` / `ValidateStrategyTool` dispatch — only
  `RunBacktestTool` is on the hot path.
- Ta's heuristic/LLM-judge `ConditionalCompositeGrader` composition —
  works fine on pydantic input unchanged.

## Prerequisite audit

Concrete file:line references against current working trees (each
repo's main).

### Deprecated ta surfaces

- **`CostTrackingLLM` + `create_llm`** —
  `ta/agents/researcher/llm.py:32, 53, 76`. Sole callers:
  `ta/agents/researcher/config.py:13, 54, 97, 143, 188`. No external
  consumers. Safe to delete in one shot.
- **`_parse_strategy_types`** — `ta/agents/researcher/sweep.py:25`
  (defined), `:117` (called). Replaced by
  `output_schema=ExplorerOutcome(strategy_types: list[StrategyType])`.
- **Three config builders** at `ta/agents/researcher/config.py:68,
  115, 161`. Six call sites:
  `sweep.py:82, 133, 185`, `__main__.py:18, 27, 35`,
  `scripts/refine_liquidity_snap.py:37`,
  `scripts/refine_orderflow_imbalance.py:43`,
  `agents/tests/test_config.py:33, 60, 66`.
- **Regex grader extraction** — `_extract_metrics` at
  `ta/agents/researcher/graders.py:48`, used by
  `StrategyQualityGrader.grade` at `:95`. `_extract_sharpe` at
  `sweep.py:38`, called at `:164, :221`.
- **Tests** touching deleted surfaces —
  `ta/agents/tests/test_graders.py:12, 15, 86, 95, 102, 111, 124,
  136, 157, 181, 201, 207, 213` and
  `ta/agents/tests/test_config.py:21, 27`. Both need rewrites, not
  find-replace — today's `test_graders` asserts JSON-from-text
  extraction which is exactly what's going away.

### Jig surfaces present (no changes needed)

- `AgentConfig.with_(...)` at `src/jig/core/runner.py:125`.
- `jig.llm.from_model` at `src/jig/llm/factory.py:27`. Prefix
  routing covers ta's models (`claude-sonnet-4-*` etc.).
- `Tool.dispatch` + `Tool.dispatch_fn_ref` at
  `src/jig/core/types.py:385-400`. `ToolRegistry` already routes
  them at `src/jig/tools/registry.py:52`.
- `jig.sweep(..., concurrency=...)` at `src/jig/sweep.py:172`.
- `Grader[T]` at `src/jig/core/types.py:329`.

### Jig has no non-ta users of the deleted surfaces

From the jig repo root,
`grep -R -nE 'CostTrackingLLM|create_llm\(|_parse_strategy_types' src/`
returned zero hits. Phase 12's "jig-side deletions" per the roadmap
is a phrasing artifact — the deletions are entirely inside ta.

### Smithers plumbing ready

- `FunctionExecutor` resolves `fn_ref` via
  `entry_points(group="jig.smithers_fn")` at
  `smithers/worker/executors/function.py:40, 69-83`. Discovery at
  startup; logs resolved names.
- `Job` / `JobSubmission` / `TaskRequest` carry `trace_context`
  post-phase-9.
- **Gap:** `smithers/deploy/install.sh` only pip-installs smithers.
  Needs an `--extra-package` hook so worker venvs pick up `backtester`.

### Ta's current parallelism

- `sweep.py:147` runs specialists one-at-a-time in a Python for-loop.
- Inside each backtest, `run_signal_study` at
  `ta/backtester/training/generic_objective.py:143` calls
  `study.optimize(objective, n_trials=N)` **serially**. No `n_jobs`.
- Optuna storage already uses `sqlite:///optuna_studies.db` with
  `load_if_exists=True` (lines 132, 140). Multi-process capable; we
  just haven't used it.
- `ta/backtester/pyproject.toml` has **no jig dep**. Keeping it that
  way matters — workers shouldn't need jig installed to execute
  backtests. The entry point is `run_signal_study` itself, a pure
  numpy/pandas/optuna callable.

## Key design calls

### 1. Delete old ta surfaces in the same PR as the migration

The five deprecated surfaces have no users outside ta. Leaving them
behind "for deprecation" means dead code with no upside. One PR:
delete, migrate callers, rewrite tests. Phase 12's row in the roadmap
fills `Breaking: yes` on the ta side, `no` on the jig side.

### 2. Dispatch boundary is `RunBacktestTool`, not `run_agent`

Dispatching the whole agent would require shipping a serializable
`AgentConfig` — infeasible today (configs hold live LLM client
instances, tracers, etc.). Dispatching only the expensive tool call
keeps the agent loop on the orchestrator (where API keys + feedback
DB + `strategies/generated/` live). Workers see only
`run_signal_study(name, coins, dates, n_jobs)`.

### 3. `output_schema=BacktestOutcome` + `Grader[BacktestOutcome]`

```python
class BacktestOutcome(BaseModel):
    strategy_name: str
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    best_params: dict[str, Any]
    top_5_sharpes: list[float]
    notes: str = ""


class ExplorerOutcome(BaseModel):
    strategy_types: list[str]
    rationales: dict[str, str]
```

Specialist + refiner submit `BacktestOutcome` via `submit_output`.
Explorer submits `ExplorerOutcome`. Graders read
`output.sharpe_ratio` / `.max_drawdown_pct` / `.num_trades` directly;
the scoring helpers (`_sharpe_score` etc.) stay — pure math, input
change is cosmetic.

### 4. Strategy module distribution: NFS-shared `generated/`

`WriteStrategyTool` writes `backtester/strategies/generated/<name>.py`
on the orchestrator. `RunBacktestTool(dispatch=True)` sends only
`name` + params. Worker's `run_signal_study` does
`importlib.import_module(f"backtester.strategies.generated.{name}")`
— needs file visibility.

Chosen: willie NFS-exports `/srv/ta-repo` ro; workers mount at
`/ta-repo` so `backtester.strategies.generated.*` imports resolve.
Alternative (serialize source into payload + worker tempdir) is
rejected as awkward (hot-reload semantics, cleanup).

### 5. Optuna parallelism: per-backtest `n_jobs>1`, not cross-machine

SQLite-on-NFS for concurrent writers is a known trap (lock files
behave poorly). We don't try cross-worker study sharing. Each worker
owns its backtest end-to-end; inside the backtest, `n_jobs=4` gives
thread-level parallelism over Optuna trials. Cross-machine concurrency
happens *above* via `jig.sweep(..., concurrency=3)`.

Optuna studies DB goes on **worker-local** disk
(`/tmp/optuna_studies.db`), not NFS. Summary JSON + strategy outputs
go to NFS (`/ta-data/results/`). Requires a new
`study_storage_path: str | None = None` param on `run_signal_study`.

### 6. 10× speedup decomposition

- 50 specialists × ~100 trials at ~6s each ≈ 500min/type serial.
- 3 strategy types sequential ≈ 25h of specialist work. Explorer +
  refiner + overhead → ~83h end-to-end (matches roadmap).
- After: 3 workers × 50 specialists / concurrency=3 ≈ 17 specialists
  per worker, each with `n_jobs=4` → backtest wall-clock ~2.5min.
  17 × 2.5min per worker = ~43min per type, 3 types ≈ 2h15min
  specialist × 3, plus explorer/refiner → ~8h.
- **10× = 3× worker fan-out × 3× in-process `n_jobs`.** Either axis
  alone would give ~3×.

### 7. Port allocations

No new listeners. Phase 12 rides phases 7–9 infra: willie:8900
(dispatch), willie:8901 (rollup), worker ports per
`routing_config.yaml`. New infra is filesystem-level (NFS exports),
not network services.

### 8. Benchmark protocol

Same input set, same models, before = current main (pre-dispatch
sweep), after = migrated. Ship only if after ≤ 15h median wall-clock
and best-sharpe distribution doesn't regress at p=0.05. Full protocol
in the benchmark section.

## File-by-file plan, in implementation order

### Step 1 — ta schemas

New `ta/agents/researcher/schemas.py`: `BacktestOutcome`,
`ExplorerOutcome`.

**Tests:** `agents/tests/test_schemas.py` — round-trip, defaults,
JSON schema shape. No external deps. Ships alone as a prep commit.

### Step 2 — `from_model` swap, delete `llm.py`

`ta/agents/researcher/config.py`: replace `create_llm(...)` with
`from_model(...)` at five sites. Replace `memory = LocalMemory(...)`
with `store, retriever = LocalMemory(...)` and update `AgentConfig`
kwargs (memory → store + retriever) — phase 6's split is already in
main.

Delete `ta/agents/researcher/llm.py`.

Update `ta/agents/tests/test_config.py` patches (`create_llm` →
`from_model`, `LocalMemory` now returns tuple).

### Step 3 — collapse config builders to base + `with_()`

`ta/agents/researcher/config.py`: new `base_researcher_config(...)`
returns `AgentConfig[BacktestOutcome]` with defaults. `explorer_config`
/ `specialist_config` / `refiner_config` become thin
`base.with_(...)` functions swapping `llm`, `grader`,
`max_tool_calls`, `name`, `system_prompt`, `output_schema`. Six call
sites keep their names — signature stays compatible.

**Tests:** `test_variant_preserves_generic` — assert
`specialist_config(...).output_schema is BacktestOutcome`.

### Step 4 — typed graders

`ta/agents/researcher/graders.py`: `StrategyQualityGrader(Grader[BacktestOutcome])`,
`.grade()` reads pydantic fields. Delete `_extract_metrics`.
`StrategyViabilityGrader` (LLM judge) takes `BacktestOutcome`, passes
`output.model_dump_json()` to the judge; judge uses
`from_model(judge_model).complete(...)` with a `ViabilityScores`
response schema — no regex on the judge response either.

Rewrite `ta/agents/tests/test_graders.py`: pass
`BacktestOutcome(...)` as `output`. Shrinks to scoring-math coverage.

### Step 5 — `RunBacktestTool(dispatch=True)`

`ta/agents/researcher/tools/run_backtest.py`:

```python
class RunBacktestTool(Tool):
    dispatch: bool = True
    
    @property
    def dispatch_fn_ref(self) -> str:
        return "backtester.training.generic_objective:run_signal_study"
```

`ta/backtester/pyproject.toml`:

```toml
[project.entry-points."jig.smithers_fn"]
run_signal_study = "backtester.training.generic_objective:run_signal_study"
```

Extend `run_signal_study` signature: add `n_jobs: int = 1` (plumbed to
`study.optimize`), `study_storage_path: str | None = None`, and
`seed: int | None = None` (plumbed to `TPESampler(seed=seed)` if
supplied; otherwise Optuna's default non-deterministic seed). The
seed param lets the benchmark pin trial selection across before/after
runs so best-sharpe variance comes from the parallelism change, not
the sampler. Defaults preserve current behavior for local callers.

> **Note:** the initial ta PR (gecko#111) landed `n_jobs` +
> `study_storage_path` without `seed`. Follow-up commit on the same
> branch threads `seed` through the signature, dispatched wrapper,
> and tool schema.

**Tests:** `test_run_backtest_dispatch.py` — attrs set, registry
routes through dispatch instead of `execute()`.
`backtester/tests/test_entry_point.py` — `entry_points(group="...")`
lists our ref.

### Step 6 — migrate sweep driver

Rewrite `ta/agents/researcher/sweep.py`:

- Explorer phase: `jig.sweep(cases=[prompt]*N, configs=[explorer_cfg],
  concurrency=1)`. Pull `run.result.parsed.strategy_types`.
- Specialist phase per type: `jig.sweep(cases=[prompt(i)]*N,
  configs=[specialist_cfg_for_type], concurrency=3)`.
- Refiner phase: concurrency 1.
- Sharpe ranking: `run.result.parsed.sharpe_ratio`.
- Delete `_parse_strategy_types`, `_extract_sharpe`.

**Tests:** `test_sweep_driver.py` — patch `run_agent` to return canned
`AgentResult(parsed=BacktestOutcome(...))`; assert ranking correctness.

### Step 7 — worker deploy: install `backtester`, mount NFS

`smithers/deploy/install.sh`: add `--extra-package` (repeatable);
invoke for worker role with
`--extra-package /ta-repo/backtester --extra-package /ta-repo/storage
--extra-package /ta-repo/decision`.

Willie NFS (springfield compose side):

- Export `/srv/ta-repo` ro.
- Export `/srv/ta-data` rw.

Worker mounts (systemd automount):

- `/ta-repo` ← `willie:/srv/ta-repo` ro (include `actimeo=0` or
  `noac` to avoid stale dentry caching for `.py` files).
- `/ta-data` ← `willie:/srv/ta-data` rw.

**Tests:** manual. Deploy, `systemctl restart smithers`, confirm
`FunctionExecutor` logs `run_signal_study` in startup summary.

### Step 8 — benchmark harness

New `ta/scripts/bench_specialist_sweep.py`. CLI:
`--mode {before,after}`, `--types`, `--n`, `--trials`, `--n-jobs`,
`--concurrency`. Emits `bench_results/<ts>.json` with wall-clock,
per-phase timing, per-worker CPU, best-sharpe distribution, cost,
error counts. Run twice per mode, report median.

### Step 9 — scripts/tests cleanup

`ta/scripts/refine_liquidity_snap.py`,
`ta/scripts/refine_orderflow_imbalance.py` — no code change (call
compatible signature).
`ta/docs/strategy-researcher-spec.md:391` — prose update to describe
typed grader.

## Risks and edge cases

- **Dispatch overhead vs backtest duration** — at ~2.5min per backtest
  with ~3s dispatch overhead, ~2% tax. Fine. Instrument `task_run`
  span vs inner `fn:run_signal_study` span; if gap exceeds 10%,
  investigate.

- **NFS dentry caching** — workers may `import_module` a stale .py if
  the orchestrator just wrote a new version. Mitigation:
  `importlib.invalidate_caches()` in the dispatched call, plus
  `actimeo=0` on the worker mount. Smoke test: 3 unique strategy
  names submitted in quick succession; each worker sees the right one.

- **Optuna SQLite on NFS** — **must** keep studies DB worker-local.
  `run_signal_study` gains `study_storage_path` param; orchestrator
  leaves it `None` for dispatch, worker resolves to `/tmp/optuna_studies.db`.
  Getting this wrong collapses the speedup to zero.

- **n_jobs GIL contention** — start at 4, measure; lower to 2 if
  single-core saturation appears in `htop`.

- **Worker API keys** — `run_signal_study` doesn't call LLMs. Safe.
  Future dispatched tools that do (e.g., moving `WriteStrategyTool`)
  would need `ANTHROPIC_API_KEY` on workers — out of phase 12 scope.

- **Grader sees `output=None`** — `AgentResult.parsed` is `None` when
  parse retries exhaust. Graders return
  `Score("parse_failure", 0.0, HEURISTIC)` — matches current behavior,
  just triggered by pydantic failure instead of regex miss. Document.

- **`ConditionalCompositeGrader` with `output=None`** — heuristic
  returns `parse_failure`, avg < threshold, skip LLM judge. Test this
  path explicitly.

- **Sweep retry semantics** — `jig.sweep` doesn't retry. 99%
  per-run success rate × 50 runs → ~1–2 failures per sweep. Acceptable;
  they show up in `SweepResult.rollup().error_categories`. `retry=N`
  is phase-10 territory.

- **Trace rollup for 50-run sweeps** — ~1000 spans total; fan-out
  across 3 workers via httpx. Should complete in seconds. If
  `FederatedTracer.get_trace(sweep_id)` takes >10s, rollup pagination
  becomes phase-13 material. Measure but don't preempt.

- **Phase 6 LocalMemory split is already merged** — ta's current
  `memory = LocalMemory(...)` is already broken against jig main.
  This migration *fixes* that breakage; there's no rolling-compat
  story, one big PR. ta's `uv.lock` bumps jig from 8eefb9e (pre-phase-0)
  to post-phase-11 main — 9 phases in one lock bump.

- **Explorer schema differs from specialist** —
  `AgentConfig[T]`'s `T` is erased at runtime. `.with_(output_schema=...)`
  changes the field; the type-checker may complain unless both base
  configs are separately parameterized. Runtime sanity: assertion
  tests pin `explorer_cfg.output_schema is ExplorerOutcome` and
  `specialist_cfg.output_schema is BacktestOutcome`.

- **Phase 13 (scout + algerknown) doesn't share these surfaces** —
  those repos live outside this workspace and don't consume
  `create_llm` / `CostTrackingLLM` (ta-local surfaces). Phase 12
  doesn't gate 13; they can run in parallel.

## Benchmark methodology

### Setup

- Models: fix explorer=`claude-sonnet-4-*`,
  specialist=`claude-haiku-4-5-*`, refiner=`claude-sonnet-4-*`,
  worker=`claude-sonnet-4-*`.
- Exchange: hyperliquid.
- Window: 6 months of 1h perp data, same both runs.
- Strategy types: `mean_reversion`, `momentum`, `volatility` (skip
  explorer variance).
- Optuna seed: `TPESampler(seed=42)` — requires passing seed through
  dispatch.

### Runs

Four total, two per mode to control for warm-up variance:

- `before-1` / `before-2`: sequential, `n_jobs=1`, no dispatch.
- `after-1` / `after-2`: `concurrency=3`, `n_jobs=4`, dispatch=True.

### Metrics

- Total wall-clock (headline).
- Per-phase wall-clock.
- Per-worker CPU seconds (from trace span aggregations).
- Cost ($): should match between runs; significant after-mode increase
  means extra LLM calls snuck in.
- Best-sharpe distribution per strategy type — Mann-Whitney at p=0.05
  to confirm quality didn't regress.
- Error-category counts from `SweepResult.rollup()`.

### Pass/fail

- **Pass (merge):** after median ≤ 15h (1.9× the 8h claim,
  conservative), no quality regression, dispatch errors <5%.
- **Conditional pass:** 15h < wall-clock ≤ 25h — merge and correct
  roadmap claim to "3–4×."
- **Fail (do not merge):** >25h, quality regression, or error rate
  >5%. Diagnose first.

## Short implementation order

1. Schemas (step 1) — pure addition, ships alone.
2. `from_model` swap + delete `llm.py` (step 2).
3. Collapse builders (step 3).
4. Typed graders (step 4).
5. `RunBacktestTool(dispatch=True)` + entry point (step 5).
6. Rewrite sweep driver (step 6).
7. Worker deploy + NFS mount (step 7) — separate PR against smithers.
8. Benchmark harness + run (step 8) — after both PRs merged.
9. Tests + docs cleanup (step 9) — rolls into ta PR.

Steps 1–6 + 9 ship as one PR against ta. Step 7 ships as a smaller PR
against smithers. Step 8 runs post-merge, results recorded for the
roadmap update.
