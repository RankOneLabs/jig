# Phase 13 plan — Scout + Algerknown lift

**Goal.** Scout and algerknown migrate from bespoke LLM + vector
plumbing to jig's redesigned stack. Scout collapses its four-call
`evaluate/generate/critique/revise` pipeline into one `run_agent`
call with a typed `ReplyCandidate` output + a revision tool.
Algerknown's proposer and synthesizer stop importing the `anthropic`
SDK, typed pydantic schemas replace regex-stripped JSON, and ChromaDB
is replaced with jig's `MemoryStore` + `DenseRetriever` split. Both
apps stop pinning pre-phase-0 jig and jump to post-phase-11 main.

## Scope

**In:**

- Scout: new `AgentConfig[ReplyCandidate]` agent on phase-0
  `jig.from_model`. `RelevanceFilter.evaluate`,
  `CommentGenerator.generate`, `CommentCritic.critique`,
  `CommentGenerator.revise` stop being four separate
  `AsyncAnthropic.messages.create` calls; they become the agent's
  initial prompt, `submit_output`, and a `revise_draft` user tool the
  model may call zero-or-more times. Critique becomes internal
  reasoning inside the same agent loop.
- Scout: `anthropic.AsyncAnthropic` deleted from `relevance_filter.py`,
  `comment_generator.py`, `comment_critic.py`. Only `jig.from_model` /
  `jig.complete`. `anthropic` pin drops out of `pyproject.toml`.
- Scout: outer `Pipeline`/`Step` per-message loop stays; the four-step
  inner pipeline goes away. `build_scout_pipeline` shrinks.
- Scout: `uv.lock` bumps jig from `2a4bce3` (pre-phase-0, PR #2) to
  post-phase-11 main — 9 phases in one bump, same as ta.
- Algerknown: `rag-backend/proposer.py` becomes an
  `AgentConfig[Proposal]` run; `rag-backend/synthesizer.py` becomes a
  `run_agent` with `output_schema=SynthesizedAnswer`. `anthropic`
  import gone from both.
- Algerknown: `rag-backend/vectorstore.py` replaced with jig's
  `SqliteStore` + `DenseRetriever`. `chromadb` removed from
  `requirements.txt`. `chroma_db/` dir retired — one-shot reindex
  from `content-agn/` YAMLs on first boot.
- Algerknown: store persists at a renamed path
  (`CHROMA_DB_DIR` → `MEMORY_DB_PATH`, default `./memory.db`).
  Docker-compose volume mount updates; `chroma_db/` folder deleted.
- Both apps: `ANTHROPIC_API_KEY` env var unchanged. A new
  `SQLiteFeedbackLoop` is constructed at startup (required by
  `AgentConfig`), `include_feedback_in_prompt=False` — no prior
  feedback data to inject.

**Out (deferred):**

- Phase 10 (callback-based sweep fan-out) — neither app runs sweeps
  today; wiring it is follow-up.
- Dispatching scout's / algerknown's LLM calls to smithers workers —
  scout's scanning is IO-bound, algerknown's proposer is interactive,
  no fan-out need.
- Scout's `PastResults` lookup for prior critiques (has its own
  `get_recent_critique_feedback` already — clean migration is nice
  but not required).
- Algerknown's OpenAI-embedding path stays behind a custom
  `Embedder` callable (see key design call 5); no Ollama hard
  requirement.
- Algerknown's changelog / diff_engine / writer / loader — operate
  on YAML on disk, don't touch the vector store.
- The `packages/web` TypeScript UI — only talks to FastAPI over HTTP.

## Prerequisite audit

Concrete file:line references against current working trees.

### Scout — current code

- **Four-call pipeline** in `engagement-scout/pipeline.py:25-74`
  (`evaluate_step`, `generate_step`, `critique_step`, `revise_step`);
  wired in `build_scout_pipeline` at `pipeline.py:103-116`. Driven
  by `run_pipeline` at `scout.py:258` inside `score_messages`.
- **Anthropic SDK call sites** (3 files, 4 call sites):
  - `relevance_filter.py:11` import, `:103` client, `:134` create
  - `comment_generator.py:8` import, `:103` client, `:126` generate,
    `:218` revise
  - `comment_critic.py:10` import, `:111` client, `:133` create
- **Typed dataclasses already in place**: `RelevanceResult`,
  `GeneratedDraft`, `CritiqueResult`, `Message` at
  `config.py:190-238`. Plain dataclasses; need pydantic
  equivalents (or compose a `BaseModel`-inheriting `ReplyCandidate`)
  because `AgentConfig.output_schema` requires `BaseModel`.
- **Free-form JSON parsing that goes away**: markdown/regex
  stripping in `relevance_filter.py:66-90` and
  `comment_critic.py:67-99`. Replaced by pydantic validation of
  `submit_output` args.
- **Existing jig integration (stays)**: `scout.py:22-23` +
  `pipeline.py:12` import `TracingLogger`, `run_pipeline`,
  `SQLiteTracer`, `PipelineConfig`, `Step`. These survive on
  post-phase-11 main (`src/jig/core/pipeline.py:12,20,52`). Outer
  pipeline scaffolding doesn't change.
- **Jig pin**: `pyproject.toml:24-25` git dep; `uv.lock:425-427`
  pins `2a4bce32d6c742a6a6980cbd544fdd4a99c7d095` (PR#2 merge,
  pre-phase-0). 9-phase bump.
- **Tests in blast radius** (~600 lines):
  `tests/test_pipeline.py` (244 lines),
  `tests/test_relevance_filter.py` (55),
  `tests/test_comment_generator.py` (155),
  `tests/test_comment_critic.py` (148). End state ~300–400 lines.
- **Deployment**: Dockerfile runs
  `uv run python -m scout --continuous`. `VOLUME /app/data`
  persists `scout.db` + `scout_traces.db` + digests.
- **Models**: `config.py:43-44` — `LLM_MODEL=claude-sonnet-4-6`,
  `CLASSIFIER_MODEL=claude-haiku-4-5-20251001`. Both match jig's
  `claude-*` prefix route (`src/jig/llm/factory.py:29-31`).

### Algerknown — current code

- **Proposer** `rag-backend/proposer.py`:
  - `:8` `import anthropic`, `:21-26` `get_anthropic_client`,
    `:190-195` `messages.create`, `:199-205` markdown code-block
    stripping + `json.loads`, `:221` `APIError` handler.
  - Model hardcoded default `claude-sonnet-4-20250514` at `:141`
    — older than scout's; migration opportunity to centralize.
- **Synthesizer** `rag-backend/synthesizer.py`:
  - Two call sites: `:87-92` one-shot, `:163-169` follow-up with
    history. Both plain `messages.create`, no tools. Free-form
    text output.
- **Vectorstore** `rag-backend/vectorstore.py`:
  - `:7-8` `chromadb` + `embedding_functions`.
  - `:103` `VectorStore` wraps `chromadb.PersistentClient`.
  - Single `algerknown` collection; mixes `type: "entry"` and
    `type: "summary"` rows. Only `where={"type": "summary"}` at
    proposer `:78`.
  - Own chunking at `:124-166`, parent-id reconstruction at
    `:168-200`. Non-trivial logic.
  - `get_summaries()` at `:300` — full scan + metadata filter, not
    a similarity query. Replaced by
    `store.iter_entries_with_embeddings()` + Python filter.
  - Embeddings: 384-dim `sentence-transformers/all-MiniLM-L6-v2`
    by default; OpenAI `text-embedding-3-small` when
    `OPENAI_API_KEY` is set; mock for tests. Jig default: Ollama
    `nomic-embed-text` at 768.
- **ChromaDB data**: `rag-backend/chroma_db/` ~11MB;
  `chroma.sqlite3` + one collection UUID dir. ~50 rows total
  (38 entries + 15 summaries). Rebuild cost: ~30s from yaml.
- **API** `rag-backend/api.py`: FastAPI app, port 4735. Five
  endpoints touch `vector_store`: `:139, :192, :367, :407, :451,
  :489, :538`.
- **Tests** (~600 lines): `test_proposer.py` (~400),
  `test_synthesizer.py` (~200), `test_vectorstore.py`. All three
  rewrite.
- **No jig dependency currently** — addition, not pin bump.
- **Deployment**: `rag-backend/docker-compose.yml` builds one
  service, port 4735, mounts `../content-agn:/app/content-agn` +
  `./chroma_db:/app/chroma_db`. Rename env +  volume.

### Jig surfaces present on main (no changes needed)

- `jig.from_model(...)` at `src/jig/llm/factory.py:27`, prefix-routes
  `claude-*` to `AnthropicClient`.
- `jig.complete(...)` at `src/jig/llm/factory.py:140` — one-shot.
- `AgentConfig[T]` at `src/jig/core/runner.py:61` with
  `output_schema` (`:102`), `max_parse_retries` (`:103`), `.with_()`
  (`:125`). Already supports revise-tool pattern — the runner keeps
  looping while the model calls non-`submit_output` tools.
- `submit_output` injection at `src/jig/core/runner.py:162, 288-297`.
  Reserved name (`:372-392`); ambiguous-turn guard in place.
- `Grader[T]` at `src/jig/core/types.py:329`.
- `MemoryStore` at `src/jig/core/types.py:228`; `Retriever` nearby.
  `SqliteStore` + `DenseRetriever` at `src/jig/memory/local.py:59,
  255` with `LocalMemory()` factory at `:304-320`.
- `SqliteStore` accepts custom `embedder: Embedder | None`
  (`:74`; type alias `Embedder` at `:34`). Hook for algerknown's
  OpenAI / sentence-transformers paths.
- `Tool` at `src/jig/core/types.py:385`. Revision tool is just a
  `Tool` subclass.

### Revision loop: no new jig abstraction needed

`run_agent` already supports "call tool X zero-or-more times before
`submit_output`." Ambiguous-turn guard only fires when the model
calls `submit_output` **alongside** another tool. Calling
`revise_draft` on one turn and `submit_output` on a later turn is
the happy path. Scout's revision loop is one more registered `Tool`
— no new runner hook.

Confirmed: `src/jig/core/runner.py:371-522` — when `submit_output`
isn't in the turn, loop falls through to "Execute user-tool calls"
at `:521`.

### Phase-10 parallelism

Phase 10 (callback-based sweep fan-out) is internal to jig — adds a
listener to `jig.dispatch`, rewires `jig.sweep`. No file overlap
with phase 13's scope (consumer repos, never `src/jig/*`). Safe to
merge either order.

## Key design calls

### 1. Collapse to one agent; critique becomes internal reasoning

Today scout makes up to **four** LLM calls per relevant message.
After: **one** `run_agent` invocation. System prompt says "evaluate
relevance; if relevant, draft; self-critique; revise if needed via
`revise_draft`; submit final answer via `submit_output`."

The `revise_draft` tool takes `{draft, feedback}` and returns a
prompt-shaped string ("Here is the draft and feedback; now rewrite
it"). Its purpose: **force the model to commit to a revised draft**
rather than drifting in one free-form response.

`max_tool_calls=3` caps the revise loop. `max_parse_retries=2`
handles pydantic validation failures on `submit_output`. Collapses
2–4 API calls to 1–2.

### 2. Typed `ReplyCandidate` covers both relevant + not-relevant

```python
class ReplyCandidate(BaseModel):
    relevant: bool
    score: float
    reason: str
    relevant_to: list[str]
    comment_text: str | None  # None iff relevant=False
    project_key: str | None   # None iff relevant=False
    critique_verdict: Literal["approve", "revise", "reject"] | None
    critique_feedback: str | None
```

When `relevant=False`, text fields are `None` and the run ends after
the first `submit_output` turn. Scout's existing `RelevanceResult` /
`GeneratedDraft` / `CritiqueResult` dataclasses stay for the DB-save
path; a `_unpack` converter translates between `ReplyCandidate` and
the three-dataclass shape.

### 3. Outer pipeline stays; inner pipeline shrinks

`scout.py:258`'s `run_pipeline(pipeline, input=msg, context=ctx)`
remains useful for tracing, skip-predicates, error handling. Inner
pipeline shrinks to:

```python
Step(name="keyword_prefilter", fn=prefilter_step, skip_when=...)
Step(name="score_and_draft", fn=agent_step)  # calls run_agent
```

### 4. Algerknown ChromaDB: migrate by rebuild, not data-copy

`chroma_db/` is 11MB; source of truth is `content-agn/*.yaml` on
disk. Every row can be rebuilt by rerunning `load_content(...)` +
`store.add(...)` on boot — which is what `api.py:lifespan` already
does via `vector_store.index_documents(...)` at `:73`.

Keep exactly that pattern. First boot of the migrated service sees
empty `memory.db`, calls `load_content`, indexes ~50 documents.
No migration script. `chroma_db/` deleted in the PR; first deploy
rebuilds in ~30s.

Trade-off: if an operator manually added anything to chromadb that
isn't in `content-agn/`, it's lost. Inspection: every write goes
through `index_documents` called from `api.py`'s ingest/approve/
reindex, each sourced from yaml. Safe.

### 5. Algerknown embeddings: custom `Embedder` callable, start with today's preferences

Options considered:

- **a.** Adopt jig default (Ollama `nomic-embed-text`). Requires
  Ollama on frink — available via ta's setup, but adds a service
  dependency.
- **b.** Port OpenAI behind jig's `Embedder` callable.
  `SqliteStore(embedder=openai_embed_fn)`. Clean.
- **c.** Port sentence-transformers as default.

**Chosen: (b) with (c) as fallback.** Mirrors today's
`USE_LOCAL_EMBEDDINGS=true`, `OPENAI_API_KEY` preference. Keeps
migration scope small — `chromadb` → `SqliteStore`, one class swap.
Keeps `sentence-transformers` in requirements for offline/no-network
fallback.

`MockEmbeddingFunction` at `vectorstore.py:55` becomes
`mock_embed_fn`.

### 6. Chunking: port into a preprocessor module

Algerknown's `_chunk_text` at `vectorstore.py:124` splits docs
>6000 chars into multiple embeddings with `parent_id` metadata.
Jig's `SqliteStore` is one-row-per-`add`. Port chunking as a
pre-`add` helper `_index_with_chunking(store, docs)`; query-side
reconstruction as `_reconstruct_documents(hits)`. ~50 lines in a
thin `memory_store.py` wrapper.

### 7. Proposer: metadata scoring stays Python-side, retriever does similarity

Today's proposer ranks candidate summaries three ways:

1. Explicit-link matches (score 1.0).
2. Semantic similarity via `vector_store.query`.
3. Tag/topic overlap boost.

Only (2) is a vector query. `DenseRetriever.retrieve(query, k)`
returns `list[MemoryEntry]` with `.score` = cosine similarity —
replaces (2). Keep (1) + (3) in a post-retrieval
`rank_candidate_summaries(...)` Python function. No loss in
semantics; cleaner separation.

### 8. Proposer: typed `Proposal` schema

```python
class Proposal(BaseModel):
    target_summary_id: str
    source_entry_id: str
    new_learnings: list[NewLearning] = []
    new_decisions: list[NewDecision] = []
    new_open_questions: list[str] = []
    new_links: list[NewLink] = []
    rationale: str
    no_updates: bool = False
```

`api.py`'s `ProposalData` pydantic model (`:315-324`) becomes an
alias for `Proposal` — same shape. Regex stripping at
`proposer.py:199-205` and `json.loads` at `:206` both disappear.

### 9. Synthesizer: `run_agent` with `output_schema=SynthesizedAnswer`

```python
class SynthesizedAnswer(BaseModel):
    answer: str
    cited_document_ids: list[str]
```

`synthesize_answer` becomes a minimal `run_agent` with no tools and
this output schema. The "agent loop" is one turn: model calls
`submit_output` with `{answer, cited_document_ids}`. Cost: no
extra API calls (submit_output is embedded in the same response).
Migration stays uniform with proposer.

### 10. FeedbackLoop instantiation

`AgentConfig.feedback` is required. Both apps instantiate
`SQLiteFeedbackLoop`. Scout: `scout_feedback.db` (separate from
`scout.db`). Algerknown: `feedback.db` next to `memory.db`.
`include_feedback_in_prompt=False` in both — no prior data to
inject.

### 11. Port allocations

No new listeners. Scout has no HTTP surface. Algerknown keeps port
4735. Env rename `CHROMA_DB_DIR` → `MEMORY_DB_PATH`, compose
volume updates (`./chroma_db` → `./memory_db`).

## Per-app step-by-step plan

**Scout and algerknown are independent.** No shared files, runtime,
or database. Both consume jig's post-phase-11 main. Branches run in
parallel; merge order doesn't matter. Each ships as one PR against
its own repo. Jig itself unchanged in phase 13.

### Scout (branch: `scout/phase-13-jig-lift`)

#### S1 — pydantic schema module

New `engagement-scout/schemas.py` with `ReplyCandidate`. Keep the
existing dataclasses in `config.py` for DB writes; add converters.

**Tests:** `tests/test_schemas.py` — round-trip, `relevant=False`
short-circuit, required-field validation. Prep commit.

#### S2 — bump jig, wire `FeedbackLoop` + `TracingLogger`

`pyproject.toml`: drop `anthropic`. `uv lock` rewrites the jig pin
to post-phase-11 main.

`scout.py`: `from jig.feedback.loop import SQLiteFeedbackLoop`;
instantiate `feedback = SQLiteFeedbackLoop(db_path="scout_feedback.db")`
next to the existing `tracer` at `:370`.

**Tests:** existing `AsyncAnthropic`-mocking tests break at import
time — mark `@pytest.mark.skip("phase 13 migration")`; rewritten or
deleted in S5.

#### S3 — scout agent config + revision tool

New `engagement-scout/scout_agent.py`:

```python
def build_scout_agent_config(
    tracer, feedback, classifier_model, lessons, grading_signals,
) -> AgentConfig[ReplyCandidate]:
    tools = ToolRegistry([ReviseDraftTool()])
    return AgentConfig[ReplyCandidate](
        name="scout",
        system_prompt=_build_system_prompt(lessons, grading_signals),
        llm=from_model(classifier_model),
        feedback=feedback,
        tracer=tracer,
        tools=tools,
        output_schema=ReplyCandidate,
        max_tool_calls=3,
        max_parse_retries=2,
    )


class ReviseDraftTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="revise_draft",
            description="Rewrite a draft based on critique feedback...",
            parameters={
                "type": "object",
                "properties": {
                    "draft": {"type": "string"},
                    "feedback": {"type": "string"},
                },
                "required": ["draft", "feedback"],
            },
        )

    async def execute(self, args: dict[str, Any]) -> str:
        return (
            f"Original draft:\n{args['draft']}\n\n"
            f"Critique feedback:\n{args['feedback']}\n\n"
            "Now produce a revised draft and call submit_output with it."
        )
```

Tool is a **prompt-builder**, not an LLM call. Model sees tool
result as a user-turn message; produces revised draft next turn;
calls `submit_output`.

**Tests:** `tests/test_scout_agent.py` — config build doesn't
raise; tool returns expected prompt; `output_schema is ReplyCandidate`.

#### S4 — rewrite pipeline + score_messages

`pipeline.py` collapses to `keyword_prefilter` + `score_and_draft`.
The `score_and_draft` step calls
`run_agent(config, input=_format_message_for_prompt(msg))` and
returns the parsed `ReplyCandidate`.

`scout.py:score_messages` swaps:

- Old: four-step `result.step_outputs` unpacking → four dataclasses.
- New: one `parsed = result.step_outputs.get("score_and_draft");
  eval_result, draft, critique, comment = _unpack(parsed)`.

`_unpack` converts `ReplyCandidate` → `(RelevanceResult,
GeneratedDraft, CritiqueResult, str)`. Zero change to
`state_manager.py`.

Delete `relevance_filter.py`'s `evaluate` body (class stays for
`keyword_prefilter` or folds into `prefilter.py`). Delete
`comment_generator.py` and `comment_critic.py` entirely —
functionality moves into the agent's system prompt + tools.

#### S5 — tests

Delete `tests/test_relevance_filter.py`,
`tests/test_comment_generator.py`, `tests/test_comment_critic.py`
(~350 lines combined).

Rewrite `tests/test_pipeline.py` as `tests/test_scout_agent.py`:

- Agent config shape (output_schema, tool registry, model routing).
- Fake `LLMClient` returning canned `submit_output` calls with
  `ReplyCandidate` args; assert parsed matches.
- Fake `LLMClient` returning `revise_draft` first then
  `submit_output`; assert `max_tool_calls=3` honored.
- `_unpack` converter tests — `relevant=False` path.

Expected: ~300 lines new replacing ~600 lines old.

#### S6 — docs

`README.md` drops four-step pipeline refs; `docs/architecture.md`
rewrites the evaluate/generate/critique/revise section.

### Algerknown (branch: `algerknown/phase-13-jig-lift`)

#### A1 — add jig dep, set up memory

`rag-backend/requirements.txt`: remove `chromadb>=0.4.18`; optionally
remove `sentence-transformers>=2.2.0` (see A4 risk section). Add
`jig @ git+https://github.com/rankonelabs/jig.git@main`.

New `rag-backend/memory_store.py` (replaces `vectorstore.py`):

```python
async def build_memory(
    db_path: str, openai_key: str | None,
    use_local: bool, use_mock: bool,
) -> tuple[SqliteStore, DenseRetriever]:
    embedder = _select_embedder(openai_key, use_local, use_mock)
    store = SqliteStore(db_path=db_path, embedder=embedder)
    retriever = DenseRetriever(store)
    return store, retriever
```

Three `Embedder` implementations in the same module: OpenAI,
sentence-transformers (if kept), mock. Port `_chunk_text`
(unchanged, from `vectorstore.py:124-166`) and
`_reconstruct_documents` (from `:168-200`).

Port `VectorStore.query(...)` as `search(retriever, store, query,
n_results, where)` — `retriever.retrieve(query, k=n_results*3)` +
metadata filter + dedupe chunks. Returns the same `list[dict]`
shape.

`VectorStore.get_summaries()` → `get_summaries(store)`: scan
`store.iter_entries_with_embeddings()` + filter `type == "summary"`.
O(n) at 50 rows — fine.

`VectorStore.index_documents(...)` → `index_documents(store, docs)`:
chunking + `await store.add(content, metadata)` loop.

**Tests:** `tests/test_memory_store.py` replaces
`tests/test_vectorstore.py`. Three `Embedder` impls (mock always;
OpenAI + sentence-transformers behind env flags), chunking
round-trip, query filtering.

#### A2 — rewrite api.py

Replace `from vectorstore import VectorStore` with
`from memory_store import build_memory, search, get_summaries,
index_documents` at `api.py:24`.

In `lifespan` (`:53-81`):

```python
store, retriever = await build_memory(
    db_path=MEMORY_DB_PATH,
    openai_key=os.getenv("OPENAI_API_KEY"),
    use_local=os.getenv("USE_LOCAL_EMBEDDINGS", "").lower() in ("true", "1"),
    use_mock=os.getenv("USE_MOCK_EMBEDDINGS", "").lower() in ("true", "1"),
)
```

Every `vector_store.query(...)` → `search(retriever, store, ...)`
(lines `:139, :192`). Every `vector_store.index_documents(...)` →
`await index_documents(store, ...)` (`:73, :367, :407, :451, :489`).
`vector_store.get_summaries()` → `get_summaries(store)` (`:538`).
`vector_store.count()` → `len(await store.all())`.

Note: these are now **async**. FastAPI handler signatures become
`async def` where today they're `def`. Mechanical change.

Env rename `CHROMA_DB_DIR` → `MEMORY_DB_PATH`, default `./memory.db`.

Delete `rag-backend/vectorstore.py` at end of step.

#### A3 — proposer rewrite as `AgentConfig[Proposal]`

New `rag-backend/proposer.py` structure: `Proposal` schema,
`build_proposer_config(...)`, `propose_updates(entry, summary,
config)` awaits `run_agent` and returns `result.parsed.model_dump()`.
`generate_all_proposals` becomes async; `api.py:380` gets `await`.

Delete `get_anthropic_client`, the JSON-parsing block at `:199-205`,
the `anthropic.APIError` handler at `:221`.

**Tests:** `tests/test_proposer.py` replaces `mock_anthropic_client`
fixture with a fake `LLMClient` returning a canned `submit_output`
call. Assert `Proposal` round-trips through `model_dump()`. Keep
`identify_related_summaries` tests — they operate on store shape,
covered by `test_memory_store.py`'s mock embedder.

#### A4 — synthesizer as `run_agent`

New `rag-backend/synthesizer.py`: `SynthesizedAnswer` schema,
`build_synthesizer_config(...)`, `synthesize_answer(query,
retrieved, config)` that calls `run_agent` and returns
`result.parsed`.

`synthesize_with_followup` ports analogously if any call site
exists outside `synthesizer.py` itself (grep `api.py` — likely
drops).

Delete anthropic imports + `get_anthropic_client` +
`APIError` handlers.

**Tests:** `tests/test_synthesizer.py` rewrites. ~200 → ~120 lines.

#### A5 — compose + env + chroma_db/ delete

- Delete `rag-backend/chroma_db/` from the working tree.
- `docker-compose.yml`: `./chroma_db:/app/chroma_db` →
  `./memory_db:/app/memory_db`; env
  `CHROMA_DB_DIR=/app/chroma_db` → `MEMORY_DB_PATH=/app/memory_db/memory.db`.
- `api.py:42` reads the new env.
- `README.md` + `LLM_INSTRUCTIONS.md` setup updates.
- Bump `healthcheck.start_period: 120s` in compose (first-boot
  reindex is ~100s).

#### A6 — deploy on frink

First boot: empty `memory_db/`, `lifespan` calls
`index_documents(store, load_content(CONTENT_DIR))`. ~50 docs ×
~2s/embed = ~100s.

## Risks and edge cases

### Both apps

- **Jig pin jump (scout) — 9 phases in one bump**. Same risk
  profile as ta phase 12. Smoke-test the single-agent flow
  locally against a real `ANTHROPIC_API_KEY` before merging.
  `Pipeline`/`Step` survive on main.

- **`FeedbackLoop` mandatory**. Neither app has prior data.
  Empty path → `SQLiteFeedbackLoop` creates schema on first use.
  `include_feedback_in_prompt=False` means agent never queries
  it. Inert.

- **`ANTHROPIC_API_KEY` unchanged**. Jig's convention matches the
  SDK's.

### Scout-specific

- **Token-budget shift**. Old: 4 × 300 = 1200 max_tokens per
  message. New: ~2400 (evaluate reasoning + draft + optional
  revise + submit_output args). Still cheaper per message: one
  call, one set of system-prompt tokens. Measure prompt+completion
  tokens before/after on ~20 messages; report in PR.

- **Tool-use on Anthropic via jig**. `AnthropicClient` supports
  tools (phase 0 — works in ta). Both configured models support
  tool use. Smoke-test before merging.

- **`revise_draft` may go unused**. Fine — it's an affordance, not
  a requirement. `max_tool_calls=3` caps runaway spin.

- **Ambiguous-turn guard**. If model emits both `submit_output`
  and `revise_draft` same turn, runner retries (`runner.py:384-392`).
  `max_parse_retries=2` caps. Document in system prompt: "Do not
  call revise_draft and submit_output in the same turn."

- **Prompt library consolidation**. Today's
  `prompts/lead_gen|temp_check|critique|comment_draft` → one
  rendered string via `_build_system_prompt(mode, lessons,
  signals)`. Keep the file-per-prompt layout; just assemble
  differently.

### Algerknown-specific

- **ChromaDB rebuild path**. Chosen: rebuild from yaml on first
  boot. Pre-merge validation:
  `diff <(old_vector_store.get_all() | jq .id | sort)
  <(new_store.all() | jq .id | sort)` — expect zero diff. If any
  manual-insert path exists outside `index_documents`, add a
  one-shot data-copy script. Current read of `vectorstore.py`: all
  writes go through `index_documents` from yaml sources. Safe.

- **Embedding dim mismatch**. ChromaDB stored 384-dim (sentence-
  transformers) or 1536-dim (OpenAI). Jig's `SqliteStore` stores
  float32 via `emb.tobytes()` at any dim. Rebuild embeds from
  scratch with the chosen embedder — dims internally consistent.

- **OpenAI embedder cost**. ~$0.00002/1K tokens on
  text-embedding-3-small. 50 docs × ~1500 tokens = ~$0.002 per
  reindex. Negligible.

- **`sentence-transformers` as local fallback**. ~500MB model
  weights. Keep in requirements (fallback path); only `chromadb`
  removes. Container size delta: ~-50MB.

- **Async migration for api.py**. `vector_store.query(...)` sync
  → `retriever.retrieve(...)` async. Handler signatures become
  `async def`. FastAPI handles transparently. Check: `api.py`
  already mixes `async def lifespan` + sync handlers.

- **`ProposalData` → `Proposal` rename**. Both pydantic, same
  fields. Alias or rename; backwards-compatible to HTTP clients.

- **First-boot indexing time on frink**. ~100s of embed calls
  during compose up. Bump `healthcheck.start_period` from `10s`
  (`docker-compose.yml:24`) to `120s` so the orchestrator
  doesn't flap-restart during reindex. Subsequent boots skip
  reindex (store populated).

- **Untouched files**: changelog / diff_engine / writer / loader
  operate on yaml; no chromadb/anthropic refs. Zero blast radius.

## Testing strategy

**No benchmark gate.** Phase 13's value is "fewer bespoke
abstractions, one less SDK dep" — not latency or cost. Validation
is "scout and algerknown still do their job."

### Scout pre/post validation

- Run scout `--rescore` mode on a known-good scan
  (`scout.py:394`). Compare digest:
  - Relevance classification matches on ≥80% of messages
    (some drift expected — single-loop reasoning differs from
    four-step flow).
  - Comment quality not regressed (subjective; manual review of 5
    drafts before vs. after).
- Trace comparison: `scout_traces.db` old (four spans per msg) vs
  new (one `AGENT_RUN` span with tool-call children).
- Cost: `agent_result.usage.total_cost` per run; sum over 20-msg
  scan before/after; expect ~30–50% reduction (shared system-
  prompt tokens).

### Algerknown pre/post validation

- Cold-start against empty `memory_db/`; `lifespan` indexes all
  ~50 docs without error.
- `/query` with fixed query ("How do nullifiers work in Noir?")
  before/after. Similar retrieved sources (if same embedder both
  runs, identical ranks). Answer text qualitatively similar.
- `/ingest` sample new entry; proposals generate with expected
  `Proposal` shape. `/approve` round-trip writes.
- Pytest green.

### Merge gate

- Scout: pytest green, one manual ~20-message scan produces
  digest a human considers "comparable or better."
- Algerknown: pytest green, one `/query` + `/ingest` + `/approve`
  round-trip against fresh `memory_db/`.

No wall-clock SLA. No quality stat test. Maintainability migration.

## Short implementation order

### Scout

1. S1 — schemas.
2. S2 — jig bump + feedback + drop anthropic.
3. S3 — scout_agent.py (config + tool).
4. S4 — pipeline.py + scout.py glue.
5. S5 — test rewrite.
6. S6 — docs.

S1–S6 all in one PR against engagement-scout.

### Algerknown

1. A1 — memory_store.py + jig dep + chunking port.
2. A2 — api.py handler async migration.
3. A3 — proposer.py rewrite.
4. A4 — synthesizer.py rewrite.
5. A5 — compose + env + chroma_db/ delete.
6. A6 — deploy on frink (runbook, not code).

A1–A5 in one PR against algerknown.
