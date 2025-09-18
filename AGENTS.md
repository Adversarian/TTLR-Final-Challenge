# Agent Operating Manual

## Non-Negotiable Ground Rules

- Orchestrate every LLM or embedding interaction through LlamaIndex primitives (FunctionAgent, Workflows, Memories, Observability) and consult upstream docs when in doubt.
- Manage Python dependencies exclusively with `uv`; do **not** invoke `pip` directly.
- Ship and run the service from a single Docker image. The chatbot must boot correctly when launched with `docker run` against Docker Engine.
- Develop scenario by scenario, extending behavior without regressing previously satisfied scenarios from the problem statement.
- Provide a self-contained `/chat` endpoint that satisfies the exact request/response schema in `PROBLEMSTATEMENT.md`; any additional endpoints are optional conveniences only.
- Capture complete, replayable judge conversations: stream traces to Arize Phoenix **and** persist structured JSONL transcripts locally along with a replay tool.
- Keep the implementation minimal, readable, and free of unnecessary abstraction—opt for maintainability under time pressure.
- Re-read the official docs for any third-party API before integrating to avoid stale assumptions.
- Assume infrastructure endpoints (Postgres, Phoenix, external APIs) are environment-driven; never hard-code deployment values.
- Maintain Phoenix instrumentation hooks even if credentials are missing; the app should degrade gracefully without telemetry.

## Reference Notes
- Re-read `PROBLEMSTATEMENT.md` whenever notified of changes; never modify it locally.
- Competition datasets include Searches, Base Views, Final Clicks, Base Products, Members, Shops, Categories, Brands, and Cities. Expect Parquet inputs loaded into Postgres.
- Evaluation spans sanity checks plus scenarios 1-9 (direct lookup, attribute Q&A, seller pricing, guided discovery, comparisons, vision, similarity, ranking). All responses must respect `{message, base_random_keys, member_random_keys}` with scenario-specific constraints.
- Judges expect a deployed `/chat` endpoint, public access, supporting documentation, and a short architecture overview video.

## Operational Constraints & Tactics
- Target <30s end-to-end latency per request; minimize token usage and keep reasoning concise.
- Favor deterministic SQL and structured retrieval from Postgres; add semantic/vector layers only when scenarios demand.
- Emit `base_random_keys`/`member_random_keys` only when confident—judges stop evaluation once non-null values appear.
- Keep verbal responses concise yet sufficient for scenario acceptance criteria.
- Introduce vector search (e.g., pgvector/Qdrant) only if similarity recall becomes a blocker (notably scenarios eight and nine).
- Choose LLM tiers pragmatically: reserve strongest models for critical reasoning steps and default to cost-effective options otherwise.

## Data & AI Stack Decisions
- Expect Torob datasets as Parquet exports; `scripts/ingest.py` should ingest them into Postgres with configurable data dirs.
- SQL-first retrieval remains the baseline. Embedding/index pipelines (OpenAI + pgvector or Qdrant) can augment recall where justified.
- LlamaIndex will broker SQL tools, retrievers, and synthesis across FastAPI handlers.
- All LLM credentials come from environment variables (e.g., `OPENAI_API_KEY`); never commit secrets.

## Telemetry & Replay Strategy
- Maintain dual logging: Phoenix traces for observability **and** append-only JSONL transcripts stored under `REPLAY_LOG_DIR`.
- Provide a CLI/pytest harness that replays transcripts to validate determinism and regression-test new scenarios.
- Document how to persist replay logs when the single-container runtime restarts (e.g., by mounting a host volume).

## Implementation Notes (to be validated/updated)
- FastAPI application code lives in `app/`. Verify existing scaffolding before reuse; modernize to LlamaIndex workflows.
- `app/config.py` should centralize environment configuration (database DSN, Phoenix URLs, replay dir, model choices).
- Add a conversation logging module that appends interactions to JSONL and exposes a replay utility under `scripts/` or `tests/`.
- Use `uv` commands (`uv run`, `uv sync`) in Dockerfiles, scripts, and docs.

## Deployment Notes
- Primary runtime is a standalone Docker image; ensure the container exposes the API without relying on Compose.
- Container entrypoint should leverage `uv run python -m app.server` (or equivalent) and honour env-based configuration.
- Keep `.env.template` synced with runtime requirements (DB URL, Phoenix, replay log dir, OpenAI keys, etc.).
- Avoid hard-coded hostnames—favor env variables for all external integrations.
