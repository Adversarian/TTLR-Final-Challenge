# Agent Operating Manual

## Non-Negotiable Ground Rules
- Use the Pydantic-AI framework for every LLM interaction; follow its best practices for orchestration.
- Manage Python dependencies exclusively with `uv`; do not shell out to `pip`.
- Package the service with a single Dockerfile that starts the FastAPI API directly (no docker-compose runtime in production).
- Evolve the assistant scenario by scenario, ensuring new behavior never breaks previously satisfied scenarios.
- Capture complete, replayable judge conversations locally; combine Logfire (from Pydantic-AI) with local log files plus a replay utility.
- Expose an autonomous `/chat` endpoint matching the problem specification; any extra endpoints must remain optional.
- Keep the implementation focused, minimal, and maintainable—no gratuitous abstractions or flair under time pressure.
- Re-check upstream documentation for all third-party libraries whenever APIs are used to avoid stale assumptions.

## Reference Notes
- Re-read `PROBLEMSTATEMENT.md` whenever the user reports updates; never edit that file.
- Competition data sources: Searches, Base Views, Final Clicks, Base Products, Members with rich metadata for retrieval/ranking.
- Evaluation scenarios (0-9) cover echo tests, direct lookup, attribute Q&A, seller pricing, guided discovery, comparisons, vision, and ranking.
- Responses must use `{message, base_random_keys, member_random_keys}` per spec with tight cardinality constraints per scenario.
- Deployment deliverables include a public `/chat` domain, a 5-minute architecture video, and repository access for judges.

## Operational Constraints & Tactics
- Budget: 30-second per-request timeout and limited API credits; minimize chain-of-thought length and token usage.
- Prefer deterministic SQL against the catalog DB over full RAG; use LLMs to synthesize precise SQL and interpret results.
- Delay emitting `base_random_keys`/`member_random_keys` until absolutely confident—judge may halt on first non-null value.
- Keep responses terse while satisfying scenario-specific formatting to preserve latency headroom.
- Qdrant vector store is optional; introduce only if similarity search materially helps (e.g., scenario eight). Until then rely on structured filters.
- Tier models by task criticality: use stronger reasoning models sparingly where accuracy gates success; default to cheaper/faster models otherwise.

## Data & AI Stack Decisions
- Torob datasets arrive as Parquet exports; `scripts/ingest.py` loads them into Postgres. Data directory is configurable per environment.
- SQL remains the primary retrieval surface. Optional OpenAI embeddings can pre-filter candidates before SQL when recall demands it.
- All LLM and embedding calls use OpenAI models, configured via `OPENAI_*` environment variables.

## Telemetry & Replay Strategy
- Implement dual logging: structured telemetry via Logfire/OpenAI tooling and append-only JSONL conversation transcripts.
- Replay tooling (pytest) will consume the JSONL logs to mimic judge conversations, including LLM-judged scenarios.
- Production must mount a persistent volume at `REPLAY_LOG_DIR` so conversation logs survive restarts.

## Implementation Notes
- FastAPI lives in `app/` and the `/chat` endpoint now invokes a single Pydantic-AI agent (`app/agent/assistant.py`).
- `app/config.py` loads runtime settings from env vars; `app/server.py` starts uvicorn with those values via `uv run`.
- `app/db.py` initialises a psycopg connection pool on startup for tools to query Postgres safely.
- `scripts/download_data.py` fetches the Torob archive from Google Drive (via `gdown`) and optionally extracts it for ingestion.
- `scripts/ingest.py` provides a reusable CLI (`uv run python -m scripts.ingest`) for Postgres loading, auto-casting all-null columns where needed.
- `start.sh` bootstraps dataset download/ingestion when `TOROB_BOOTSTRAP=1`, then launches the API via `uv run`.
  It also defaults `UV_LINK_MODE=copy` to avoid hardlink warnings inside containers.
- Docker builds copy `pyproject.toml` plus `uv.lock` and run `uv sync --locked` so container deps mirror the locked versions.

## Deployment Notes
- Production hosting uses the Dockerfile-only flow; the container launches the API directly through `uv run python -m app.server`.
- External services such as Postgres are provisioned separately. The app reads connection strings from env vars (or `.env` when local).
- `.env.template` lists required configuration including database URL, OpenAI keys, and replay log directory.
- `.dockerignore` trims build contexts; defaults for `APP_HOST`/`APP_PORT` are baked into the Dockerfile.
- `dev-compose.yaml` exists for local development only—never shipped to production.
- `start.sh` is the container entrypoint; set `TOROB_BOOTSTRAP=1`, `TOROB_DRIVE_ID`, and `TOROB_DATA_DIR` to seed data automatically during deployment.
