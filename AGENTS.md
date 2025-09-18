# Agent Operating Manual

## Non-Negotiable Ground Rules
- Orchestrate every LLM interaction with LlamaIndex (FunctionAgent, Workflows, Memory, or other first-class primitives) and follow the latest best practices/documentation when integrating tools.
- Manage Python dependencies exclusively with `uv`; never invoke `pip` or edit `requirements.txt`-style files.
- Ship the assistant inside a `compose.yaml` stack: the FastAPI service must build from our Dockerfile and come up with `docker compose up` on Docker Engine.
- Grow capabilities scenario by scenario without regressing behaviors that were already working in earlier scenarios.
- Capture complete judge conversations for replay: use Arize Phoenix traces **and** append-only JSONL logs stored locally under `REPLAY_LOG_DIR`, plus provide tooling to replay them offline.
- Expose a fully functional `/chat` endpoint that matches the problem specification; any additional endpoints are optional helpers and must not be required by judges.
- Keep the implementation lean, well-structured, and easy to maintain—avoid unnecessary abstractions or flourish.

## Reference Notes
- Re-read `PROBLEMSTATEMENT.md` whenever the user reports updates; never edit that file.
- Competition datasets include Searches, Base Views, Final Clicks, Base Products, Members, Shops, Categories, Brands, and Cities tables.
- Evaluation scenarios (0-9) cover sanity checks, direct lookup, attribute Q&A, seller pricing, guided discovery, comparisons, vision tasks, similarity, and ranking. New scenarios build on prior requirements.
- Responses must follow `{"message", "base_random_keys", "member_random_keys"}` with the cardinality limits described in the scenarios.

## Operational Constraints & Tactics
- Budget: assume a 30-second timeout per request and finite API credits—control token usage and prompt length.
- Prefer deterministic Postgres queries (SQL) for catalog retrieval; reserve heavier RAG or embedding flows for cases where structured search falls short.
- Delay emitting `base_random_keys`/`member_random_keys` until confident—judges may stop the conversation on the first non-null array.
- Keep Phoenix instrumentation configurable via environment variables. If Phoenix is unavailable, execution must still succeed while logging locally.
- Maintain `.env`/environment-driven configuration (database URLs, OpenAI keys, Phoenix settings, replay log directory, etc.).

## Data & AI Stack Decisions
- Torob datasets ship as Parquet archives; `scripts/ingest.py` loads them into Postgres (run with `uv run python -m scripts.ingest`).
- SQL is the primary retrieval surface. Introduce pgvector/Qdrant similarity search only when it materially improves recall (e.g., scenarios eight or nine).
- All LLM and embedding calls go through LlamaIndex with provider credentials supplied via env vars (e.g., `OPENAI_API_KEY`).

## Telemetry & Replay Strategy
- Dual logging is mandatory: Arize Phoenix traces plus structured JSONL transcripts per conversation stored in `REPLAY_LOG_DIR`.
- Provide pytest-style replay tooling that can read the JSONL logs and simulate conversations locally.
- Ensure containers mount a persistent volume at `REPLAY_LOG_DIR` so logs survive restarts.

## Implementation Notes
- FastAPI application code lives under `app/`; `/chat` should delegate to a LlamaIndex-powered agent workflow defined there.
- `app/config.py` reads runtime settings (database DSN, Phoenix URL/token, replay log directory, etc.).
- `app/db.py` (or equivalent) manages the Postgres connection pool for tools to execute SQL safely.
- `start.sh` remains the container entrypoint. It may bootstrap datasets when `TOROB_BOOTSTRAP=1` and finally launches the API via `uv run`.
- Docker builds must copy `pyproject.toml` and `uv.lock` then execute `uv sync --locked` so container deps mirror the lockfile.

## Deployment Notes
- `compose.yaml` defines the API service (and any local dev dependencies). Production still uses Docker Engine with this compose stack.
- External services such as Postgres and Phoenix are provisioned separately; the app reads connection strings from environment variables (or `.env` locally).
- `.dockerignore` should keep build contexts minimal. Defaults for `APP_HOST`/`APP_PORT` may live in the Dockerfile or config.
- `start.sh` is the container entrypoint; ensure it respects environment overrides for host/port and logging paths.
- Keep `CODEX_REMOTE_API_URL` (or similar remote validation URLs) up to date before remote testing.
