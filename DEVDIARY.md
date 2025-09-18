# Entry #1: Kickoff Alignment
- Reviewed the full problem statement to extract dataset schema, scenario expectations, and response format.
- Logged non-negotiable operating rules and open technical questions in `AGENTS.md` for ongoing reference.
- Identified critical early design concerns: conversation replay tooling, multimodal support, and data ingestion strategy.

# Entry #2: Runtime Strategy Checkpoint
- Clarified latency and budget constraints: target <30s total per request, minimize LLM usage via short prompts and limited tool calls.
- Decided to prioritize SQL-based retrieval over broad RAG; vector search (Qdrant) will only join if similarity matching becomes essential.
- Established cautious emission rule for product/member keys to avoid premature scenario termination in multi-turn flows.
- Added plan to tier LLM models by scenario criticality to balance accuracy with token spend.

# Entry #3: Scenario Zero Scaffold
- Created FastAPI skeleton with `/chat` endpoint returning deterministic outputs for judge sanity checks.
- Added Pydantic models and service layer to make later integration with Pydantic-AI tools straightforward.
- Declared dependencies via `pyproject.toml` for `uv` to manage; testing pending once packages are installed.

# Entry #4: Deployment Constraints Update
- Documented shift from docker-compose to single Dockerfile deployment; container must boot the API directly.
- Noted plan to read external service endpoints (e.g., managed Postgres) from environment variables supplied via `.env`.
- Scheduled creation of `.env.template` to track required configuration fields.

# Entry #5: Docker Bootstrap
- Added lightweight config loader and `app.server` runner to honour env-driven host/port/log level settings.
- Authored Dockerfile using the `uv` package manager to install dependencies and launch the FastAPI app via `uv run`.
- Refreshed docs to capture deployment workflow and environment wiring updates.

# Entry #6: Container Polish
- Added `.dockerignore` to trim build context and keep images lean.
- Baked default `APP_HOST`/`APP_PORT` env vars into the Dockerfile for clarity on deployment defaults.
- Updated agent manual to record these containerization tweaks.

# Entry #7: Scenario One Planning
- Catalog confirmed as Parquet exports; will script ingestion into Postgres while keeping option for auxiliary embedding search.
- Committed to OpenAI-only model stack (LLMs + embeddings) and recorded env requirements.
- Outlined logging duality: telemetry + replayable JSONL transcripts with pytest-based integration replays; production will need a persistent log volume.

# Entry #8: Data & Dev Workflow Planning
- Added `dev-compose.yaml` for local runs (API + Postgres + replay log volume) while keeping production on single Dockerfile.
- Drafted `scripts/ingest.py` to load Parquet exports into Postgres; updated dependencies to include Polars/psycopg/ADBC drivers.
- Expanded `.env.template` with OpenAI model selections, replay log dir, and documented multi-model strategy in agent manual.

# Entry #9: Data Bootstrap Automation
- Created `scripts/download_data.py` to pull the official archive from Google Drive using `gdown` and unpack it for ingestion.
- Confirmed dependency list (plain `polars`, `psycopg`, `gdown`) aligns with `uv` extras support after earlier warning.

# Entry #10: Automated Bootstrap Entrypoint
- Added `start.sh` to orchestrate optional dataset download + ingestion via env toggles before launching the API.
- Updated Dockerfile to use the new entrypoint so deployments stay self-contained.
- Documented new env configuration for bootstrap flags and Drive file ID.

# Entry #11: Dev Compose Sync
- Pointed `dev-compose.yaml` at the new `start.sh` entrypoint and mounted a writable dataset volume.
- Defaulted local compose to `TOROB_BOOTSTRAP=0`, keeping optional seeding behind a flag.

# Entry #12: Ingestion Fixes
- Set `UV_LINK_MODE=copy` in the container entrypoint to silence repeated hardlink warnings from `uv run`.
- Cast all-null Polars columns to text before loading to Postgres to avoid ADBC type mapping errors during bootstrap.

# Entry #13: Scenario One Agentization
- Replaced the static scenario-zero handler with a unified Pydantic-AI agent using OpenAI models and a lookup tool backed by Postgres.
- Added psycopg connection pooling plus SQL tooling that casts null-only columns to keep ingestion stable.
- Prepared groundwork for SQL-grounded responses so the agent can resolve direct product lookup requests (scenario 1) without breaking spec accuracy.

# Entry #14: OpenAI Routing Update
- Added support for custom OpenAI base URLs via `OPENAI_BASE_URL` so traffic can be routed through the competition proxy.

# Entry #15: Reasoning Control
- Wired `OPENAI_REASONING_EFFORT` so we can throttle reasoning token budgets when using a dedicated reasoning model.

# Entry #16: Dependency Lock-in
- Updated Docker build to copy `uv.lock` and run `uv sync --locked` ensuring container installs match the resolved versions.

# Entry #17: Documentation Discipline
- Agreed to consult latest upstream docs for LlamaIndex, psycopg, OpenAI SDKs, etc., before using new APIs to avoid stale assumptions.

# Entry #18: Advanced Search & Telemetry
- Replaced brittle heuristics with an agent-driven lookup pipeline layered over deterministic filters, full-text search, trigram similarity, and pgvector semantic retrieval.
- Instrumented FastAPI, psycopg, OpenAI, and LlamaIndex via Arize Phoenix so traces flow to our shared observability stack.
- Introduced the `product_embeddings` table and an embedding ingestion script so pgvector can serve semantic matches efficiently.

# Entry #19: Remote Testing Pivot
- Dropped local compose files; testing now targets the remote dev environment exposed through `CODEX_REMOTE_API_URL`.
- Documented new env variables (`PHOENIX_SERVER_URL`, `CODEX_REMOTE_API_URL`) in templates and manuals to keep deployments aligned.

# Entry #20: Scenario 1-3 Alignment & Ops Refresh
- Re-baselined requirements with new ground rules (compose-first deployment, single-tool agent, replayable logs) and rewrote `AGENTS.md` accordingly.
- Planned enriched Postgres hybrid search tool that hydrates features + seller stats in one call to satisfy scenarios 1-3 without extra tool hops.
- Scheduled compose stack (app + pgvector Postgres) and JSONL replay logging utility to keep local debugging aligned with judge conversations.

# Entry #21: Lookup Tool Hardening
- Updated `lookup_products` so the tool validates raw dict payloads via `ProductLookupArgs.model_validate` before clamping limits, preventing attribute errors during agent calls.
- Documented the tool parameters in a Google-style docstring to improve function selection while keeping validation centralized.
- Removed `compose.yaml` in favour of the Dockerfile-only deployment path now required by operations.
