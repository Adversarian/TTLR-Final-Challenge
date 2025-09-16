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
