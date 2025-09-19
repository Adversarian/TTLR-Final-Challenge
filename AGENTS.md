# TTLR Assistant Agent Guidelines

## Global Rules
1. **Framework**: All LLM orchestration must go through [Pydantic-AI](https://docs.pydantic.dev/latest/ai/). Do not call LLM APIs directly; wire everything via the framework's agents and tools.
2. **Package Management**: Use `uv` for Python dependency management (`uv pip install`, `uv run`, etc.). Avoid `pip` commands.
3. **Containerization**: The runnable service must be packaged in a Docker image. The Dockerfile must:
   - Install runtime + build dependencies via `uv`.
   - Download competition data with `gdown` inside the image build.
   - Run an ingestion script at the end that loads the parquet payloads into Postgres.
4. **Database**: Assume a PostgreSQL 17 instance is reachable through an environment variable provided at runtime. Do not hardcode credentials.
5. **API Contract**: Implement an HTTP `/chat` endpoint exactly matching the request/response schema from `PROBLEMSTATEMENT.md`. Additional endpoints for internal testing are allowed, but `/chat` must function standalone.
6. **Latency Goal**: Keep p95 latency under 500 ms (hard cap 10 s). Design database queries and agent flow accordingly.
7. **Tooling Constraints**: Each user turn may trigger at most one tool call. Prefer memory over repeated tool usage. Clarify missing info with a single short question.
8. **Logging & Replay**: Capture every conversation turn. Provide a replay mechanism that works locally (e.g., persisted structured logs + a dev tool). Consider Logfire integration but also keep durable local files.
9. **Prompting Principles**: Craft the system prompt from the specification—do not hardcode scenario names. Enforce deterministic behavior (temperature 0) and concise responses. Numeric seller stats must return digits only.
10. **Data Access**: Use Postgres FTS + trigram for product resolution. Do not introduce embeddings or hybrid search in the hot path.
11. **Statistics Queries**: Implement seller stat aggregation with a single SQL query matching the provided CTE structure.
12. **Coding Style**: Use type hints and docstrings where practical. Avoid overly fancy abstractions; keep code maintainable and short.
13. **Testing & Checks**: Run all required linters/tests mentioned in this file after changes. Add new checks here as they become relevant.
14. **Documentation**: Defer README authoring until instructed. Never edit `PROBLEMSTATEMENT.md`.
15. **Env Templates**: Maintain an `.env.template` enumerating required environment variables.

## Logging Notes
- Decide on a structured log format (JSON lines recommended) capturing chat id, turn index, role, content, tool invocations, and timestamps.
- Build a CLI or script to replay stored logs against the agent locally without hitting external services.

## Pending Decisions / TODOs
- Add automated tests validating agent behaviours (sanity guards, tool fan-out, numeric formatting).
- Implement end-to-end scenario fixtures once data access is available.
- Evaluate performance/latency monitoring hooks (Logfire integration optional).

Update this file whenever significant architectural decisions are made.

## Implementation Notes (2025-03-01)
- **Service stack**: FastAPI app in `app/main.py` orchestrates `pydantic_ai.Agent` configured via `app/agent/agent.py`. Runtime dependencies are read from `app/config.py`; database access relies on a shared `asyncpg` pool (`app/db.py`).
- **Pydantic-AI wiring**: Tools live in `app/agent/tools.py` and expose ProductResolve, FeatureLookup, and SellerStats exactly per spec. System prompt is dynamically assembled in `app/agent/prompt.py` so the LLM sees cached base keys.
- **Memory & logging**: `app/memory.py` stores TTL-bound chat state (history + last base key). Structured JSONL logs are emitted through `app/logging.py`; replay utility exists at `scripts/replay.py` supporting both inspection and optional HTTP replays.
- **Data ingestion**: `app/ingestion.py` downloads the parquet bundle with `gdown`, materialises required tables (brands, categories, shops, members, base_products), builds search text/feature flattening, and ensures FTS/trigram + seller indexes. Docker `CMD` runs ingestion before booting uvicorn so the image is self-sufficient.
- **Environment**: `.env.template` lists `DATABASE_URL`, `PRIMARY_MODEL`, and optional logging/data variables. Dependencies are pinned in `pyproject.toml`; Docker installs them with `uv pip install --system .`.