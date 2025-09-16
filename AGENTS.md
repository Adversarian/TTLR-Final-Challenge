# Agent Operating Manual

## Non-Negotiable Ground Rules
- Use the Pydantic-AI framework for every LLM interaction; follow its best practices for orchestration.
- Manage Python dependencies exclusively with `uv`; do not shell out to `pip`.
- Package the service with a single Dockerfile that starts the FastAPI API directly (no docker-compose runtime).
- Evolve the assistant scenario by scenario, ensuring new behavior never breaks previously satisfied scenarios.
- Capture complete, replayable judge conversations locally; combine Logfire (from Pydantic-AI) with local log files plus a replay utility.
- Expose an autonomous `/chat` endpoint matching the problem specification; any extra endpoints must remain optional.
- Keep the implementation focused, minimal, and maintainable—no gratuitous abstractions or flair under time pressure.

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
- Tier models by task criticality: use stronger reasoning models sparingly where accuracy gates scenario success; default to cheaper/faster models otherwise.

## Open Questions / To Refine Soon
- Detailed schema for local conversation replay store (format, indexing, tooling).
- Strategy for multimodal (image) handling in scenarios six and seven.
- Data ingestion pipeline for Torob datasets inside Dockerized environment.

## Implementation Notes
- Baseline FastAPI service lives in `app/` with `/chat` handling scenario zero statically; ready for future Pydantic-AI agent integration.
- `app/config.py` loads runtime settings from env vars; `app/server.py` starts uvicorn with those values via `uv run`.

## Deployment Notes
- Target hosting expects a single Dockerfile entrypoint, so the container must launch the API directly through `uvicorn`.
- External services such as PostgreSQL will be provisioned separately; application should read their URLs from environment variables (e.g., via `.env`).
- Maintain a `.env.template` enumerating required variables for runtime configuration.
- Dockerfile installs dependencies with `uv sync`, sets default host/port env vars, and launches the API through `uv run python -m app.server`.
- `.dockerignore` mirrors key git ignores to keep build contexts lean.
