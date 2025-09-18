# Agent Operating Manual

## Core Ground Rules
- Every LLM interaction must run through LlamaIndex (FunctionAgent/Workflows/Memories) and respect the latest official docs.
- Manage Python dependencies exclusively with `uv`; never shell out to `pip`.
- Ship and run the service via Docker + `compose.yaml`. The FastAPI app lives in a single image and can be started with
  `docker compose up` on any Docker Engine host.
- Grow the assistant scenario-by-scenario. Each new behaviour must keep earlier scenarios (0-3 already in scope) working.
- Capture complete, replayable conversations: stream traces to Arize Phoenix **and** persist JSONL logs locally with a replay utility.
- Expose an autonomous `/chat` endpoint that satisfies the problem spec; auxiliary endpoints are fine but optional.
- Keep the implementation lean, readable, and production-friendly—no flair, no premature abstractions.

## Retrieval & Data Strategy
- Source of truth is Postgres. Enable `pg_trgm`, `fuzzystrmatch`, and `vector` extensions during ingestion.
- Product search relies on a hybrid PGVector store (semantic + lexical). Fall back to trigram SQL only if the vector store is empty.
- Embedding payload = Persian name, English name, and extra features text. Metadata stores random_key, names, brand/category ids, and match type.
- Scenario 1-3 answers come from deterministic SQL over base products, members, shops, and categories (aggregated seller stats + parsed features).
- Keep tool outputs richly structured so the LLM can answer with a single tool call per user turn.

## Agent Behaviour
- Default tool set: a single `lookup_products` FunctionTool returning product context (features + seller stats) for fuzzy queries, base keys, or member keys.
- The agent should call at most **one** tool per turn; design prompts/tools to make extra calls unnecessary.
- Honour scenario-zero sanity checks explicitly (ping/pong and key echoes) before invoking the agent loop.
- Emit `base_random_keys`/`member_random_keys` only when confident—judge stops on the first non-null key list.
- When orchestrating conversations via `FunctionAgent.run`, pass either `chat_history` or `memory` (not both). LlamaIndex applies memory state internally, and providing chat history alongside it will override that memory context (see <https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/>).

## Logging & Replay
- Log directory is driven by `REPLAY_LOG_DIR`. Each `/chat` call appends `{timestamp, request, response}` JSONL entries named by `chat_id`.
- Provide a simple `scripts/replay_chat.py` CLI that replays a stored chat locally (optionally posting back to a running API).
- Mount the replay log directory as a Docker volume so judge conversations persist between restarts.

## Deployment & Ops
- `compose.yaml` defines at least two services: the FastAPI app container and a pgvector-backed Postgres database.
- `start.sh` handles optional dataset download/ingestion and (when enabled) kicks off embedding generation via `scripts.embed_products`.
- `.env.template` must stay up to date with every new required env var (database, OpenAI, Phoenix, logging, bootstrap flags, etc.).
- Always run project scripts with `uv run` to inherit the managed virtual environment.

## Documentation Discipline
- Update `DEVDIARY.md` after notable design or architecture changes.
- Defer `README.md` authoring until the codebase stabilises; keep notes in `AGENTS.md` meanwhile.
- Re-read `PROBLEMSTATEMENT.md` whenever the user signals an update; never edit that file.
