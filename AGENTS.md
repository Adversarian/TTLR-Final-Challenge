# Agent Guidelines

## Current implementation
- FastAPI `/chat` endpoint handles the static sanity checks ("ping", base key echo, member key echo) before delegating to the agent.
- A Pydantic-AI agent instrumented with Logfire resolves catalogue lookups through PostgreSQL fuzzy search and feature extraction tools.
- Feature extraction returns the entire flattened feature list for a base product so the agent can pick the correct attribute without repeated tool calls.
- Seller-focused questions are handled via a `get_seller_statistics` tool that joins members, shops, and cities to compute price, score, warranty, and availability aggregates while feeding a numeric answer back to the HTTP layer.
- The data layer assumes PostgreSQL connection details are provided via the environment variables listed in `.env.template`.
- The monolithic `app/agent.py` module has been decomposed into an `app/agent/` package with dedicated files for dependencies, schemas, tools, prompts, and the agent factory to keep the codebase maintainable.
- The system prompt now teaches the model how to compare multiple candidate products in one turn while still returning a single base key.
- The OpenAI client enables `parallel_tool_calls` so the runtime can execute independent catalogue lookups concurrently when the model requests it.

## Ground rules for new changes
- Keep solutions simple, well-documented, and strongly typed; prefer the minimal implementation that satisfies the competition scenarios without per-scenario branching (scenario 0 may remain hard-coded).
- Use `uv` for dependency management, FastAPI for the HTTP layer, and Pydantic-AI for agent workflows. Avoid introducing conflicting or deprecated libraries.
- When modifying agent behaviour, express shared heuristics in tool descriptions or the system prompt rather than hard-coding logic paths per scenario.
- Seller statistics must continue to populate `numeric_answer` so that the `/chat` endpoint can enforce digit-only responses for competition checks.
- Maintain parity across scenarios when enhancing prompts—new rules should describe general behaviours (e.g., comparisons, confidence handling) instead of referencing specific test IDs.
- Run `uv run pytest` (and any other affected checks) before completing a task to keep the test suite passing.
- Update this file whenever project rules or capabilities change so future tasks inherit accurate guidance.
