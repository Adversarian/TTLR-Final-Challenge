# Agent Guidelines

## Current implementation
- FastAPI `/chat` endpoint handles the static sanity checks ("ping", base key echo, member key echo) before delegating to the agent.
- A Pydantic-AI agent instrumented with Logfire resolves catalogue lookups through PostgreSQL fuzzy search and feature extraction tools.
- Fuzzy catalogue search collects trigram-matched candidates and reranks them with PostgreSQL full-text `ts_rank_cd` scoring derived from a generated search vector so TF/IDF precision wins.
- Feature extraction returns the entire flattened feature list for a base product so the agent can pick the correct attribute without repeated tool calls.
- Seller-focused questions are handled via a `get_seller_statistics` tool that joins members, shops, and cities to compute price, score, warranty, and availability aggregates while feeding a numeric answer back to the HTTP layer.
- The data layer assumes PostgreSQL connection details are provided via the environment variables listed in `.env.template`.
- The monolithic `app/agent.py` module has been decomposed into an `app/agent/` package with dedicated files for dependencies, schemas, tools, prompts, and the agent factory to keep the codebase maintainable.
- The system prompt now teaches the model how to compare multiple candidate products in one turn while still returning a single base key.
- The OpenAI client enables `parallel_tool_calls` so the runtime can execute independent catalogue lookups concurrently when the model requests it.
- Base product ingestion now preserves the raw list of member random keys from the dataset in addition to the normalised `members` table for parity with the published schema.
- Prompting updates steer the agent to craft richer product search queries, avoid duplicate tool calls, and finish once confident instead of looping on identical tool invocations.
- Image traffic is routed to a dedicated vision agent that consumes the uploaded BinaryContent directly and answers with a few Persian words describing the dominant object, without invoking catalogue tools.
- The `/chat` endpoint treats the incoming `messages` array as the modalities of a single user turn; the presence of any `image` part triggers the vision agent even if the final element is textual.
- Vision inference reuses the `OPENAI_MODEL` configuration through Pydantic-AI's multimodal support, so no separate vision-specific environment variables are required.
- A lightweight conversation router now runs after vision hand-off to decide whether a text-only turn should follow the default single-response flow or the multi-turn member selector. The `multi_turn` branch now delegates to the dedicated agent described below.
- A dedicated multi-turn agent now owns ambiguous catalogue requests. It persists a compact `TurnState` per `chat_id`, asks at most one focused question per turn, and delegates catalogue lookups to the new `search_members` tool while respecting the two-call relaxation limit.
- Multi-turn state is kept in-process via `TurnStateStore`; tests patch the store to avoid cross-test contamination. When a conversation ends, the state entry is discarded immediately so fresh chats start from turn 1.

## Database indexes
- `base_products`
  - `idx_base_products_persian_name_trgm` (GIN with `gin_trgm_ops`) accelerates fuzzy name lookups for the `search_base_products` tool.
  - `idx_base_products_english_name_trgm` (GIN with `gin_trgm_ops`) backs English-name fuzzy searches for the same tool.
  - `idx_base_products_search_vector` (GIN on a generated `tsvector`) supports TF/IDF reranking without full table scans.
  - `idx_base_products_category` and `idx_base_products_brand` remain available for potential category/brand filters while exploring catalogue data.
- `members`
  - `idx_members_base_random_key` ensures the seller statistics aggregation can quickly collect offers for a base product.
  - `idx_members_shop_id` keeps lookups by shop efficient for warranty/score joins.
- `idx_base_products_extra_features_vector` (GIN on the persisted `extra_features_vector`) ensures the multi-turn `search_members` tool can score feature text without rebuilding `to_tsvector` for every row.
- The `search_members` tool blends the existing trigram and FTS indexes on `base_products` with the numeric filters above while relying on the persisted `extra_features_vector`; pricing buckets are derived dynamically so the query remains a single CTE pipeline.

## Ground rules for new changes
- Keep solutions simple, well-documented, and strongly typed; prefer the minimal implementation that satisfies the competition scenarios without per-scenario branching (scenario 0 may remain hard-coded).
- Use `uv` for dependency management, FastAPI for the HTTP layer, and Pydantic-AI for agent workflows. Avoid introducing conflicting or deprecated libraries.
- When modifying agent behaviour, express shared heuristics in tool descriptions or the system prompt rather than hard-coding logic paths per scenario.
- Seller statistics must continue to populate `numeric_answer` so that the `/chat` endpoint can enforce digit-only responses for competition checks.
- Maintain parity across scenarios when enhancing prompts—new rules should describe general behaviours (e.g., comparisons, confidence handling) instead of referencing specific test IDs.
- When handling images, keep the response to a single concise sentence or noun phrase, default to Persian phrasing, and avoid returning product keys while the vision agent is limited to dominant-object descriptions.
- Run `uv run pytest` (and any other affected checks) before completing a task to keep the test suite passing.
- Update this file whenever project rules or capabilities change so future tasks inherit accurate guidance.
- Keep tool descriptions aligned with their real capabilities (search terms can include distinctive attributes; seller statistics accepts only the base random key with an optional city filter and returns the full aggregate payload).
- Enclose every `agent.run` invocation in the shared `_run_agent_with_retry` helper so retries remain consistent across the API.
- The conversation router is instrumented like other agents, uses `OPENAI_ROUTER_MODEL` (defaulting to `gpt-4.1-mini`), and must respond with the bare labels `single_turn` or `multi_turn`—no rationale is expected from the model.
- Router decisions are cached per `chat_id` via `RouterDecisionStore` so follow-up turns skip reclassification; clear the cache alongside the multi-turn state once a conversation finishes.
- Multi-turn interactions must go through `get_multi_turn_agent` plus `search_members`; always persist and reload `TurnState` via `get_turn_state_store()` instead of relying on transcript replay.
- Configure the multi-turn agent with `OPENAI_MULTI_TURN_MODEL` (default `gpt-4.1-mini`) to keep model selection independent from the single-turn path.
- The multi-turn prompt now requires the agent to open turn one with the product-focused question "What is your price range, and do you have a specific brand in mind? Does your product have any specific features?" and to follow up with the shop-focused question "Please let me know what kind of shop you have in mind, warranty and score all help narrow it down." before presenting numbered options or returning a key.
- The `search_members` ordering prioritises lower prices and higher shop scores when the user has not already imposed price or score constraints, then falls back to relevance-driven ties.
