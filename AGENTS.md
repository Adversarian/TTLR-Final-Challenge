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
- Scenario 4 (multi-turn) requests are now delegated to `Scenario4Coordinator`, which manages constraint extraction, clarification planning, catalogue filtering, member resolution, and finalisation through a graph of specialised Pydantic-AI agents. The coordinator persists conversation state per `chat_id`, enforces the five-turn limit, and always returns exactly one member key by the final turn.
- Constraint extraction records a `dismissed_aspects` list (brand, warranty, shop_score, city, price, features) whenever the customer says a topic does not matter so clarification and fallback prompts avoid repeating the same questions.
- Forced finalisation on the fifth turn now relaxes only the dimensions the user marked as optional, falls back to deterministic SQL lookups when the resolver agent cannot find a shop, and records any unmet expectations so the final message can acknowledge them.
- Multi-turn tools include `category_feature_statistics`, `filter_base_products_by_constraints`, and `filter_members_by_constraints`, covering feature discovery, catalogue narrowing, and seller resolution so the coordinator can converge on a single member within the turn budget. The product filter now scores rather than rejects candidates when "required" features are missing, boosting items that satisfy more constraints while still returning options to discuss. Name tokens supplement `extra_features` so attributes that only appear in the base-product title are recognised. The member filter ranks offers by a `match_score`, annotates which constraints they satisfy, and never hard-fails on price, city, warranty, or score so the dialogue can always present fallback sellers unless the catalogue is truly empty.
- The dedicated scenario 4 agents are configured with Logfire instrumentation (via `_ensure_logfire` and `InstrumentationSettings`) so their traces match the single-turn and vision agents.

## Database indexes
- `base_products`
  - `idx_base_products_persian_name_trgm` (GIN with `gin_trgm_ops`) accelerates fuzzy name lookups for the `search_base_products` tool.
  - `idx_base_products_english_name_trgm` (GIN with `gin_trgm_ops`) backs English-name fuzzy searches for the same tool.
  - `idx_base_products_search_vector` (GIN on a generated `tsvector`) supports TF/IDF reranking without full table scans.
  - `idx_base_products_category` and `idx_base_products_brand` remain available for potential category/brand filters while exploring catalogue data.
  - `idx_base_products_extra_features_gin` (GIN on the `jsonb` `extra_features` column) supports containment and statistics queries for scenario 4 feature analysis.
- `members`
  - `idx_members_base_random_key` ensures the seller statistics aggregation can quickly collect offers for a base product.
  - `idx_members_shop_id` keeps lookups by shop efficient for warranty/score joins.
  - `idx_members_base_price` (B-tree on `(base_random_key, price)`) accelerates price range filters when resolving member offers.
- Alembic revision `20250216_000005_add_indexes_for_multi_turn_filters` creates the extra-feature and member-price indexes that power the new scenario 4 filtering tools; apply migrations after pulling.

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
- Enclose every `agent.run` invocation in the shared `_run_agent_with_retry` helper so retries remain consistent across the API. Scenario 4's coordinator uses dedicated helpers but still relies on `result.new_messages()` to extend each agent's chat history after every call.
- A lightweight router in `app/agent/router.py` classifies textual requests as `single_turn` or `multi_turn` using `OPENAI_ROUTER_MODEL`. After vision handling the `/chat` endpoint now routes multi-turn traffic to `get_scenario4_coordinator()`; single-turn requests continue to use the legacy assistant.
