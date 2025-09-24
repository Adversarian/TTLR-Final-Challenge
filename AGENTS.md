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
- After handing off any image-only requests, a lightweight router powered by `OPENAI_ROUTER_MODEL` classifies the remaining text into `single_turn` (direct answers) versus `multi_turn` (needs discovery). Single-turn traffic still goes to the legacy agent, while multi-turn traffic now flows through the dedicated scenario-4 manager.
- When the router labels a request as `multi_turn`, control moves to the scenario-4 manager which keeps per-chat memory (filters, asked questions, message history) and orchestrates clarifying turns with a five-response budget.

## Scenario 4 multi-turn workflow
- `MultiTurnManager` caches a `MultiTurnSession` per `chat_id`. Each session tracks processed user messages, filter state, asked/pending questions, previously shown candidates, and persists the constraint extractor's `message_history` via `AgentRunResult.all_messages()`.
- A lightweight extraction agent (`member-constraint-extractor`) converts every new user message into a `ConstraintUpdate` JSON payload. It appends `text_queries`, recognises explicit refusals via `excluded_fields`, and captures shop selections (shop IDs or member keys). The agent runs with `OPENAI_MODEL` at temperature 0 and keeps message payloads compact by serialising the state summary.
- `search_members` executes a single SQL query per turn, joining members, shops, cities, and base products. Textual hints drive a trigram-based similarity score while structured preferences (city, warranty, price range, score thresholds, preferred shops) apply as hard filters. The weighted CTE score blends text relevance, seller warranty, and shop score; candidates are capped to ten and sorted by total score. If tri-gram lookups over `extra_features` become slow we may need to add a dedicated GIN index.
- Question planning starts broad (`broad_intro`) to capture category, price, city, and warranty in one go, then targets the attribute with the highest variation across remaining candidates (city, warranty, price, brand, score, feature hints) while respecting the `excluded_fields` list. When ≤5 candidates remain and haven't been presented before, the manager lists them as `نام محصول — فروشگاه {shop_id}` so the user can pick directly.
- The workflow enforces the 5-turn ceiling: after four assistant replies the manager will pick the highest-scoring candidate, acknowledge the fallback, and return its `member_random_key`. Earlier completion happens whenever the search narrows to a single candidate or the user chooses a presented option.
- Sessions increment their `turns_taken` counter on every assistant message and reuse the constraint agent's message history to stay consistent across HTTP calls. Once a member key is delivered the session holds the result in case the judge resends the prior conversation.

## Database indexes
- `base_products`
  - `idx_base_products_persian_name_trgm` (GIN with `gin_trgm_ops`) accelerates fuzzy name lookups for the `search_base_products` tool.
  - `idx_base_products_english_name_trgm` (GIN with `gin_trgm_ops`) backs English-name fuzzy searches for the same tool.
  - `idx_base_products_search_vector` (GIN on a generated `tsvector`) supports TF/IDF reranking without full table scans.
  - `idx_base_products_category` and `idx_base_products_brand` remain available for potential category/brand filters while exploring catalogue data.
- `members`
  - `idx_members_base_random_key` ensures the seller statistics aggregation can quickly collect offers for a base product.
  - `idx_members_shop_id` keeps lookups by shop efficient for warranty/score joins.

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
- Populate `.env.template` whenever a new environment variable (such as `OPENAI_ROUTER_MODEL`) is introduced so deployments remain reproducible.
