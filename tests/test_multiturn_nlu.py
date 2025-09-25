"""Unit tests for the multi-turn conversation contracts and NLU helpers."""

from __future__ import annotations

import os

import pytest
from types import SimpleNamespace

DEFAULT_ENV = {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_DB": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
}

for env_key, env_value in DEFAULT_ENV.items():
    os.environ.setdefault(env_key, env_value)


from app.agent.multiturn import (
    ConversationMemory,
    MemberDetails,
    MemberDelta,
    TurnState,
    parse_user_message,
)
from app.agent.multiturn.nlu import (
    LLMExtraction,
    normalize_text,
    reset_nlu_agent_cache,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("قیمت حدود ۳,۵۰۰,۰۰۰ تومان", "قیمت حدود 3,500,000 تومان"),
        ("گارانتی ندارد", "گارانتی ندارد"),
        ("رنگ طلایی و جنس استیل", "رنگ طلایی و جنس استیل"),
    ],
)
def test_normalize_text_converts_digits(raw: str, expected: str) -> None:
    """Persian and Arabic numerals are converted to ASCII digits."""

    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "prompt, expected_min, expected_max",
    [
        (
            "سلام! من دنبال یه کاسه بزرگ برای استفاده در مهمونی‌ها هستم. قیمتش هم حدود 500,000 تا 700,000 تومان باشه.",
            500_000,
            700_000,
        ),
        (
            "سلام! من دنبال یه گیاه بونسای هستم که قیمتش حدوداً بین ۳,۷۰۰,۰۰۰ تا ۴,۱۰۰,۰۰۰ تومان باشه.",
            3_700_000,
            4_100_000,
        ),
        (
            "سلام! من دنبال یک ملحفه کشدار برای تشک دو نفره هستم. ترجیح می‌دم که قیمتش حدود ۱۷۵۰۰۰۰ تومان باشه.",
            1_750_000,
            1_750_000,
        ),
    ],
)
def test_price_extraction(prompt: str, expected_min: int, expected_max: int) -> None:
    """The NLU extracts reasonable price bounds from sample prompts."""

    result = parse_user_message(prompt)
    delta = result.delta
    assert delta.min_price == expected_min
    assert delta.max_price == expected_max
    assert "price" in delta.asked_fields


def test_warranty_and_score_detection() -> None:
    """Warranty preferences and score hints are mapped to structured fields."""

    prompt = (
        "سلام! من دنبال یک ملحفه کشدار هستم. فروشنده امتیاز بالایی داشته باشه و گارانتی داشته باشه."
    )
    result = parse_user_message(prompt)
    delta = result.delta
    assert delta.warranty_required is True
    assert pytest.approx(delta.min_shop_score, rel=0.0, abs=1e-6) == 4.0
    assert "warranty" in delta.asked_fields
    assert "score" in delta.asked_fields


def test_product_attribute_extraction() -> None:
    """Material and colour attributes are captured in product attributes."""

    prompt = (
        "سلام! یک کاسه بزرگ می‌خوام که جنسش استیل باشه و رنگ طلایی داشته باشه و ارسال رایگان هم داشته باشه."
    )
    state = TurnState()
    result = parse_user_message(prompt)
    state.apply_delta(result.delta)
    details = state.details
    assert details.min_price is None and details.max_price is None
    assert details.product_attributes["material"] == "استیل"
    assert details.product_attributes["color"] == "طلایی"
    assert details.product_attributes["shipping"] == "free"
    assert "کاسه" in details.keywords


def test_dont_care_slots_recorded() -> None:
    """Phrases that mark a slot as unimportant populate excluded_fields."""

    prompt = "قیمت مهم نیست اما گارانتی داشته باشه و برند مهم نیست"
    delta = parse_user_message(prompt).delta
    assert delta.warranty_required is True
    assert "price" in delta.excluded_fields
    assert "brand" in delta.excluded_fields


def test_turn_state_merges_deltas() -> None:
    """Applying multiple deltas merges their data deterministically."""

    state = TurnState()
    first_delta = MemberDelta(keywords={"ملحفه", "کشدار"}, min_price=1_500_000, max_price=2_000_000)
    state.apply_delta(first_delta)
    second_prompt = "شهر مهم نیست اما امتیاز بالای ۴ باشه"
    state.apply_delta(parse_user_message(second_prompt).delta)
    details = state.details
    assert details.min_price == 1_500_000
    assert details.max_price == 2_000_000
    assert set(details.keywords) >= {"ملحفه", "کشدار"}
    assert "score" in details.asked_fields
    assert "city" in details.excluded_fields


def test_llm_extraction_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the LLM extractor succeeds its payload seeds the delta."""

    reset_nlu_agent_cache()

    fake_output = LLMExtraction(
        min_price=900_000,
        max_price=1_100_000,
        warranty_required=True,
        keywords=["کاسه", "استیل"],
        asked_fields=["price", "warranty"],
        excluded_fields=["brand"],
        product_attributes={"material": "استیل"},
    )

    class FakeAgent:
        def run_sync(self, message: str, **kwargs: str) -> SimpleNamespace:
            assert "کاسه" in message
            return SimpleNamespace(output=fake_output)

    monkeypatch.setattr("app.agent.multiturn.nlu.get_nlu_agent", lambda: FakeAgent())

    prompt = "سلام! یک کاسه استیل می‌خواهم که حدود یک میلیون تومان باشد. برند مهم نیست."
    result = parse_user_message(prompt)

    delta = result.delta
    assert delta.min_price == 900_000
    assert delta.max_price == 1_100_000
    assert delta.warranty_required is True
    assert "brand" in delta.excluded_fields
    assert delta.product_attributes["material"] == "استیل"
    assert "price" in delta.asked_fields
    assert "کاسه" in delta.keywords

    reset_nlu_agent_cache()


def test_memory_round_trip_compacts_state() -> None:
    """Conversation memory stores compact state snapshots and summaries."""

    state = TurnState()
    state.details = MemberDetails(keywords=["ملحفه"], min_price=1_000_000, max_price=2_000_000)
    state.summary = "توصیف کوتاه"

    memory = ConversationMemory()
    record = memory.remember(
        "chat-1",
        state,
        agent_messages=[
            "پیشنهاد اول: ملحفه کشدار با قیمت ۱۶۵۰۰۰۰",
            "پیشنهاد دوم: گزینه با گارانتی",
        ],
    )
    assert record.summary is not None
    assert len(record.summary) <= 200

    exported = memory.export("chat-1")
    assert exported is not None
    assert exported["state"]["details"]["min_price"] == 1_000_000

    memory.forget("chat-1")
    assert memory.recall("chat-1") is None

    imported = memory.import_state("chat-1", exported)
    assert imported.state.details.min_price == 1_000_000
    assert imported.summary == record.summary
    assert memory.recall("chat-1") is not None
