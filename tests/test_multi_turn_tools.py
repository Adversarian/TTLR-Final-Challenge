from decimal import Decimal

import pytest

from app.agent.multi_turn.tools import _match_feature, _score_member_offer


def test_match_feature_uses_name_fallback():
    flattened = []
    matched, note = _match_feature(
        flattened,
        required_name="install_method",
        required_value="سقفی",
        match="contains",
        fallback_texts=["لوستر سقفی نقره ای مدل کلاسیک"],
    )

    assert matched is True
    assert note and "سقفی" in note


@pytest.mark.parametrize(
    "price, expected_score",
    [
        (550000, 0.95),
        (800000, 0.65),
    ],
)
def test_score_member_offer_ranks_matches(price: int, expected_score: float) -> None:
    matched, score = _score_member_offer(
        price=price,
        has_warranty=True,
        shop_score=Decimal("4.5"),
        city_name="تهران",
        price_min=500000,
        price_max=600000,
        require_warranty=True,
        min_shop_score=4.0,
        city="تهران",
        dismissed=set(),
    )

    assert "گارانتی" in " ".join(matched)
    assert score >= expected_score


def test_score_member_offer_respects_dismissed_aspects() -> None:
    matched, score = _score_member_offer(
        price=450000,
        has_warranty=False,
        shop_score=None,
        city_name=None,
        price_min=400000,
        price_max=500000,
        require_warranty=None,
        min_shop_score=None,
        city=None,
        dismissed={"price"},
    )

    assert all("قیمت" not in note for note in matched)
    assert score == pytest.approx(0.1, rel=0, abs=1e-6)
