from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from app.agent.multiturn import MemberDetails
from app.agent.multiturn.search import (
    CandidateSearchResult,
    _build_candidate_search_statement,
    search_candidates,
)


@pytest.fixture
def anyio_backend() -> str:
    """Limit asynchronous tests to the asyncio backend for portability."""

    return "asyncio"


class _StubResult:
    """Async execution result returning a pre-defined row."""

    def __init__(self, row: SimpleNamespace) -> None:
        self._row = row

    def one(self) -> SimpleNamespace:
        return self._row


class _StubSession:
    """Fake async session capturing the executed statement."""

    def __init__(self, row: SimpleNamespace) -> None:
        self.row = row
        self.last_statement = None

    async def execute(self, statement):  # pragma: no cover - signature matches SQLAlchemy
        self.last_statement = statement
        return _StubResult(self.row)


@pytest.mark.anyio
async def test_candidate_search_formats_results() -> None:
    """The search helper converts database payloads into ranked candidates."""

    row = SimpleNamespace(
        total_count=7,
        candidates=[
            (
                "mem-1",
                "base-1",
                "کاسه استیل طلایی",
                "پارس",
                "تهران",
                2_500_000,
                4.5,
                True,
                0.87,
            )
        ],
        brand_counts=[("پارس", 3)],
        city_counts=[("تهران ", 4)],
        price_counts=[(3, 5, 2_400_000, 2_600_000)],
        warranty_counts=[(True, 6), (False, 1)],
    )

    session = _StubSession(row)

    details = MemberDetails(
        keywords=["کاسه", "استیل"],
        product_attributes={"color": "طلایی"},
    )

    result = await search_candidates(session, details)

    assert isinstance(result, CandidateSearchResult)
    assert result.count == 7
    assert len(result.candidates) == 1

    candidate = result.candidates[0]
    assert candidate.member_random_key == "mem-1"
    assert candidate.brand_name == "پارس"
    assert candidate.city_name == "تهران"
    assert candidate.price == 2_500_000
    assert candidate.shop_score == pytest.approx(4.5)
    assert candidate.label.startswith("«کاسه استیل طلایی")
    assert "۲,۵۰۰,۰۰۰" not in candidate.label  # Persian digits are not injected automatically
    assert "فروشنده امتیاز 4.5»" in candidate.label

    brand_bucket = result.distributions.brand[0]
    assert brand_bucket.value == "پارس"
    assert brand_bucket.count == 3

    city_bucket = result.distributions.city[0]
    assert city_bucket.value == "تهران"

    price_bucket = result.distributions.price_band[0]
    assert "2,400,000" in price_bucket.value
    assert "2,600,000" in price_bucket.value

    warranty_bucket = result.distributions.warranty[0]
    assert warranty_bucket.value == "گارانتی دارد"
    assert warranty_bucket.count == 6


def test_candidate_statement_respects_filters() -> None:
    """Generated SQL includes the relevant hard filters and skips excluded slots."""

    details = MemberDetails(
        brand_names={"پارس"},
        category_names={"لوازم خانگی"},
        city_names={"تهران"},
        min_price=1_000_000,
        max_price=3_000_000,
        min_shop_score=4.0,
        warranty_required=True,
        excluded_fields={"price"},
        keywords=["کاسه"],
    )

    compiled = _build_candidate_search_statement(details)
    sql = str(compiled.statement.compile(dialect=postgresql.dialect()))
    sql_lower = sql.lower()

    assert "lower(brands.title) in" in sql_lower
    assert "lower(cities.name) in" in sql_lower
    assert "lower(categories.title) in" in sql_lower
    assert "shops.has_warranty is true" in sql_lower
    assert "cast(shops.score as float) >=" in sql_lower
    assert "members.price >=" not in sql_lower  # price slot excluded
    assert "members.price <=" not in sql_lower
