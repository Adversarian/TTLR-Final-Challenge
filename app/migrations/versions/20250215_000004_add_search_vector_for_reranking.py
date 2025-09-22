"""Add generated search vector for TF/IDF reranking."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20250215_000004"
down_revision = "20250214_000003"
branch_labels = None
depends_on = None


_SEARCH_VECTOR_EXPRESSION = (
    "to_tsvector('simple', coalesce(persian_name, '') || ' ' || coalesce(english_name, ''))"
)


def upgrade() -> None:
    op.add_column(
        "base_products",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(_SEARCH_VECTOR_EXPRESSION, persisted=True),
            nullable=False,
        ),
    )

    ctx = op.get_context()
    with ctx.autocommit_block():
        op.create_index(
            "idx_base_products_search_vector",
            "base_products",
            ["search_vector"],
            postgresql_using="gin",
            postgresql_concurrently=True,
        )


def downgrade() -> None:
    ctx = op.get_context()
    with ctx.autocommit_block():
        op.drop_index(
            "idx_base_products_search_vector",
            table_name="base_products",
            postgresql_concurrently=True,
        )

    op.drop_column("base_products", "search_vector")
