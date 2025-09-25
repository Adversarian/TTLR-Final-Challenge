"""Materialise extra features full-text vector."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20250215_000005"
down_revision = "20250215_000004"
branch_labels = None
depends_on = None


_EXTRA_FEATURES_VECTOR_EXPRESSION = "to_tsvector('simple', coalesce(extra_features::text, ''))"


def upgrade() -> None:
    op.add_column(
        "base_products",
        sa.Column(
            "extra_features_vector",
            postgresql.TSVECTOR(),
            sa.Computed(_EXTRA_FEATURES_VECTOR_EXPRESSION, persisted=True),
            nullable=False,
        ),
    )

    ctx = op.get_context()
    with ctx.autocommit_block():
        op.create_index(
            "idx_base_products_extra_features_vector",
            "base_products",
            ["extra_features_vector"],
            postgresql_using="gin",
            postgresql_concurrently=True,
        )


def downgrade() -> None:
    ctx = op.get_context()
    with ctx.autocommit_block():
        op.drop_index(
            "idx_base_products_extra_features_vector",
            table_name="base_products",
            postgresql_concurrently=True,
        )

    op.drop_column("base_products", "extra_features_vector")
