"""Add trigram index for base_products.extra_features text search."""

from __future__ import annotations

from alembic import op


revision = "20250216_000005"
down_revision = "20250215_000004"
branch_labels = None
depends_on = None


_INDEX_NAME = "idx_base_products_extra_features_trgm"


def upgrade() -> None:
    ctx = op.get_context()
    with ctx.autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
            f"{_INDEX_NAME} ON base_products USING gin "
            "((extra_features::text) gin_trgm_ops)"
        )


def downgrade() -> None:
    ctx = op.get_context()
    with ctx.autocommit_block():
        op.execute(
            f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX_NAME}"
        )

