"""Add indexes supporting multi-turn filtering tools."""

from __future__ import annotations

from alembic import op


revision = "20250216_000005"
down_revision = "20250215_000004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    ctx = op.get_context()
    with ctx.autocommit_block():
        op.create_index(
            "idx_base_products_extra_features_gin",
            "base_products",
            ["extra_features"],
            postgresql_using="gin",
            postgresql_concurrently=True,
        )

    with ctx.autocommit_block():
        op.create_index(
            "idx_members_base_price",
            "members",
            ["base_random_key", "price"],
            postgresql_concurrently=True,
        )


def downgrade() -> None:
    ctx = op.get_context()
    with ctx.autocommit_block():
        op.drop_index(
            "idx_members_base_price",
            table_name="members",
            postgresql_concurrently=True,
        )

    with ctx.autocommit_block():
        op.drop_index(
            "idx_base_products_extra_features_gin",
            table_name="base_products",
            postgresql_concurrently=True,
        )
