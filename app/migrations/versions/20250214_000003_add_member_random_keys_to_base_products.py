"""Add member_random_keys column to base_products."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20250214_000003"
down_revision = "20250210_000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "base_products",
        sa.Column(
            "member_random_keys",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )


def downgrade() -> None:
    op.drop_column("base_products", "member_random_keys")

