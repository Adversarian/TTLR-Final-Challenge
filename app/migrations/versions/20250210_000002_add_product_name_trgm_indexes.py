"""Add trigram indexes for base product names."""

from __future__ import annotations

from alembic import op

revision = "20250210_000002"
down_revision = "20250210_000001"
branch_labels = None
depends_on = None


_INDEXES = (
    (
        "idx_base_products_persian_name_trgm",
        "persian_name",
    ),
    (
        "idx_base_products_english_name_trgm",
        "english_name",
    ),
)


def upgrade() -> None:
    ctx = op.get_context()
    with ctx.autocommit_block():
        for name, column in _INDEXES:
            op.create_index(
                name,
                "base_products",
                [column],
                postgresql_using="gin",
                postgresql_ops={column: "gin_trgm_ops"},
                postgresql_concurrently=True,
            )


def downgrade() -> None:
    ctx = op.get_context()
    with ctx.autocommit_block():
        for name, _ in _INDEXES:
            op.drop_index(
                name,
                table_name="base_products",
                postgresql_concurrently=True,
            )
