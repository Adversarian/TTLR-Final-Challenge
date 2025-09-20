"""Initial database schema for competition tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20250210_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cities",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
    )

    op.create_table(
        "brands",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
    )

    op.create_table(
        "categories",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("parent_id", sa.BigInteger(), nullable=False, server_default=sa.text("-1")),
    )

    op.create_table(
        "shops",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("city_id", sa.BigInteger(), nullable=False),
        sa.Column("score", sa.Numeric(2, 1), nullable=False),
        sa.Column(
            "has_warranty",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.ForeignKeyConstraint(["city_id"], ["cities.id"]),
    )

    op.create_table(
        "base_products",
        sa.Column("random_key", sa.Text(), primary_key=True),
        sa.Column("persian_name", sa.Text(), nullable=False),
        sa.Column("english_name", sa.Text()),
        sa.Column("category_id", sa.BigInteger(), nullable=False),
        sa.Column("brand_id", sa.BigInteger()),
        sa.Column(
            "extra_features",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
        sa.Column("image_url", sa.Text()),
        sa.ForeignKeyConstraint(["category_id"], ["categories.id"]),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"]),
    )

    op.create_index(
        "idx_base_products_category",
        "base_products",
        ["category_id"],
    )
    op.create_index(
        "idx_base_products_brand",
        "base_products",
        ["brand_id"],
    )

    op.create_table(
        "members",
        sa.Column("random_key", sa.Text(), primary_key=True),
        sa.Column("base_random_key", sa.Text(), nullable=False),
        sa.Column("shop_id", sa.BigInteger(), nullable=False),
        sa.Column("price", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["base_random_key"], ["base_products.random_key"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["shop_id"], ["shops.id"]),
    )

    op.create_index(
        "idx_members_base_random_key",
        "members",
        ["base_random_key"],
    )
    op.create_index(
        "idx_members_shop_id",
        "members",
        ["shop_id"],
    )

    op.create_table(
        "searches",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("uid", sa.Text(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column(
            "result_base_product_rks",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
        sa.Column(
            "category_id",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "category_brand_boosts",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )

    op.create_table(
        "base_views",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("search_id", sa.BigInteger(), nullable=False),
        sa.Column("base_product_rk", sa.Text(), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["search_id"], ["searches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["base_product_rk"], ["base_products.random_key"], ondelete="CASCADE"
        ),
    )

    op.create_table(
        "final_clicks",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("base_view_id", sa.BigInteger(), nullable=False),
        sa.Column("shop_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["base_view_id"], ["base_views.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["shop_id"], ["shops.id"]),
    )


def downgrade() -> None:
    op.drop_table("final_clicks")
    op.drop_table("base_views")
    op.drop_table("searches")
    op.drop_index("idx_members_shop_id", table_name="members")
    op.drop_index("idx_members_base_random_key", table_name="members")
    op.drop_table("members")
    op.drop_index("idx_base_products_brand", table_name="base_products")
    op.drop_index("idx_base_products_category", table_name="base_products")
    op.drop_table("base_products")
    op.drop_table("shops")
    op.drop_table("categories")
    op.drop_table("brands")
    op.drop_table("cities")
