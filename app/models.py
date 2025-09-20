"""Database models for the competition dataset."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, JSON, Numeric, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


class City(Base):
    """Represents a city that hosts one or more shops."""

    __tablename__ = "cities"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)


class Brand(Base):
    """Represents a product brand."""

    __tablename__ = "brands"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)


class Category(Base):
    """Represents a product category in a hierarchical tree."""

    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[int] = mapped_column(
        BigInteger, server_default=text("-1"), nullable=False
    )


class Shop(Base):
    """Represents an online shop within the platform."""

    __tablename__ = "shops"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    city_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("cities.id"), nullable=False)
    score: Mapped[float] = mapped_column(Numeric(2, 1), nullable=False)
    has_warranty: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )


class BaseProduct(Base):
    """Represents a base product shared by multiple shop listings."""

    __tablename__ = "base_products"

    random_key: Mapped[str] = mapped_column(Text, primary_key=True)
    persian_name: Mapped[str] = mapped_column(Text, nullable=False)
    english_name: Mapped[Optional[str]] = mapped_column(Text)
    category_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("categories.id"), nullable=False
    )
    brand_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("brands.id"))
    extra_features: Mapped[dict] = mapped_column(
        JSON, server_default=text("'{}'::json"), nullable=False
    )
    image_url: Mapped[Optional[str]] = mapped_column(Text)

    members: Mapped[List["Member"]] = relationship(back_populates="base")

    __table_args__ = (
        Index("idx_base_products_category", "category_id"),
        Index("idx_base_products_brand", "brand_id"),
    )


class Member(Base):
    """Represents a shop-specific listing of a base product."""

    __tablename__ = "members"

    random_key: Mapped[str] = mapped_column(Text, primary_key=True)
    base_random_key: Mapped[str] = mapped_column(
        Text, ForeignKey("base_products.random_key", ondelete="CASCADE"), nullable=False
    )
    shop_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("shops.id"), nullable=False)
    price: Mapped[int] = mapped_column(BigInteger, nullable=False)

    base: Mapped[BaseProduct] = relationship(back_populates="members")

    __table_args__ = (
        Index("idx_members_base_random_key", "base_random_key"),
        Index("idx_members_shop_id", "shop_id"),
    )


class Search(Base):
    """Represents a logged search request from a user."""

    __tablename__ = "searches"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    uid: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    result_base_product_rks: Mapped[List[str]] = mapped_column(
        JSON, server_default=text("'[]'::json"), nullable=False
    )
    category_id: Mapped[int] = mapped_column(
        BigInteger, server_default=text("0"), nullable=False
    )
    category_brand_boosts: Mapped[list] = mapped_column(
        JSON, server_default=text("'[]'::json"), nullable=False
    )


class BaseView(Base):
    """Represents a base product view triggered from a search result."""

    __tablename__ = "base_views"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    search_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("searches.id", ondelete="CASCADE"), nullable=False
    )
    base_product_rk: Mapped[str] = mapped_column(
        Text, ForeignKey("base_products.random_key", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )


class FinalClick(Base):
    """Represents a final click event on a shop listing."""

    __tablename__ = "final_clicks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    base_view_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("base_views.id", ondelete="CASCADE"), nullable=False
    )
    shop_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("shops.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
