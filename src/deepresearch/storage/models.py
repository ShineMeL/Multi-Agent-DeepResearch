"""Service-owned schema. LangGraph owns its separate checkpoint schema."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Date, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    pricing_status: Mapped[str] = mapped_column(String)
    pricing_snapshots_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    provider_profile_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    provider_profile_sha256: Mapped[str] = mapped_column(String(64))
    config_sha256: Mapped[str] = mapped_column(String(64))
    owner_scope_sha256: Mapped[str] = mapped_column(String(64))
    idempotency_scope_sha256: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str | None] = mapped_column(String)
    admission_reservation_id: Mapped[str | None] = mapped_column(String)
    admission_attempt_no: Mapped[int | None] = mapped_column(Integer)
    stop_reason: Mapped[str | None] = mapped_column(String)
    is_partial: Mapped[bool] = mapped_column(Boolean)
    report_artifact_id: Mapped[str | None] = mapped_column(String)
    evidence_graph_artifact_id: Mapped[str | None] = mapped_column(String)
    manifest_artifact_id: Mapped[str | None] = mapped_column(String)
    final_usage_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    version: Mapped[int] = mapped_column(Integer)

    __table_args__ = (
        Index(
            "uq_runs_scoped_idempotency",
            "idempotency_scope_sha256",
            "idempotency_key",
            unique=True,
            sqlite_where=idempotency_key.is_not(None),
            postgresql_where=idempotency_key.is_not(None),
        ),
    )


class RunEventRow(Base):
    __tablename__ = "run_events"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class ArtifactRow(Base):
    __tablename__ = "artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    path: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class UsageLedgerRow(Base):
    __tablename__ = "usage_ledger"

    reservation_id: Mapped[str] = mapped_column(String, primary_key=True)
    day: Mapped[date] = mapped_column(Date)
    # Admission intentionally precedes creation of its run.
    run_id: Mapped[str] = mapped_column(String)
    attempt_no: Mapped[int] = mapped_column(Integer)
    # Decimal strings preserve exact values on both SQLite and PostgreSQL.
    amount: Mapped[str] = mapped_column(String)
    state: Mapped[str] = mapped_column(String)

    __table_args__ = (UniqueConstraint("day", "run_id", "attempt_no"),)


class ServiceSchemaVersion(Base):
    __tablename__ = "service_schema_versions"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
