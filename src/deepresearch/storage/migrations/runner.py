from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib import import_module

from sqlalchemy import insert, select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from deepresearch.storage.models import Base, ServiceSchemaVersion


class ServiceMigrationError(RuntimeError):
    def __init__(self, version: int, cause: Exception) -> None:
        super().__init__(f"service schema migration {version} failed")
        self.version = version
        self.cause = cause


@dataclass(frozen=True)
class ServiceMigration:
    version: int
    upgrade: Callable[[AsyncConnection], Awaitable[None]]


_initial = import_module("deepresearch.storage.migrations.001_initial")
MIGRATIONS = (ServiceMigration(1, _initial.upgrade),)


async def lock_service_transaction(connection: AsyncConnection) -> None:
    """Serialize writes across processes, including first admission of a day.

    Locking existing ledger rows alone cannot lock an empty day. The service
    advisory lock also gives migrations and restart recovery a common order.
    SQLite uses an actual BEGIN so DDL rolls back under legacy driver behavior.
    """
    if connection.dialect.name == "sqlite":
        await connection.execute(text("BEGIN IMMEDIATE"))
    elif connection.dialect.name == "postgresql":
        await connection.execute(text("SELECT pg_advisory_xact_lock(697503211)"))
    else:
        raise ValueError("service storage requires SQLite or PostgreSQL")


async def upgrade_service_schema(engine: AsyncEngine) -> None:
    version = 0
    try:
        async with engine.begin() as connection:
            await lock_service_transaction(connection)
            await connection.run_sync(
                lambda sync: Base.metadata.tables["service_schema_versions"].create(
                    sync, checkfirst=True
                )
            )
            applied = set(
                (
                    await connection.execute(select(ServiceSchemaVersion.version).with_for_update())
                ).scalars()
            )
            versions = [migration.version for migration in MIGRATIONS]
            if len(versions) != len(set(versions)) or any(item < 1 for item in versions):
                raise ValueError("migration versions must be unique positive integers")
            for migration in sorted(MIGRATIONS, key=lambda item: item.version):
                version = migration.version
                if version not in applied:
                    await migration.upgrade(connection)
                    await connection.execute(insert(ServiceSchemaVersion).values(version=version))
    except Exception as error:
        raise ServiceMigrationError(version, error) from error
