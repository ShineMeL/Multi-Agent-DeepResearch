from sqlalchemy.ext.asyncio import AsyncConnection

from deepresearch.storage.models import Base

VERSION = 1


async def upgrade(connection: AsyncConnection) -> None:
    await connection.run_sync(Base.metadata.create_all)
