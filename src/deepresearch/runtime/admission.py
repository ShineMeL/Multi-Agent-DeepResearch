from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True)
class Admission:
    reservation_id: str | None
    attempt_no: int | None


class AdmissionController(Protocol):
    async def admit(
        self,
        *,
        run_id: str,
        client_ip: str,
        session_id: str,
        access_profile: str,
        requested_cost_usd: Decimal,
    ) -> Admission: ...

    async def settle(self, reservation_id: str | None, actual_cost_usd: Decimal) -> None: ...

    async def release(self, reservation_id: str | None) -> None: ...

    async def defer_settlement(self, reservation_id: str) -> None: ...


class NoOpAdmissionController:
    async def admit(
        self,
        *,
        run_id: str,
        client_ip: str,
        session_id: str,
        access_profile: str,
        requested_cost_usd: Decimal,
    ) -> Admission:
        return Admission(reservation_id=None, attempt_no=None)

    async def settle(self, reservation_id: str | None, actual_cost_usd: Decimal) -> None:
        return None

    async def release(self, reservation_id: str | None) -> None:
        return None

    async def defer_settlement(self, reservation_id: str) -> None:
        return None


__all__ = ["Admission", "AdmissionController", "NoOpAdmissionController"]
