from decimal import Decimal

import pytest

from deepresearch.runtime.admission import NoOpAdmissionController


@pytest.mark.asyncio
async def test_noop_admission_makes_manager_dependency_available_before_limits() -> None:
    admission = await NoOpAdmissionController().admit(
        run_id="r1",
        client_ip="127.0.0.1",
        session_id="local",
        access_profile="local",
        requested_cost_usd=Decimal(0),
    )

    assert admission.reservation_id is None
    assert admission.attempt_no is None
