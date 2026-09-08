"""Recover durable usage lower bounds without pretending a partial bill is final."""

from collections.abc import Sequence

from deepresearch.domain import ResourceUsage, RunEvent


def recover_usage(
    final_usage: ResourceUsage | None,
    events: Sequence[RunEvent],
    *,
    never_started: bool = False,
) -> ResourceUsage:
    # Event usage is incremental. A recovered prefix cannot prove that the last
    # in-flight provider call was free, even when every persisted delta is priced.
    base = final_usage or ResourceUsage.zero(cost_known=never_started and not events)
    fields = (
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "search_calls",
        "pages",
        "retries",
        "wall_seconds",
    )
    values = {
        name: max(getattr(base, name), sum(getattr(event.usage_delta, name) for event in events))
        for name in fields
    }
    values["total_tokens"] = sum(values[name] for name in fields[:3])
    return base.model_copy(update=values)
