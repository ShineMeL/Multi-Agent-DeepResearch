"""Run with: python -m streamlit run apps/ui/app.py.

Set DEEPRESEARCH_API_URL to the hosted Task 8 API. Replay profiles and bundles
are configured by that API; this page only exchanges public HTTP payloads.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any, Literal, cast

import streamlit as st

from apps.ui.api_client import (
    ArtifactKind,
    HTTPError,
    HTTPStatusError,
    ResearchApiClient,
    RunView,
    StreamReconnectExhausted,
    TransportError,
)
from apps.ui.replay import ShowcaseSession, replay_payload


def _release_session(session: ShowcaseSession) -> None:
    session.close()


@st.cache_resource(scope="session", on_release=_release_session)
def _session_resource(base_url: str) -> ShowcaseSession:
    """Streamlit releases this resource on browser disconnect/session shutdown."""
    return ShowcaseSession(ResearchApiClient(base_url))


def _table(rows: list[dict[str, Any]]) -> None:
    # Streamlit's overload includes optional dependency types without stubs.
    st.dataframe(rows, hide_index=True)  # pyright: ignore[reportUnknownMemberType]


def _error_message(error: Exception) -> str:
    if isinstance(error, StreamReconnectExhausted):
        return f"Event connection paused at sequence {error.last_event_id}. Reconnect to continue."
    if isinstance(error, HTTPStatusError):
        try:
            code = error.response.json().get("code")
        except (ValueError, AttributeError):
            code = None
        messages = {
            "PROVIDER_PROFILE_DRIFT": "The API replay profile is unavailable or incomplete. "
            "Configure a complete replay catalog and matching bundle on the API server.",
            "RESEARCH_GRAPH_UNAVAILABLE": "Production research-v1 is unavailable.",
            "CHECKPOINT_RESUME_UNAVAILABLE": "Checkpoint continuation is unavailable in this "
            "service version. The interrupted run and its artifacts remain available.",
            "RUN_NOT_FOUND": "Run unavailable in this browser session.",
            "RATE_LIMITED": "Service capacity or usage limit reached. Retry later.",
            "DEPLOYMENT_POLICY_VIOLATION": "The API policy does not permit this replay request.",
            "INVALID_REQUEST": "The API rejected the request. Check the replay inputs.",
        }
        return messages.get(
            code if isinstance(code, str) else "",
            f"API request failed (HTTP {error.response.status_code}).",
        )
    if isinstance(error, TransportError):
        return "Cannot reach the API. Retry using the retained session and request."
    return "The API response or request could not be processed."


def _download(
    session: ShowcaseSession, kind: ArtifactKind, artifact_id: str | None
) -> bytes | None:
    if artifact_id is None or session.run_id is None:
        return None
    key = (kind, artifact_id)
    if key not in session.downloads:
        session.downloads[key] = session.api.download_artifact(session.run_id, kind)
    return session.downloads[key]


def _json_object(data: bytes | None) -> dict[str, Any]:
    if data is None:
        return {}
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return cast(dict[str, Any], value)


def _details(session: ShowcaseSession, manifest: dict[str, Any]) -> None:
    payloads = [event.model_dump(mode="json")["public_payload"] for event in session.events]
    for title, names in (
        ("Plan / subquestions", ("plan", "subquestions")),
        ("Screening reasons", ("screening_reasons", "ranking_decisions")),
        ("Coverage", ("coverage", "coverage_ledger")),
    ):
        with st.expander(title):
            values = [payload[name] for payload in payloads for name in names if name in payload]
            if values:
                for value in values:
                    st.json(value)
            else:
                st.info(f"{title} details are not exposed by this run's public API data.")
    with st.expander("Queries", expanded=True):
        queries = [
            {"query": call.get("normalized_query"), "outcome": call.get("outcome")}
            for call in manifest.get("provider_calls", [])
            if call.get("operation") == "search"
        ]
        if queries:
            _table(queries)
        else:
            st.info("Queries are available when the API publishes the run manifest.")


def _results(session: ShowcaseSession, view: RunView) -> None:
    st.write(f"Run: {view.run_id} · Status: {view.status} · Durable cursor: {session.cursor}")
    if view.is_partial:
        st.warning("Partial result")
    if view.error_code:
        st.write(f"Service error code: {view.error_code}")
    st.write(f"Stop reason: {view.stop_reason or 'Not reported'}")
    usage = view.final_usage
    token_count = usage.total_tokens if usage else None
    wall_seconds = usage.wall_seconds if usage else None
    cost_usd = usage.cost_usd if usage else None
    if view.status in {"queued", "running"}:
        deltas = [event.usage_delta for event in session.events]
        token_count = sum(delta.total_tokens for delta in deltas)
        wall_seconds = sum(delta.wall_seconds for delta in deltas)
        known_costs = [delta.cost_usd for delta in deltas if delta.cost_usd is not None]
        cost_usd = (
            sum(known_costs, Decimal(0)) if deltas and len(known_costs) == len(deltas) else None
        )
    tokens, elapsed, cost = st.columns(3)
    tokens.metric("Tokens", token_count if token_count is not None else "Unknown")
    elapsed.metric(
        "Time (seconds)", f"{wall_seconds:.2f}" if wall_seconds is not None else "Unknown"
    )
    cost.metric("Cost (USD)", str(cost_usd) if cost_usd is not None else "Unknown")
    with st.expander("Durable event timeline", expanded=True):
        _table(
            [
                {
                    "seq": event.seq,
                    "node": event.node,
                    "kind": event.kind,
                    "status": event.status,
                    "tokens": event.usage_delta.total_tokens,
                    "seconds": event.usage_delta.wall_seconds,
                }
                for event in session.events
            ]
        )

    artifacts: dict[ArtifactKind, bytes | None] = {}
    for kind, identifier, filename, mime in (
        ("report", view.report_artifact_id, "report.md", "text/markdown"),
        ("evidence", view.evidence_graph_artifact_id, "evidence.json", "application/json"),
        ("manifest", view.manifest_artifact_id, "manifest.json", "application/json"),
    ):
        typed_kind = cast(ArtifactKind, kind)
        try:
            data = _download(session, typed_kind, identifier)
            artifacts[typed_kind] = data
            if data is not None:
                st.download_button(
                    f"Download {kind}",
                    data=data,
                    file_name=filename,
                    mime=mime,
                    key=f"download_{kind}",
                )
        except (HTTPError, ValueError) as error:
            st.warning(_error_message(error))
    manifest = _json_object(artifacts.get("manifest"))
    _details(session, manifest)
    st.subheader("Evidence")
    evidence = _json_object(artifacts.get("evidence"))
    if evidence:
        _table(evidence.get("evidence", []))
        st.caption("Evidence-to-source links")
        edges = [
            f"{json.dumps('evidence: ' + str(row.get('evidence_id')))} -> "
            f"{json.dumps('source: ' + str(row.get('source_id')))};"
            for row in evidence.get("evidence", [])
        ]
        # Streamlit also types an optional, unstubbed graphviz object overload.
        st.graphviz_chart(  # pyright: ignore[reportUnknownMemberType]
            "digraph Evidence { rankdir=LR; " + " ".join(edges) + " }"
        )
        with st.expander("Sources"):
            _table(evidence.get("sources", []))
    else:
        st.info("Evidence will appear when its API artifact is available.")
    st.subheader("Report")
    report = artifacts.get("report")
    if report is not None:
        st.markdown(report.decode("utf-8"), unsafe_allow_html=False)
    else:
        st.info("Report is not yet available.")


def _run_panel(
    session: ShowcaseSession, watching_at_render: bool, automatic_at_render: bool = False
) -> None:
    if session.run_id is None:
        return
    session.drain()
    if watching_at_render and not session.watching:
        # Refresh the form outside this fragment when a reader finishes, so a
        # completed run does not leave Start replay disabled indefinitely.
        st.rerun()
    session.refresh()
    view = session.view
    if session.poll_error:
        st.error(_error_message(session.poll_error))
    if session.stream_error:
        st.warning(_error_message(session.stream_error))
    if not session.automatic_refresh:
        st.caption("Automatic refresh is paused. Use Retry status to check again.")
    try:
        if st.button("Retry status"):
            session.retry_status()
            st.rerun()
        resume, cancel, reconnect = st.columns(3)
        if resume.button("Resume", disabled=view is None or view.status != "interrupted"):
            view = session.api.resume(session.run_id)
            session.update_view(view)
            session.watch()
            st.rerun()
        if cancel.button(
            "Cancel",
            disabled=view is None or view.status not in {"queued", "running", "interrupted"},
        ):
            view = session.api.cancel(session.run_id)
            session.update_view(view)
            st.rerun()
        if reconnect.button("Reconnect events", disabled=session.watching):
            session.retry_status()
            session.watch()
            st.rerun()
        if view is not None:
            _results(session, view)
        else:
            st.info("Waiting for run status.")
    except (HTTPError, ValueError, TypeError) as error:
        st.error(_error_message(error))
    if automatic_at_render and not session.automatic_refresh:
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Research Replay Showcase", layout="wide")
    st.title("Research Replay Showcase")
    st.caption("Baseline replay · P1 fixed planner · R1 similarity ranking")
    st.warning(
        "Production research-v1 is unavailable. This showcase submits baseline-v1 replay runs."
    )
    st.info(
        "A complete replay profile and matching bundle must be configured on the API server. "
        "The default replay-default catalog entry is empty. Replay inputs must match the bundle."
    )
    if "showcase" not in st.session_state or st.session_state["showcase"].closed:
        st.session_state["showcase"] = _session_resource(
            os.environ.get("DEEPRESEARCH_API_URL", "http://127.0.0.1:8000")
        )
    session = cast(ShowcaseSession, st.session_state["showcase"])
    if session.pending:
        st.info("A submission reply is pending. Start replay retries the original request and key.")
    with st.form("replay_request"):
        question = st.text_area("Question recorded in the replay bundle", key="question")
        language = st.selectbox("Report language", ["en", "zh"])
        source_language = st.selectbox("Source language", ["en", "zh"])
        budget = st.selectbox("Budget", ["low", "medium"])
        profile = st.text_input(
            "API replay profile",
            value=os.environ.get("DEEPRESEARCH_REPLAY_PROFILE_ID", "replay-default"),
        )
        seed = st.number_input("Seed", min_value=0, value=0, step=1)
        submitted = st.form_submit_button("Start replay", disabled=session.watching)
    if submitted:
        try:
            payload = replay_payload(
                question,
                report_language=language,
                source_languages=(source_language,),
                budget_preset=cast(Literal["low", "medium"], budget),
                provider_profile_id=profile,
                seed=int(seed),
            )
            session.submit(payload)
            session.watch()
            st.rerun()
        except (HTTPError, ValueError) as error:
            st.error(_error_message(error))
    automatic = session.automatic_refresh
    st.fragment(run_every=1 if automatic else None)(_run_panel)(
        session, watching_at_render=session.watching, automatic_at_render=automatic
    )


if __name__ == "__main__":
    main()
