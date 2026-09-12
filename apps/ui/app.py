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
    CapabilityProfile,
    DemoCapabilities,
    HTTPError,
    HTTPStatusError,
    ReplayExample,
    ResearchApiClient,
    RunView,
    StreamReconnectExhausted,
    TransportError,
)
from apps.ui.replay import ShowcaseSession, live_payload, replay_payload


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
        return f"事件连接已暂停（游标 {error.last_event_id}）。请点击“重新连接事件”。"
    if isinstance(error, HTTPStatusError):
        try:
            raw_body = cast(object, error.response.json())
            body = cast(dict[str, object], raw_body) if isinstance(raw_body, dict) else {}
            candidate = body.get("code")
            nested = body.get("error")
            if candidate is None and isinstance(nested, dict):
                candidate = cast(dict[str, object], nested).get("code")
            code = candidate if isinstance(candidate, str) else None
        except (ValueError, AttributeError):
            code = None
        messages = {
            "PROVIDER_PROFILE_DRIFT": "服务端离线配置不完整。请刷新服务能力或检查录制包。",
            "RESEARCH_GRAPH_UNAVAILABLE": "服务端不支持所选研究流程，请刷新服务能力后重试。",
            "CHECKPOINT_RESUME_UNAVAILABLE": "当前服务版本不能继续该检查点；已有结果仍可查看。",
            "RUN_NOT_FOUND": "当前浏览器会话无权访问该任务，请重新提交。",
            "RATE_LIMITED": "服务容量或用量已达上限，请稍后重试。",
            "DEPLOYMENT_POLICY_VIOLATION": "服务端策略不允许这次请求，请刷新服务能力。",
            "INVALID_REQUEST": "请求被服务端拒绝，请检查研究问题后重试。",
            "REPLAY_MISS": "固定录制与请求不匹配。请使用原始离线示例，或切换到在线 API。",
            "AUTHENTICATION": "服务端配置的 API 密钥被上游拒绝，请在 API 主机修正后重启服务。",
        }
        if isinstance(code, str) and code in messages:
            return messages[code]
        suffix = f"，错误代码 {code}" if isinstance(code, str) else ""
        return f"API 请求失败（HTTP {error.response.status_code}{suffix}）。请重试。"
    if isinstance(error, TransportError):
        return "无法连接 API。请确认服务已启动，然后使用保留的会话重试。"
    return "无法处理 API 响应或请求。请刷新服务能力后重试。"


def _run_status_message(view: RunView) -> str:
    if view.status in {"queued", "running"}:
        return "正在研究，尚未结束"
    explicit_status = {
        "failed": "研究失败",
        "cancelled": "研究已取消",
        "interrupted": "研究已中断",
    }
    if view.status in explicit_status:
        return explicit_status[view.status]
    reasons = {
        "SUFFICIENT": "信息已充分，研究正常完成",
        "PLATEAU": "新增信息不足，已返回现有证据的部分结果",
        "BUDGET_EXHAUSTED": "已达到预算/时间上限",
        "BLOCKED": "研究受阻",
    }
    if view.is_partial:
        if view.stop_reason == "SUFFICIENT":
            return "研究已结束，但结果不完整"
        return reasons.get(view.stop_reason or "", "研究已结束，但结果不完整")
    if view.stop_reason is not None and view.stop_reason != "SUFFICIENT":
        return reasons[view.stop_reason]
    return reasons["SUFFICIENT"]


def _run_error_message(code: str) -> str:
    return {
        "REPLAY_MISS": "固定录制不匹配：请使用原始离线示例，或切换到在线 API。",
        "AUTHENTICATION": "服务端 API 密钥被上游拒绝，请联系 API 主机管理员检查配置。",
        "PROVIDER_NOT_CONFIGURED": "服务端尚未配置在线模型或搜索提供商。",
    }.get(code, "研究任务返回错误；请查看错误代码并重试。")


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
    st.write(f"任务：{view.run_id} · 状态：{view.status} · 持久游标：{session.cursor}")
    status_message = _run_status_message(view)
    if (
        view.status == "completed"
        and not view.is_partial
        and view.stop_reason
        in {
            None,
            "SUFFICIENT",
        }
    ):
        st.success(status_message)
    elif view.status in {"queued", "running"}:
        st.info(status_message)
    elif view.status in {"failed", "cancelled", "interrupted"} or view.stop_reason == "BLOCKED":
        st.error(status_message)
    else:
        st.warning(status_message)
    if view.error_code:
        st.error(_run_error_message(view.error_code))
        with st.expander("错误详情 / Error details"):
            st.code(view.error_code)
    if view.stop_reason is not None:
        st.caption(f"停止原因 / Stop reason: {view.stop_reason}")
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


def _run_panel(session: ShowcaseSession, automatic_at_render: bool = False) -> None:
    if session.run_id is None:
        return
    session.drain()
    session.refresh()
    view = session.view
    if session.poll_error:
        st.error(_error_message(session.poll_error))
    if session.stream_error:
        st.warning(_error_message(session.stream_error))
    if not session.automatic_refresh:
        st.caption("Automatic refresh is paused. Use Retry status to check again.")
    try:
        if st.button("重试状态 / Retry status"):
            session.retry_status()
            st.rerun(scope="app")
        resume, cancel, reconnect = st.columns(3)
        if resume.button("继续 / Resume", disabled=view is None or view.status != "interrupted"):
            view = session.api.resume(session.run_id)
            session.update_view(view)
            session.watch()
            st.rerun(scope="app")
        if cancel.button(
            "取消 / Cancel",
            disabled=view is None or view.status not in {"queued", "running", "interrupted"},
        ):
            view = session.api.cancel(session.run_id)
            session.update_view(view)
            st.rerun(scope="app")
        if reconnect.button("重新连接事件 / Reconnect events", disabled=session.watching):
            session.retry_status()
            session.watch()
            st.rerun(scope="app")
        if view is not None:
            _results(session, view)
        else:
            st.info("Waiting for run status.")
    except (HTTPError, ValueError, TypeError) as error:
        st.error(_error_message(error))
    if automatic_at_render and not session.automatic_refresh:
        # Exactly one terminal transition refreshes the outer form. The former
        # reader-completion rerun raced the final status drain and could abort
        # repeated AppTest/full-page renders before a terminal view appeared.
        st.rerun(scope="app")


def _profile_for(
    capabilities: DemoCapabilities, execution_mode: Literal["replay", "live"]
) -> CapabilityProfile | None:
    candidates = tuple(
        profile for profile in capabilities.profiles if profile.execution_mode == execution_mode
    )
    return next(
        (profile for profile in candidates if profile.available),
        candidates[0] if candidates else None,
    )


def _mode_unavailable_message(mode: Literal["replay", "live"], reason: str) -> str:
    if reason == "DEPLOYMENT_POLICY_VIOLATION":
        return (
            "服务端策略未开放 Demo 用途或匹配的预算（DEPLOYMENT_POLICY_VIOLATION）。"
            "请管理员检查允许的任务用途与预算；固定离线录制需要 medium 预算。"
        )
    if mode == "replay":
        return f"离线配置当前不可用：{reason}。请检查服务端录制配置后刷新。"
    if reason == "PROVIDER_NOT_CONFIGURED":
        return (
            f"在线模式不可用：{reason}。请在 API 主机的 `.env.demo` 配置 "
            "`MODEL_API_KEY` 与 `SEARCH_API_KEY`（Tavily），然后重启 "
            "`python -m scripts.run_demo`。"
        )
    return f"在线配置当前不可用：{reason}。请管理员检查服务端 Provider 配置后刷新。"


def _fixed_replay_summary(example: ReplayExample) -> None:
    st.markdown(f"**固定问题 / Fixed question:** {example.question}")
    st.markdown(
        "**固定参数 / Fixed settings:** "
        f"report={example.report_language}, sources={', '.join(example.source_languages)}, "
        f"budget={example.budget_preset}, seed={example.seed}"
    )
    st.caption("确定性英文示例；不访问在线模型或搜索服务，也不会产生在线费用。")


def main() -> None:
    st.set_page_config(page_title="深度研究演示 / Deep Research Demo", layout="wide")
    st.title("深度研究演示 / Deep Research Demo")
    st.caption("离线固定录制与在线 HTTP API，共用服务端研究流程")
    if "showcase" not in st.session_state or st.session_state["showcase"].closed:
        st.session_state["showcase"] = _session_resource(
            os.environ.get("DEEPRESEARCH_API_URL", "http://127.0.0.1:8000")
        )
    session = cast(ShowcaseSession, st.session_state["showcase"])
    capabilities = session.load_capabilities()
    if st.button("刷新服务能力 / Refresh capabilities"):
        capabilities = session.load_capabilities(force=True)
    if session.capability_error is not None:
        st.error(_error_message(session.capability_error))
        st.caption("服务能力加载失败；不会降级为虚假的成功模式。请修复 API 后点击刷新。")
    if session.pending:
        st.info("提交回复尚未确认；重试会保留原请求与幂等键。")

    if capabilities is not None:
        mode_label = st.radio("运行模式", ("离线示例", "在线 API"), horizontal=True)
        execution_mode: Literal["replay", "live"] = "replay" if mode_label == "离线示例" else "live"
        example = capabilities.replay_example
        profile = (
            (
                capabilities.replay_profile()
                if example is not None
                else _profile_for(capabilities, "replay")
            )
            if execution_mode == "replay"
            else _profile_for(capabilities, execution_mode)
        )
        available = profile is not None and profile.available and bool(capabilities.budget_presets)
        if execution_mode == "replay":
            available = (
                available
                and example is not None
                and example.budget_preset in capabilities.budget_presets
            )
            st.info("离线模式只运行服务端公布的固定录制，不回答任意问题。")
            if example is not None:
                _fixed_replay_summary(example)
            elif profile is None or profile.available:
                st.warning("服务端没有可用的固定离线示例（PROVIDER_PROFILE_DRIFT）。")
            if profile is not None and not profile.available:
                st.warning(
                    _mode_unavailable_message(
                        execution_mode, profile.reason or "PROVIDER_PROFILE_DRIFT"
                    )
                )
        else:
            st.info("在线 API 会调用外部模型与 Tavily 搜索；密钥只配置在 API 主机。")
            if capabilities.unpriced_live:
                st.warning("在线调用可能产生费用；当前服务未提供价格，费用显示为 Unknown（未知）。")
            if not available:
                reason = (
                    "DEPLOYMENT_POLICY_VIOLATION"
                    if not capabilities.budget_presets
                    else (profile.reason if profile is not None else None)
                    or "PROVIDER_NOT_CONFIGURED"
                )
                st.warning(_mode_unavailable_message(execution_mode, reason))

        with st.form("research_request"):
            online_question = ""
            report_language = "en"
            budget = capabilities.budget_presets[0] if capabilities.budget_presets else "medium"
            if execution_mode == "live":
                online_question = st.text_area("研究问题")
                report_language = st.selectbox("报告语言", ("en", "zh"))
                if capabilities.budget_presets:
                    budget = st.selectbox("预算", capabilities.budget_presets)
            submitted = st.form_submit_button(
                "开始研究", disabled=not available or session.automatic_refresh
            )
        if submitted and profile is not None:
            try:
                if execution_mode == "replay":
                    if example is None:
                        raise ValueError("No fixed replay example")
                    payload = replay_payload(
                        example.question,
                        report_language=example.report_language,
                        source_languages=example.source_languages,
                        budget_preset=example.budget_preset,
                        provider_profile_id=profile.profile_id,
                        seed=example.seed,
                        workflow_id=profile.workflow_id,
                    )
                else:
                    payload = live_payload(
                        online_question,
                        report_language=report_language,
                        budget_preset=budget,
                        provider_profile_id=profile.profile_id,
                        workflow_id=profile.workflow_id,
                        planner_id=profile.planner_id,
                        ranker_id=profile.ranker_id,
                    )
                session.submit(payload)
                session.watch()
                st.rerun(scope="app")
            except (HTTPError, ValueError) as error:
                st.error(_error_message(error))
    automatic = session.automatic_refresh
    st.fragment(run_every=1 if automatic else None)(_run_panel)(
        session, automatic_at_render=automatic
    )


if __name__ == "__main__":
    main()
