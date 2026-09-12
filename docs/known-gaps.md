# 已知缺口与发布前置条件

本文只记录当前代码可证实的边界，不把“已有算法类但尚未接入生产”、
“已接入但缺少目标环境验证”和“完全缺失”混为一谈。优先级按对用户与发布声明的影响排序。

## P0：在外部证据补齐前禁止正式发布结论

- 当前正式 Benchmark 结果仍未封存，`docs/results.md:3` 明确保留
  `primary result is not yet sealed`。这不是零分或失败结论。
- C1 仍需要模型锁、推理环境锁和私有数据 manifest；对应固定检查位于
  `scripts/release_preflight.py:50-53`。随后还需用同一封存配置完成 C2 的
  A/B/C/D、ranker、planner、稳定性、成本子集、P0/ORACLE 与 10,000 次 bootstrap。
- C3 仍需要已授权且不可变的外部数据锁、`formal-portfolio.yaml`、严格
  10/20/10 外部结果，以及 20 个任务、每任务 3 位不同匿名评审的原始评分和 sidecar seal；
  文件前置检查见 `scripts/release_preflight.py:288-327`，内容校验见
  `scripts/release_c_gate.py` 的 `_validate_external` 与 `_validate_human`。
- 缺少上述输入时必须保持 fail-closed；不得生成替代分数、伪造人工评分或把 skipped/blocked
  写成通过。

## P1：生产研究能力仍不完整

- **P2/R2 生产接线缺失。** `P2AdaptivePlanner` 与 `R2EvidenceUtility` 算法类已经存在
  （`src/deepresearch/planning/planners.py:208`、
  `src/deepresearch/evidence/rankers.py:161`），正式实验 runner 也认识这些组件；但服务
  builder 仍固定构造 `FixedPlanner` 与 `SimilarityRanker`
  （`src/deepresearch/runtime/runner_factory.py:873-891`），并在
  `:745-749` 拒绝非 P1/R1 的 `research-v1`。因此这是“算法已实现、生产组合未实现”，
  不能宣称生产自适应规划或 R2 排序已交付。API 默认值暂用 P1/R1，避免默认请求选择
  尚不可执行的组合。
- **定向补搜仍是占位路径。** 生产 claim verification 只路由到
  `RESOLVE_UNSUPPORTED` 或 `FINALIZE`（`src/deepresearch/workflow/research_handlers.py:634-642`）；
  `targeted_research` 明确不生成新查询（`:644-651`）。需要把缺口/冲突转换为受预算约束的
  typed query，并补充审计、停止条件、重放与恢复测试后，才能算真实 directional research。
- **服务级续跑缺失。** Core runner 与 CLI 已支持精确 checkpoint identity 的恢复；但服务
  `RunManager.resume` 在验证 owner、配置和冻结 provider route 后仍无条件抛出
  `CheckpointResumeUnavailable`（`src/deepresearch/runtime/manager.py:368-389`）。需要设计
  非终止 interruption boundary、事件序列占用规则，并在真实 Postgres 重启场景验证后再开放。

## P1：已接入但尚未完成目标环境验证

- Live 模型、Tavily 搜索、固定 peer fetch、HTML/PDF parser 与 embedder 已接入生产 provider
  registry；受控响应测试不等于真实账户可用。授权 smoke 仍需新轮换的 `MODEL_API_KEY`、
  `SEARCH_API_KEY`、32 字节以上 session key，以及完整的 provider/pricing catalogs，并会产生费用。
- Compose/Postgres 配置和离线 HTTP/SSE 验收代码已经存在，但当前 Windows 实现环境没有
  Docker；`docs/deployment.md:187-196` 也明确未声称本机完成容器启动或真实 Postgres 恢复。
  发布前仍需 Linux/目标平台的镜像构建、完整 Compose、卷备份恢复和代理/TLS 验收。

## P2：C4 发布事务的已知边界

- C4 现在会通过正式 summary verifier 重验 sealed group、formal config、完整协议覆盖、ORACLE
  与八项输出 manifest，要求 10,000 次 bootstrap，并只接受明确的 sealed provenance 与事实结论；
  hash-sealed 但不满足 20×3 的 human JSON、空白/待验证 staging 输出、symlink/junction 发布父路径
  都会被拒绝。双重渲染比对通过后，若普通 I/O 失败会尝试回滚本轮已经替换的文件。
  若回滚本身发生 I/O 异常，程序会继续恢复其余文件，并保留
  `.deepresearch-publication-*` 事务目录中的剩余备份供人工恢复，而不会宣称已经完整回滚。
- 多文件发布仍不是跨文件系统崩溃原子事务。进程被强制终止、主机断电或文件系统自身故障
  可能发生在两个替换之间；本轮不作 crash-atomic 声明，也不保证在回滚所依赖的 I/O
  同时失效时自动恢复所有文件。
  正式发布应在版本化工作树中执行，成功后核对 diff/哈希再提交；失败时保留旧提交作为恢复点。
