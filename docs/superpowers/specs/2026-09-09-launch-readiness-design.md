# Multi-Agent DeepResearch 最终上线收口设计

日期：2026-09-09  
状态：待评审  
基线：`feature/service-demo-deployment` / `39877cd`

## 1. 目标与发布边界

本设计把“最终可以上线”拆成三个可独立验收的发布层级。每一层都有
自己的运行模式、证据要求和阻塞条件；后一层不得用前一层的结果冒充。

### Release A：可复现 Demo（本次优先交付）

Release A 是一个可公开展示的、默认离线的 Replay Showcase：

- API 使用已有的 FastAPI 生命周期、持久化 Store、SSE durable cursor、权限隔离、
  脱敏和额度控制。
- UI 只通过 HTTP/SSE 访问 API，不导入 provider、数据库或 checkpoint 实现。
- `baseline-v1` / `P1` / `R1` 通过已校验 replay bundle 运行；服务端从 bundle
  snapshot 派生 `replay_parent`，不接受客户端提供父级审计身份。
- Replay 输入必须匹配录制请求；未知问题返回明确的 Replay miss，不自动转 Live。
- 发布页保留“Benchmark 尚未封存”的事实，不生成任何虚构质量、成本或置信区间。

Release A 的上线定义是“可审计的公开 Demo”，不是“已完成研究论文结论”。

### Release B：Kimi Live Demo（可选的第二阶段）

Release B 在 Release A 的服务边界上增加显式 opt-in 的 Live profile：

- Kimi 通过现有 `OpenAICompatibleModelProvider` 的 `/chat/completions` 接口作为
  模型 provider。
- API key 只从服务端 `MODEL_API_KEY` 环境变量或 secret manager 注入；catalog、
  request、manifest、浏览器和日志中不出现 key 的值。
- 搜索先复用已存在的 Tavily/Serper `SearchProvider`。Kimi 原生 `$web_search`
  使用另一套 tool-call/加密结果协议，在有专门 adapter、证据映射和回放测试前
  不启用。
- Live profile 必须同时提供完整 route catalog、搜索凭证、embedding 配置和
  所需 `PricingSnapshot`；缺任一项在 provider 调用前拒绝。
- 默认部署仍是 Replay；Live 通过单独的 profile、环境变量和额度上限启用，不能
  被用户请求中的 `access_profile` 或 `execution_mode` 绕过。

### Release C：完整研究产品与正式 Benchmark（资源满足后）

Release C 才把 `research-v1` 作为生产 workflow 暴露，并执行正式评测：

- 在服务 runner factory 中构造真实的 `ResearchGraphDependencies`，接通
  Planner P0/P1/P2、Evidence Ranker R0/R1/R2、claim extraction/judging、一次
  targeted research、typed unsupported-claim resolution、Citation Guard 和
  `PersistResults`。
- A/B/C/D、P0、预算敏感性、seed/repeat、10,000 次分层 paired bootstrap 只从
  sealed formal config 派生，结果通过现有 evaluator 和 renderer 发布。
- `docs/results.md` 只有在 public summary、artifact manifest、code/config/model/
  environment hashes 全部验证后才从占位状态更新。
- Portfolio 外部 10/20/10 与三人盲评必须保持独立 seal，不并入主实验置信区间。

Release C 的代码可以先在离线 fixture 上接通并验证，但正式数字需要真实且授权的
数据、模型/运行环境锁、运行资源和人工评审；缺少这些输入时必须保持 fail-closed。

## 2. 当前基线与不可伪造的阻塞项

当前分支已经包含服务 11 个任务、Planner/Evidence/Benchmark 工具链、离线 frozen
dataset 开发视图和结果 renderer。当前检查确认：

| 项目 | 当前状态 | 上线影响 |
| --- | --- | --- |
| Replay API/UI | 已有真实 HTTP/SSE 成功路径 | 可进入 Release A |
| `research-v1` 生产组合 | `RESEARCH_GRAPH_UNAVAILABLE` | Release C 前不可宣称完整多 Agent 产品 |
| `CHECKPOINT_RESUME_UNAVAILABLE` | 真实 Core continuation 尚未支持 | UI/API 必须继续公开稳定 409 |
| `docs/results.md` | public aggregate 未封存 | 不得发布 Benchmark 数字 |
| formal model/environment lock | Qwen/vLLM/GPU 锁缺失 | 正式 primary run 不可开始 |
| private Gold/test runtime | 不在仓库 | evaluator/agent 隔离运行不可完成 |
| external raw/lock/results | 未下载、未授权或未封存 | Portfolio Full 保持离线/fail-closed |
| human ratings | 未收齐三人盲评 | 不得发布人工结论 |
| Docker/Postgres/paid smoke | 当前 Windows 主机未完成实机验证 | 发布说明必须标为未验证，CI/目标环境复核后再宣称 |

实现不得通过生成零值、合成评分、把 Replay 结果标作 Live 或把一次 Kimi smoke
标作正式 Benchmark 来清除这些阻塞。

## 3. Release A 架构

### 3.1 配置与数据流

Release A 使用以下固定数据流：

```text
浏览器 Streamlit
    │ HTTP / SSE（保留 owner cookie 与 Last-Event-ID）
    ▼
FastAPI create_app / lifespan
    ├─ ServiceSettings + DeploymentPolicy
    ├─ FileProviderRouteCatalog（replay profile）
    ├─ FilePricingCatalog
    ├─ SqlAlchemyRunStore + Core checkpointer
    ├─ LimitManager + RunManager
    └─ DefaultCoreRunnerBuilder
         ├─ verified ReplayBundle
         ├─ replay_parent = snapshot.run_id
         ├─ baseline parser router（HTML/PDF）
         └─ paired monotonic/UTC runtime hooks
```

Catalog 与 bundle 是服务端只读输入。请求只声明问题和公开运行选项；服务端根据
policy 归一化后冻结 route/config/pricing identity。报告、证据图和 manifest 通过
artifact endpoint 下载，事件只从持久化 Store 重放。

### 3.2 跨平台冻结文件规则

所有 hash-addressed JSON/JSONL、manifest、snapshot 和结果 sidecar 必须在 Git 中
保持 LF。新增 `.gitattributes` 只作用于这些确定的内容类型，不改变 Markdown 或
源代码的既有换行约定。Windows 本地若已被 CRLF 污染，只能从 Git blob 或干净 LF
checkout 恢复；不得重新计算 manifest hash 来接受换行转换。

### 3.3 Release A 门禁

发布前必须在干净的 commit 上通过：

1. `uv lock --check` 和锁定依赖安装。
2. 服务关键 API/SSE、Replay、security、limits、lifespan 和 UI contract 测试。
3. `ruff check .`；生产/application `pyright src apps benchmarks experiments`。
4. 本地真实 HTTP/SSE replay：最终状态 `completed`、`run_completed`、报告/证据/
   manifest 可下载，manifest 的 `replay_parent` 与 bundle snapshot 一致。
5. 两个不同 owner 的 404 同构隔离、Last-Event-ID 重连不重复、未知问题不 Live
   fallback、日志/事件/manifest 无 secret。
6. CI Linux 上的离线测试、Compose 静态配置和 Docker build。Docker/Postgres 实机
   启动仍需在有 Docker/数据库的目标环境单独验收。

## 4. Release B Kimi 接入

### 4.1 既有 provider 的复用

不新增一个“伪 Kimi” provider。Kimi 的 OpenAI-compatible 接口使用现有模型适配器，
由 catalog 冻结：

```json
{
  "operation": "model",
  "provider_id": "openai-compatible",
  "endpoint_type": "chat.completions",
  "model_id": "kimi-k3",
  "model_revision": "provider-managed",
  "base_url": "https://api.moonshot.cn/v1",
  "credential_ref": "MODEL_API_KEY",
  "fallback_rank": 0,
  "parameters": {}
}
```

这是示意 route；正式 profile 仍必须包含唯一的 search/fetch/parse/embed route 和
匹配 pricing snapshots。不同区域使用不同 base URL，但 URL 不得包含 query、fragment
或 credentials。

### 4.2 Secret、成本与安全

- `MODEL_API_KEY` 只在 server-side credential resolver 成功后进入 provider 对象。
- `public_provider_profile`、checkpoint state、event payload、SSE、日志和 artifact
  下载继续经过现有 redaction boundary。
- Live profile 必须使用 public deployment policy 的低/中预算、每日成本上限和
  provider allowlist；请求不能提高预算或切换到未批准 profile。
- 未提供 `SEARCH_API_KEY`、pricing 或 embedding lock 时，在 admission/factory
  阶段返回稳定错误，不尝试部分 Live 执行。

### 4.3 Kimi 原生联网搜索的明确边界

Kimi 官方 `$web_search` 返回的 tool-call/加密结果不是当前 SearchProvider 的
`SearchHit`/`RawDocument` 合同。后续若接入，必须新增一个独立 adapter，完成：

1. schema/版本和请求 hash 的冻结；
2. 结果到 `SearchHit`、来源和证据 locator 的完整映射；
3. redaction、SSRF、超时、取消和成本核算；
4. Replay 录制/回放和未知 query 的 fail-closed 测试。

在上述测试完成前，Kimi 只作为模型 provider，不宣称已提供项目的联网搜索能力。

## 5. Release C research-v1 与正式评测

### 5.1 生产研究图组合

Release C 不修改现有 baseline graph，也不创建第二个 runner。`DefaultCoreRunnerBuilder`
继续返回唯一的 `LangGraphResearchRunner`，但在 workflow 为 `research-v1` 且配置已
通过完整能力校验时，额外构造并注入一个 `build_research_graph(...)` 结果。依赖实例
必须共享同一组 model、frozen search/fetch/materializer、parser、artifact/evidence
store、budget/accounting hooks 和 checkpointer。

研究图节点顺序固定为：

```text
ValidateRequest → Plan → DecideNext
    ├─ SEARCH → Search → Fetch → ParseAndNormalize → StoreEvidence → RankEvidence → DecideNext
    └─ STOP → DraftReport → ExtractClaims → VerifyClaims
                      ├─ TARGETED_RESEARCH → Search（最多一轮）
                      ├─ RESOLVE_UNSUPPORTED → ResolveUnsupportedClaims → FinalizeCitations
                      └─ FINALIZE → FinalizeCitations
                                      → PersistResults → END
```

每个有副作用节点使用稳定 idempotency key 和 checkpoint-safe usage settlement。状态
只保存计划、ID、typed decisions、预算/coverage 摘要和 artifact references；完整正文、
raw provider response、credential 和隐藏思维链留在受控 store 或不进入状态。研究图
必须在 provider 调用前拒绝不支持的 planner/ranker、缺失 snapshot、Live fetcher
混入 frozen formal composition、非法 citation 或超预算继续搜索。

### 5.2 Formal seal 输入

正式 primary run 只能在以下输入全部存在并通过 hash 校验后启动：

- Frozen AI/CS Research 60 的完整 dev/test 运行视图、私有 Gold 和每题 snapshot；
- sealed `formal.yaml`、当前 40 字符 commit、code tree hash；
- pinned model lock、inference-environment lock、serving/decoding profile；
- embedding/model provider identity 和 pricing snapshot；
- evaluator/agent 进程隔离与无 Gold 泄漏证明；
- A/B/C/D、P0、三预算和预注册 seed/repeat 计划。

结果生成后，renderer 只读取 hash-verified public aggregate，写入 deterministic
Markdown/SVG 和 sidecar manifest。任何缺失输入均保持 `not sealed`，不以默认值代替。

## 6. API、错误和运维策略

- Release A 默认 `ALLOWED_EXECUTION_MODES=["replay"]`；Release B 通过独立部署 profile
  显式开启 live，不能复用 showcase 默认。
- `research-v1` 不可用时返回稳定 `RESEARCH_GRAPH_UNAVAILABLE`；resume 继续返回
  `CHECKPOINT_RESUME_UNAVAILABLE`，不伪造续跑成功。
- `REPLAY_MISS`、invalid route/pricing、SSRF、超时、限流和 owner mismatch 不返回
  路径、DSN、credential、provider 原始响应或内部 traceback。
- readiness 只代表服务基础设施可用，不代表 provider credential、bundle coverage、
  research graph 或 formal seal 可用。
- Docker/Compose 使用非 root API、Postgres `postgres` 账户、cap drop、no-new-
  privileges 和 secret-free 配置；正式云部署还需 TLS、secret manager、proxy CIDR、
  backup/restore 和目标数据库复核。

## 7. 测试与发布证据

### Release A 必须提交的代码/测试

- `.gitattributes` 的 LF 规则和跨平台恢复 contract；
- release readiness 命令或 CI job，输出只含版本、状态、计数和 hash，不含 secret；
- README/部署文档的 Demo、Live、Benchmark 三种边界和启动命令；
- 真实 strict replay API/SSE 回归及 artifact/hash assertions；
- Kimi route 的 mock HTTP contract（成功、错误、超时、redaction、region URL）；
- 当前已存在的服务安全、额度、隔离、SSE、UI、Docker config 测试保持通过。

### Release B 必须额外证明

- 使用 Kimi-compatible endpoint 的 mock/online smoke 均只从环境读取 key；
- Live profile 的价格、额度、超时、重试、provider identity 和 manifest 一致；
- Tavily/Serper search 与 Kimi model 的真实端到端运行有单独标记和费用审计；
- Kimi `$web_search` 未接入前，文档和错误信息不将其描述为已支持搜索。

### Release C 必须额外证明

- research-v1 所有节点、停止路径、冲突补搜、unsupported claim resolution、citation
  guard、checkpoint/idempotency 和 replay provider 覆盖；
- primary formal seal、external seal、human sidecar seal 分开验证；
- `docs/results.md` 由 renderer 从 public summary 生成，第二次渲染逐字节一致；
- 负面结果、失败、缺失和估算成本都公开标注，不把 ORACLE/P0 混入 agent 结果。

## 8. 回滚与停止条件

任何发布候选出现以下任一情况，停止发布并保留上一版：secret 泄漏、owner 隔离失败、
manifest/hash 不匹配、Replay 自动 Live fallback、unknown-cost reservation 被清除、
服务端 route policy 可被请求绕过、研究图缺节点或 Benchmark 输入缺 seal。回滚只切换
镜像/commit 和只读 catalog，不删除数据库、checkpoint、artifact 或 usage ledger；恢复
后先运行 readiness、账务 reconciliation 和 durable event replay。

## 9. 预期交付顺序

1. Release A：LF 规则、readiness gate、文档/README、CI/Docker/目标环境验收说明。
2. Release B：Kimi model-only live profile、Tavily/Serper 组合和显式费用/密钥 smoke。
3. Release C：research-v1 生产 composition；资源满足后执行 formal primary，再分别
   执行 Portfolio 和 human evaluation，最后更新结果页。

该顺序允许 Demo 尽快上线，同时保证任何一层的展示都不会误称为更高层级的研究或
Benchmark 结果。
