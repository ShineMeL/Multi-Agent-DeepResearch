# Multi-Agent Deep Research 设计文档

- 日期：2026-08-29
- 状态：已批准
- 项目仓库：`ShineMeL/Multi-Agent-DeepResearch`
- 目标周期：4–6 周
- 求职定位：主投 LLM / Agent 算法岗，次投 AI 应用 / 后端岗

## 1. 项目定位

本项目实现一个可控、可评测、可重放的 Deep Research Agent。系统接收复杂问题，拆解研究任务，生成和调度搜索查询，从网页或论文中提取证据，判断证据是否充分，并生成带可验证引用的研究报告。

项目不以“能搜索并生成长报告”作为主要创新。现有开源项目已经覆盖多 Agent 编排、搜索、引用、实时进度和长报告。本项目重点研究两个可独立替换、独立评测的模块：

1. **Adaptive Planner**：根据子问题覆盖度、证据冲突、最近边际收益和剩余预算决定下一步搜索与停止时机。
2. **Evidence Ranker**：以 claim-level 证据价值而非单一语义相似度筛选材料，并构建 `claim → evidence span → source` 可追溯关系。

核心研究假设是：

> 在固定候选池下，Evidence Ranker 能提高引用支持率和信息覆盖率；在相同搜索服务快照与硬预算下，Adaptive Planner 能以更少的重复搜索取得不劣于 Fixed Planner 的覆盖度。两者组合应改善端到端质量—成本关系。

## 2. 用户与使用场景

### 2.1 主要用户

- 需要进行开放网页技术调研的学生、开发者和研究人员。
- 评估 Agent 规划与证据筛选方法的实验人员。
- 通过公开 Demo 了解项目能力的招聘者。

### 2.2 产品范围

- Demo 接受开放域问题。
- 正式 Benchmark 聚焦 AI / 计算机技术调研，以便构造可靠的证据标准和人工评测协议。
- 默认输出中文报告，同时允许检索中文和英文资料；内部数据模型不绑定输出语言。

### 2.3 三种运行入口

1. **Replay Showcase**：无需密钥，重放已保存的真实研究轨迹，是公开 Demo 的默认入口。
2. **Constrained Live**：使用服务端密钥和硬预算执行受限的在线研究。
3. **Local Full**：用户配置自己的 Provider，可调整模型、搜索后端和预算，并运行完整评测。

## 3. 目标与非目标

### 3.1 MVP 目标

- 实现从复杂问题到带引用报告的完整闭环。
- Planner、Ranker、Writer 和 Provider 之间具有明确接口。
- 每个事实 claim 能追溯到具体原文片段、来源、哈希和抓取时间。
- 支持在线运行、缓存、checkpoint、恢复、取消和结构化事件流。
- 支持固定输入的离线 Replay，确保消融实验公平、可重复。
- 提供 Planner × Ranker 的 2×2 消融、质量—成本曲线和失败分析。
- 提供最小 CLI、FastAPI 服务、Streamlit Demo、Docker 和本地一键运行方式。

### 3.2 MVP 明确不做

- SFT、GRPO、RL 或其他模型训练。
- 自由对话式 Agent 群、复杂 Supervisor 层或无限递归研究。
- 浏览器自动化集群、代理池和绕过网站访问策略。
- Redis、Celery、Kafka、Kubernetes 或多进程任务系统。
- 复杂多租户、OAuth、计费和生产级 RBAC。
- 大规模向量数据库或 FAISS 索引。单次研究的候选池在内存中完成排序。
- 多格式导出工厂。MVP 只导出 Markdown 报告、Evidence JSON 和 Run Manifest。
- 同时维护多套 UI 或十几个 Provider Adapter。

## 4. 总体架构

```text
Streamlit UI
    ⇅ HTTP / SSE
FastAPI Run Service
    ↓ typed RunConfig / RunEvent
LangGraph Controlled Research Graph
    ├── Scope Agent
    ├── Adaptive Planner
    ├── Query Scheduler
    ├── Search / Fetch / Parse
    ├── Evidence Ranker
    ├── Coverage Verifier
    ├── Report Writer
    └── Citation Verifier
    ↓ artifact IDs / evidence IDs
Persistence
    ├── Graph Checkpoint Store
    ├── Artifact Cache
    ├── Evidence Store
    └── Run Manifest / Event Store
    ⇅
Provider Adapters
    ├── OpenAI-compatible LLM
    ├── Tavily Search
    ├── HTTPX Fetcher
    ├── Trafilatura HTML Parser / PyMuPDF PDF Parser
    └── Replay Providers
```

### 4.1 三条强制边界

1. **核心不依赖供应商**：领域模型和工作流只依赖项目自己的 Protocol。Provider SDK 只存在于 Adapter 层。
2. **证据先于写作**：Writer 只能引用已经进入 Evidence Store 的证据片段，不能临时生成 URL 或引用。
3. **在线与实验分离**：Live 追求新鲜度和可用性；Benchmark 使用冻结快照和 Replay，确保不同消融配置看到同一候选资料。

### 4.2 为什么使用受控状态图

“Multi-Agent”在本项目中表示职责隔离的 Agent 节点，而不是多个 Agent 自由聊天。LangGraph 用于显式循环、条件路由、checkpoint 和恢复。语义判断由 LLM 完成，预算、校验、去重、排程和停止由确定性控制器执行。

## 5. 领域数据模型

### 5.1 ResearchRequest

```text
question
output_requirements
report_language
source_languages
freshness_requirement
execution_mode: live | replay | hybrid
access_profile: showcase | public_live | local
provider_profile_id
run_purpose: demo | benchmark | test
budget_preset: low | medium | high
```

`execution_mode` 决定全部 Provider 为真实、全部为 Replay，或按 Provider Profile 混合；`access_profile` 决定限额和密钥来源；`provider_profile_id` 指向服务端或本地配置，配置内容不进入 ResearchRequest；`run_purpose=benchmark` 强制启用锁定配置和实验校验。正式首次评测使用冻结 Search/Document Corpus + 锁定的真实 Model Provider，随后保存的完整轨迹可用 strict Replay 精确重放。

### 5.2 ResearchPlan 与 SubQuestion

```text
ResearchPlan
  plan_id
  scope
  subquestions[]
  created_by_model
  prompt_version

SubQuestion
  id
  question
  rationale_code
  importance: 0..1
  dependencies[]
  information_needs[]
  evidence_requirements
  status: pending | active | covered | blocked
```

`information_needs` 描述需要确认的事实、比较、限制或争议，不预先写入结论，避免 Planner 诱导系统只寻找支持既有结论的材料。

### 5.3 CoverageLedgerEntry

```text
subquestion_id
coverage_score
independent_source_count
unresolved_conflict_ids[]
uncertainty_score
last_marginal_gain
evidence_ids[]
attempt_count
last_decision_code
```

### 5.4 RunBudget

```text
max_search_calls
max_pages
max_total_tokens
max_wall_time_seconds
max_cost_usd
max_retries
used_by_node
```

预算预设如下：

| 预设 | 搜索 | 页面 | 总 Token | 总时长 | 费用上限 |
|---|---:|---:|---:|---:|---:|
| low | 4 | 8 | 20k | 180 秒 | 0.25 USD |
| medium | 8 | 12 | 40k | 300 秒 | 0.50 USD |
| high | 12 | 20 | 70k | 480 秒 | 1.00 USD |

Token 预算采用 Provider 返回的 `total_tokens`；若 Provider 只返回分项，则按 `input_tokens + output_tokens + reasoning_tokens` 计算，并避免把已包含在 input 中的 cached tokens 重复相加。所有成功调用和已产生 usage 的失败/重试调用都扣账；纯缓存命中记录 saved usage，但不再次扣 Token 或费用。每次调用仍有独立的最大输出 Token 限制，该限制属于 Provider 请求配置，不属于全局 RunBudget。

费用统一以 USD 记录。Provider 费用随价格变化，因此 `max_cost_usd` 是运行配置；Run Manifest 必须保存模型 ID、价格快照日期、输入/输出/缓存/推理单价和实际费用估算。Public Live 和正式 Benchmark 在缺少价格配置时拒绝启动；Local 模式可以只使用 Token 硬上限，但费用字段明确标记为 `unknown`。Public Live 的 medium 默认费用上限为 0.50 USD。

全局预算在调度前预留、调用完成后按实际 usage 结算；取消和失败释放未使用预留。`used_by_node` 分别记录 Planner、Ranker、Writer、Judge 和 Tool，用于成本归因，但所有节点共享同一个硬上限。

### 5.5 Source、Evidence 与 Claim

```text
SourceDocument
  source_id
  canonical_url
  title
  authors
  published_at
  retrieved_at
  content_hash
  parsed_content_hash
  source_type
  source_family_id
  parser_version

EvidenceSpan
  evidence_id
  source_id
  locator: HtmlLocator | PdfLocator
  excerpt
  excerpt_hash
  language
  information_need_ids[]

Claim
  claim_id
  text
  claim_type
  entities[]
  numbers[]
  qualifiers[]
  report_section
  verification_status

ClaimEvidenceLink
  claim_id
  evidence_id
  relation: support | contradict | context | insufficient
  entailment_score
  relevance_score
  judge_model
  prompt_version
  decision_code
```

完整正文属于 Artifact Store。LangGraph State 和 checkpoint 只保存计划、账本、预算、状态和 artifact/evidence ID。

### 5.6 关键类型、定位和哈希规范

除带 `?` 的字段外，本文 schema 字段均必填；时间使用带时区的 RFC 3339，日期使用 ISO 8601，分数为有限浮点数并裁剪到 `[0,1]`，ID 为项目生成的不可变字符串。`scope` 是 `{included_topics: list[str], excluded_topics: list[str], date_range?: {start?, end?}, answer_shape: str}`。

- `freshness_requirement`：`none`、`published_after: YYYY-MM-DD` 或 `retrieved_within_days: int` 三选一。
- `evidence_requirements`：`min_independent_sources: int`、`allowed_source_types: set[SourceType]`、`must_include_primary: bool` 和可选 freshness。
- `SourceType`：`paper | official_documentation | standard | primary_data | first_party_statement | secondary_analysis | news | unknown`。
- `ClaimType`：`fact | numeric | comparison | trend | causal | limitation`。
- `verification_status`：`supported | contradicted | uncertain | unsupported`。
- `decision_code`：版本化的公开枚举；未知值必须保留原字符串并按 `unknown` 处理，不能使旧 Run 无法读取。

正文先转换为 Unicode NFC、LF 换行和稳定 block 顺序。连续行内空白折叠为单空格，但不跨 block 合并。所有 `start_offset/end_offset` 均指**规范化文本的 Unicode code point 索引**，区间为左闭右开。

证据定位是可辨识联合类型：

```text
HtmlLocator = {paragraph_id, start_char, end_char}
PdfLocator  = {page_index, block_index, start_char, end_char}
```

`content_hash` 是原始响应 bytes 的 SHA-256；`parsed_content_hash` 是规范化全文 UTF-8 bytes 的 SHA-256；`excerpt_hash` 是规范化 span UTF-8 bytes 的 SHA-256。

`source_family_id` 按以下确定性顺序生成：相同 canonical URL 或相同 `parsed_content_hash` 直接归为同一 family；否则仅当正文 SimHash 汉明距离不超过 3 且规范化标题相似度不低于 0.90 时合并。人工 gold 可以覆盖自动 family 标注，但覆盖记录必须进入 dataset 版本。

Provider 边界使用以下最小类型：

```text
SearchHit = {url, title, snippet, rank, published_at?, provider_metadata}
RawDocument = {requested_url, final_url, status, headers, content_type, body_bytes, retrieved_at}
ParsedDocument = {canonical_url, title, authors, published_at?, normalized_text, blocks, parser_id, parser_version, parsed_content_hash}
RerankScore = {evidence_id, total, feature_scores, model_id?, prompt_version?}
ProviderError = {code, retryable, provider, operation, public_message, retry_after?, usage?}
```

所有 Provider 方法均为 async，接受 deadline 和 cancellation token，返回 typed result 或抛出 `ProviderError`；Adapter 不能把供应商异常对象泄漏到核心层。

## 6. Adaptive Planner

### 6.1 首次规划

Scope Agent 将用户问题转为结构化研究 brief。Planner 生成子问题、依赖、重要度、信息需求和证据要求。Plan Validator 随后执行：

- Schema 校验。
- 子问题去重。
- 依赖环检测。
- 预算可行性校验。
- 空问题、越界问题和不可执行目标检查。

无效计划最多修复一次；再次失败则以 `PLAN_INVALID` 结束，而不是进入不可控循环。

### 6.2 调度策略

Scheduler 每轮选择单位成本期望收益最高的未满足子问题：

```text
priority = importance × evidence_gap × expected_gain / (estimated_cost + ε)
```

MVP 不训练策略模型。`expected_gain` 使用以下结构化特征估计：

- 未覆盖 information need 比例。
- 最近两轮边际证据收益。
- 当前独立来源数量。
- 未解决冲突的重要度。
- 相似查询的历史成功率。
- Provider 失败与预计成本。

所有调度特征都归一化到 `[0,1]`：`evidence_gap = 1 - coverage_score`；`expected_gain = 0.40 × recent_gain + 0.25 × new_source_need + 0.20 × conflict_resolution_need + 0.15 × historical_success`，无历史时 `recent_gain` 和 `historical_success` 取 0.5；`estimated_cost = 0.50 × token_fraction + 0.30 × search_fraction + 0.20 × time_fraction`；`ε = 0.05`。所有预测成本使用该动作预计消耗占 medium 预算的比例并裁剪到 `[0,1]`。

LLM 只为选中的信息缺口生成候选查询。确定性 Query Scheduler 负责查询规范化、近重复检测、并发和预算扣账。查询经 Unicode NFC、大小写折叠、空白/标点规范化后精确去重；对其余候选使用与 R1 相同的 embedding，余弦相似度不低于 0.92 时视为语义重复，只保留优先级最高的一条。

### 6.3 定向重规划

每次搜索和 Evidence Ranking 后更新 Coverage Ledger。如果继续研究，只针对以下情况生成增量查询：

- 关键 information need 未覆盖。
- 只有单一来源。
- 存在高优先级冲突。
- 来源时效不符合要求。
- 上一次查询失败但仍有不同检索策略。

系统不在每轮重写整个计划。

### 6.4 停止策略

Planner 必须输出下列公开停止码之一：

- `SUFFICIENT`：所有关键子问题的 coverage 不低于 0.85、满足各自的独立来源要求、整体加权 coverage 不低于 0.80，且没有高优先级未解决冲突。默认要求两个独立来源；如果 claim 本身描述某一份一手材料的内容，可以由该一手材料单独满足。
- `PLATEAU`：连续两轮 `last_marginal_gain < 0.05`。
- `BUDGET_EXHAUSTED`：搜索、页面、Token、时间或费用触达硬限制。
- `BLOCKED`：必要来源无法获得，替代策略和重试均失败。

上述数值是 MVP 默认值；只允许在开发集上调整，并在封存测试前写入正式实验配置。停止结果、结构化原因和未覆盖项进入报告限制章节。系统不保存或展示隐藏思维链。

### 6.5 Planner 变体

- `P0 ReAct`：无显式 Coverage Ledger，作为额外外部参考。
- `P1 Fixed Plan`：只规划一次，固定每个子问题的搜索深度。
- `P2 Adaptive Planner`：使用 Coverage Ledger、增量重规划和动态停止。

## 7. Evidence Ranker

### 7.1 Pass A：写作前候选筛选

1. **Normalize and Dedupe**：canonical URL、精确内容哈希、近重复片段和转载来源分组。
2. **Cheap Prefilter**：关键词或向量召回、语言、日期、内容长度、解析质量。
3. **Semantic Rerank**：针对 SubQuestion 和 information need 计算细粒度相关性。
4. **Marginal Utility Selection**：在上下文预算内选择新增覆盖最大且来源独立的证据。

### 7.2 可解释效用分数

```text
utility(e, need) = clip(
    0.25 × relevance
  + 0.20 × support_strength
  + 0.15 × source_quality
  + 0.20 × coverage_gain
  + 0.10 × independence
  + 0.05 × freshness
  - 0.15 × redundancy
  - 0.05 × risk,
  0, 1)
```

这些是初始权重，只能在 30 题开发集上调整；封存测试前冻结。每个结果必须保存逐项 score breakdown。总分只用于排序，不解释为真值概率。

`freshness` 按任务需求启用。冲突证据不自动降权，因为独立、高质量的反例本身具有研究价值。

Pass A 中的 `support_strength` 表示片段是否直接回答当前 information need，不表示它已经蕴含最终报告中的 claim。最终 claim 的蕴含关系只在 Pass B 中计算。

### 7.3 来源质量

来源质量不使用简单域名白名单，而由可观察特征构成：

- 来源类型：论文、官方文档、标准、原始数据、第一方声明或次级分析。
- 作者、日期和出版信息完整性。
- 是否提供可核验原始数据或方法。
- 与当前任务相关的时效性。
- 内容是否具体支持所需事实和限定条件。
- 是否属于其他来源的近重复转载。

各特征统一为 `[0,1]`：`relevance=(cosine+1)/2`；`support_strength` 使用结构化四级 rubric `0/⅓/⅔/1`；`source_quality=0.40×source_type + 0.20×provenance_completeness + 0.20×directness + 0.20×data_verifiability`；`coverage_gain` 是新覆盖 information need 的重要度占比；`independence` 对新 source family 为 1、已见 family 为 0；`freshness` 满足要求为 1、未知为 0.5、违反为 0，无 freshness 要求时固定为 1；`redundancy` 是与已选择证据的最大归一化余弦相似度；`risk` 对完整且可定位正文为 0、截断/低置信解析为 0.5、仅 snippet 或缺少稳定 locator 为 1。开发集调权时不得改变这些特征定义。

### 7.4 Pass B：claim-level 验证

Writer 先使用已选择证据生成带 Evidence ID 的草稿。Claim Extractor 将草稿拆成 atomic claims。Evidence Judge 输出结构化的 `support / contradict / context / insufficient` 关系。

不充分 claim 按顺序处理：

1. 查找已有但尚未链接的证据。
2. 在预算内触发最多一轮定向补搜。
3. 仍不充分时删除、改写为明确的不确定表述，或进入报告限制章节。

Citation Verifier 确认所有行内引用均指向已存在的 EvidenceSpan，并验证 span 的原文定位和来源哈希。

### 7.5 Ranker 变体

- `R0 Search Order`：直接采用搜索引擎返回顺序。
- `R1 Similarity Only`：仅使用语义相关性排序。MVP 默认使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 的余弦相似度；任何替换都必须创建新的实验配置并写入 manifest。
- `R2 Evidence Utility`：使用完整的质量、覆盖、独立性、时效和冗余特征。

## 8. 检索与 Provider Adapter

### 8.1 Protocol

```text
ModelProvider
  complete
  structured
  stream

SearchProvider
  search(query, limit, filters) -> SearchHit[]

Fetcher
  fetch(url) -> RawDocument

Parser
  supports(content_type) -> bool
  parse(raw_document) -> ParsedDocument

Reranker
  score(information_need, evidence_spans) -> Score[]
```

Model Adapter 统一返回文本、结构化结果、usage、tool calls、provider model ID 和原始响应引用。核心包不直接依赖 LangChain Provider 类或具体 SDK。

### 8.2 MVP Adapter

- `OpenAICompatibleModelProvider`：通过 `base_url`、`api_key` 和 `model_id` 配置 OpenAI-compatible 服务。
- `TavilySearchProvider`：唯一必做的在线搜索 Adapter。
- `HttpxFetcher`：连接池、超时、重定向、Content-Type 和响应大小限制。
- `TrafilaturaHtmlParser`：HTML 正文、metadata 和段落边界提取。
- `PyMuPdfParser`：文本型 PDF 的页、block 和字符定位；MVP 不做扫描件 OCR，无法提取文本的 PDF 记录为 `PARSE_UNSUPPORTED` 并寻找 HTML/摘要或其他来源。
- `FrozenCorpusSearchProvider`：正式内部 Benchmark 使用的确定性 BM25 检索；在任务级冻结 corpus 上接受任意 query，以 score 降序、再以 evidence/source ID 升序打破平局。
- `ReplayModelProvider`、`ReplaySearchProvider`、`ReplayFetcher`：Showcase、测试和既有正式轨迹的精确固定输入。

第二个网页搜索服务、论文专用检索服务和浏览器渲染不属于 MVP。

模型 Adapter 不设置隐式默认模型。Live 和正式 Benchmark 都必须在配置中显式给出 `base_url` 和 `model_id`；正式结果使用的精确模型、参数和价格快照在封存评测前写入 `benchmarks/configs/formal.yaml`。

## 9. 缓存、Replay 与可复现性

### 9.1 缓存键

- Search：`snapshot_id + normalized_query + provider_id + endpoint_type + locale + complete_parameters + time_policy`。
- Fetch：`snapshot_id + canonical_url + fetch_policy + accepted_content_types`。
- Parse：`snapshot_id + raw_content_hash + parser_id + parser_version + normalization_version`。
- Model：`provider_id + endpoint_type + model_id + prompt_version + system_prompt_hash + tool_schema_hash + output_schema_hash + temperature + seed + canonical_request_hash`。

LangGraph checkpoint 不是缓存。恢复旧 checkpoint 时仍可能重新执行外部调用，因此所有有副作用的节点必须先检查 artifact/cache，并使用稳定 idempotency key。

Replay snapshot 至少包含 `snapshot.json`、`search.jsonl`、`documents.jsonl`、`model_responses.jsonl` 和 `manifest.sha256`。所有请求用排序键 JSON 序列化后计算 request hash。严格 Replay 遇到未知 query、未知 model request 或 hash 不匹配时返回 `REPLAY_MISS`，整次 Benchmark run 标记为 invalid，不能回退到 Live。Showcase fixture 可以只包含一条完整预录轨迹，但仍使用相同 schema。

### 9.2 Run Manifest

每次运行保存：

- 输入和 RunConfig。
- Git commit、依赖 lock hash。
- Provider、endpoint、模型、temperature、seed、tool/output schema 和 prompt 版本。
- 搜索查询、时间戳和参数。
- Snapshot ID、Artifact、解析文本和 evidence hash。
- 节点事件、停止原因和错误码。
- 输入、输出、缓存和 reasoning token 使用量。
- 搜索调用、页面数、延迟和费用估算。

### 9.3 版权边界

原始 HTML/PDF 快照只存放在本地或受控 Artifact Store，不提交公开仓库。公开测试 fixture 仅包含自建/合成页面、许可允许的内容、必要短片段、URL、内容哈希和抓取元数据。公开 Benchmark 提供刷新脚本，但刷新后的实验必须使用新的 snapshot ID。

“一条命令重放完整研究轨迹”指仓库内许可明确的 Showcase/CI fixture。正式网页 Benchmark 的**精确**重放依赖运行者拥有对应 snapshot artifact；仓库提供 snapshot manifest、hash 校验和重建命令，但不会为了可重复性重新分发无许可的完整网页。重建得到的新内容属于新 snapshot，结果不能与旧 snapshot 混为同一实验。

## 10. 运行服务与 API

### 10.1 生命周期

```text
POST /runs
  → validate request and budget
  → create run_id and thread_id
  → return 202
  → run graph asynchronously
  → persist events/checkpoints/artifacts
  → finish with report/evidence graph/manifest
```

Run 状态为：

```text
queued | running | interrupted | completed | failed | cancelled
```

### 10.2 HTTP 接口

```text
POST /runs
GET  /runs/{run_id}
GET  /runs/{run_id}/events
POST /runs/{run_id}/resume
POST /runs/{run_id}/cancel
```

`GET /events` 使用 SSE。事件包含递增 `seq/id`，客户端可通过 `Last-Event-ID` 断线续传。

### 10.3 RunEvent

```text
seq
run_id
timestamp
node
kind
status
public_payload
usage_delta
artifact_ids
error_code
```

UI 只消费项目自己的 RunEvent，不能直接依赖 LangGraph 原始事件。`public_payload` 只包含查询、结构化决策、评分明细、进度和公开错误，不包含密钥、完整 Provider 响应或隐藏思维链。

### 10.4 Run 状态机与端点语义

```text
queued → running → completed
              ├→ interrupted → running   (resume)
              ├→ cancelled              (terminal)
              └→ failed                 (terminal system failure)
```

`stop_reason` 与 Run 状态分离：`SUFFICIENT`、`PLATEAU` 和 `BUDGET_EXHAUSTED` 均可产生 `completed`；`BLOCKED` 在仍能生成最低限度部分报告时产生 `completed`，并设置 `is_partial=true`，否则为 `failed`。Schema/内部异常、无法创建任何报告或数据损坏产生 `failed`。

`POST /resume` 只允许 `interrupted → running`；对正在运行的 run 重复调用返回当前状态而不创建第二个任务，对其他终态返回 409。`POST /cancel` 对 queued/running/interrupted 为幂等操作；已 cancelled 再调用仍返回 cancelled；completed/failed 返回 409。取消是 best effort，已经完成并产生 usage 的外部调用仍计入预算。

RunEvent 先持久化再发送。`Last-Event-ID=n` 使服务从数据库返回 `seq>n` 的事件，然后切换到实时流；同一 run 的 seq 严格单调且唯一。客户端重复收到事件时按 `(run_id, seq)` 去重。

## 11. 错误处理与安全

### 11.1 外部故障

- 429、5xx 和网络超时使用带 jitter 的指数退避。
- 每个 Provider 和域名使用独立 semaphore。
- 4xx、禁止访问、格式不支持和超大响应不盲目重试。
- 无法抓取的来源记录为结构化错误并尝试替代来源。
- Live 允许显式 Provider fallback；Replay/Benchmark 禁止静默 fallback。

默认每个外部操作最多重试两次。Search、Fetch 和 Model 的单次默认 deadline 分别为 30、30 和 120 秒，并受更短的剩余 Run 时间覆盖。退避等待计入 wall time，失败调用若报告 usage 则计入 Token/费用。Fallback 只适用于配置中明确列出的 Model/Search Adapter；Fetch/Parse 失败通过替代来源处理。取消信号在每个退避、批次边界和流式 chunk 检查。

### 11.2 部分结果

预算耗尽或部分来源不可用时，系统生成带未覆盖项、证据限制和停止原因的部分报告。不能把 `BUDGET_EXHAUSTED` 或 `BLOCKED` 写成充分研究结论。

### 11.3 安全边界

- 密钥只存在于服务端环境变量或 secret store。
- 日志、checkpoint、SSE、异常和 manifest 统一脱敏。
- Fetcher 禁止 `file:`、localhost、私网、link-local 和非 HTTP(S) 目标，防止 SSRF。
- 限制响应大小、重定向次数、Content-Type 和连接/读取超时。
- 网页内容作为不可信数据处理，不执行脚本，不渲染未净化 HTML。
- Prompt 明确将检索内容视为证据数据，网页中的指令不能改变工具权限、系统目标或引用规则。
- 测试 fixture 必须包含 prompt injection、伪引用、恶意重定向和秘密泄漏案例。

## 12. 报告与引用

最终报告包含：

1. 研究范围与问题解释。
2. 执行摘要。
3. 按子问题组织的分析。
4. 证据冲突和不确定性。
5. 趋势、限制和结论。
6. 未覆盖问题与停止原因。
7. 可点击参考来源。

Markdown 使用稳定 citation ID。Evidence JSON 保存 citation ID 到 ClaimEvidenceLink、EvidenceSpan 和 SourceDocument 的完整映射。报告 Writer 不直接访问搜索工具，避免写作阶段产生无法追踪的新引用。

## 13. Benchmark 与实验

### 13.1 内部主集

建立 `Frozen AI/CS Research 60`：

- 30 题开发集：允许调整 prompt、阈值和 Ranker 权重。
- 30 题封存测试集：正式设计冻结后运行。
- 六类任务各 10 题：技术路线综述、方法比较、多跳/历史演进、时效性问题、中英混合来源、来源冲突。

每题包含：

- ResearchRequest。
- Gold atomic information needs 和可接受 claim 表述。
- 信息重要度。
- 冻结候选文档/片段池、gold source family 和 snapshot ID。
- Gold evidence spans，以及 `0=无关、1=背景、2=部分支持、3=直接支持` 的 graded relevance。
- Claim 到可接受 EvidenceSpan 的多对多链接；允许多个等价来源。
- 报告评测 rubric。
- 数据创建时间和标注版本。

Gold rubric 和参考答案必须与 Agent 可访问的文件、搜索空间和网页服务隔离。

Recall/MRR/nDCG 和 claim coverage 只在带上述 gold 的冻结候选池上计算。系统发现 gold 之外的替代证据时，由锁定的 evidence judge 标记为候选等价项，并对测试集随机抽取 20% 人工复核；经确认的替代项进入下一版 dataset，不能回写当前正式结果。

封存测试题在第一次正式评测前不进入 Agent 可访问路径。正式结果冻结后可以发布题目与 rubric 供审查，但 evaluator gold 仍与运行容器隔离；任何发布后的复测必须注明可能的数据泄漏风险。

### 13.2 外部评测

外部评测属于六周完整版，不阻塞四周核心交付：

- LiveDRBench：选择 10 个 computer-science、prior-art 或 dataset-discovery 相关任务，用于 Planner/claim 检索过程。
- FRAMES：选择 20 题，用于固定语料下的 Evidence Ranker 和多文档召回。
- DeepResearch Bench：选择 10 个相关任务，用于报告与引用的外部抽样对照。

任何外部集均锁定数据版本、evaluator commit、judge model、prompt 和评测日期。外部集只评估其设计覆盖的能力，不能用短答案得分代替长报告质量。

### 13.3 2×2 主消融

| 配置 | Planner | Ranker |
|---|---|---|
| A | P1 Fixed Plan | R1 Similarity Only |
| B | P1 Fixed Plan | R2 Evidence Utility |
| C | P2 Adaptive Planner | R1 Similarity Only |
| D | P2 Adaptive Planner | R2 Evidence Utility |

额外运行：

- 在固定 10 题子集运行 P0 ReAct 参考基线。
- 10 题 oracle-evidence 上限，用于区分检索失败和生成失败。
- 主测试集 30 题使用 medium 预算；预先固定的 20 题成本子集运行 low、medium、high 三档，生成质量—成本 Pareto 曲线。

实验分为三个互补协议，不能混写成一种“公平性”：

1. **Ranker component test**：固定 SubQuestion、information need 和候选 EvidenceSpan 池，只替换 R0/R1/R2，用于 Recall/MRR/nDCG 和 score-feature 消融。
2. **Planner policy test**：固定 R1 或 R2、Writer、模型、任务级冻结文档语料和检索索引，并给 P1/P2 相同硬预算。Planner 可以生成不同 query、取得不同候选结果，这正是待测能力；公平性来自相同 corpus/search implementation/snapshot，而不是强求相同候选集合。
3. **2×2 end-to-end test**：A/B/C/D 使用同一冻结任务语料、Search Adapter、Writer、模型、prompt 和预算。查询轨迹、候选集合和实际搜索次数允许随 Planner/Ranker 改变，并作为结果报告。

内部正式评测的 Provider Profile 为 hybrid：Search/Fetch/Parse 使用冻结 snapshot 和本地确定性索引，Model 使用锁定的真实 Provider。每次正式运行记录模型响应，随后可用完整 strict Replay 精确重放。若 frozen search 对任意合法 query 无法返回确定性结果，该任务配置无效，不得临时访问公开搜索引擎。

正式实验不依赖“系统当前默认模型”。`benchmarks/configs/formal.yaml` 必须固定模型 ID、endpoint 类型、temperature、seed（若 Provider 支持）、prompt 版本、R1 模型、Ranker 权重、预算、corpus/index 版本和 snapshot ID；这些字段任一变化都产生新的实验组。

### 13.4 指标

Planner：

- information-need coverage。
- query redundancy。
- execution adherence。
- stop calibration。
- marginal utility per search。
- backtracking gain。

Evidence Ranker：

- Recall@k、MRR、nDCG@k。
- claim coverage@k。
- independent-source coverage。
- redundancy ratio。
- citation support accuracy。

端到端报告：

- citation support precision。
- citation coverage。
- unsupported-claim rate。
- information completeness。
- atomic factuality。
- analysis depth、instruction following 和 readability。

效率：

- 搜索和抓取次数。
- 各节点 Token。
- p50/p95 wall-clock 和 tool latency。
- 重试次数和失败率。
- 费用、quality/search、quality/1k tokens、quality/cost。

不以单一自定义总分替代分项结果。

### 13.5 评测协议

1. 在 30 题开发集完成所有调参。
2. 冻结代码、prompt、权重、预算和 evaluator。
3. 运行 30 题封存测试集的 A/B/C/D。
4. 在冻结前按六类任务分层选定 20 题稳定性子集；每个配置共运行三个 seed。Provider 不支持 seed 时执行三次独立重复并如实标注。
5. 选择 20 题，对 A 和 D 的报告进行盲化成对人工评测；目标三名具备 AI/CS 背景的评审。
6. 报告 paired bootstrap 95% CI、人工评审一致性和自动指标与人工评分相关性。

每个 task/config 先对 seed 取均值，再计算 task-level 配对差值。置信区间以 task ID 为重采样单位，在任务类别内分层做 10,000 次 paired bootstrap；不能把同一题的多个 seed 当作独立样本扩大样本量。

人工评审对事实正确性、证据充分性、信息覆盖、分析深度、结构可读性和引用可验证性分别使用 1–5 分，并给出 A 胜、D 胜或平局的总体偏好。三名评审的维度分数取均值；总体偏好报告多数结果和平局率；一致性使用 ordinal Krippendorff's alpha，并报告自动指标与人工维度分数的 Spearman 相关。

预先冻结的主要判断：

- Ranker 主估计量是固定候选池上 R2−R1 的 task-level citation support precision 差值；同时报告 unsupported-claim rate，不要求两个指标都显著才发布结果。
- Planner 非劣条件是 P2−P1 的平均 information completeness 差值之 95% CI 下界高于 −0.03；满足后再检验平均搜索次数和 query redundancy 是否下降。
- 质量—成本不合成单一分数。分别绘制 citation support precision–USD、information completeness–search calls 两个平面；若 D 的平均质量不低于 A、平均成本不高于 A，且至少一个方向严格改善，则记为样本均值 Pareto dominance，并同时报告 bootstrap 中满足该条件的比例。

若主要假设不成立，仍发布完整负结果、置信区间和失败分析，不事后切换主指标。

## 14. 测试策略

### 14.1 Unit Tests

- Plan Schema、依赖环和重复检测。
- Scheduler priority 和 budget accounting。
- Ranker score breakdown、边界值和去重。
- Citation offset、hash 和 ClaimEvidenceLink 完整性。
- Cache key 和 idempotency key。
- URL 安全、SSRF、防秘密泄漏和 prompt injection fixture。

### 14.2 Provider Contract Tests

所有 Adapter 必须通过同一组成功、超时、限流、无效响应、usage 和取消测试。默认使用模拟 HTTP，不在普通 CI 调用真实付费 Provider。

### 14.3 Replay Integration Tests

覆盖：

- 正常充分完成。
- 固定计划与 Adaptive Planner 路径。
- 证据冲突触发补搜。
- `PLATEAU`、`BUDGET_EXHAUSTED` 和 `BLOCKED`。
- checkpoint 恢复和幂等调用。
- 取消、部分报告和 SSE 断线续传。

### 14.4 Online Smoke Tests

单独标记并只在配置密钥的手动或定时 CI 中运行。验证一个最小 Live query、部署健康、公开 Demo 和密钥脱敏。

## 15. Demo 与部署

### 15.1 Streamlit 页面

主页面同时展示：

- 问题、模式、语言和预算预设。
- Research Plan 与子问题状态。
- 当前搜索 query 和结构化决策。
- 搜索结果与 Evidence Ranker 保留/淘汰理由。
- Evidence 表和 Claim–Evidence 图。
- Coverage、搜索、Token、时间和费用进度。
- 停止条件和最终停止原因。
- 最终报告、Evidence JSON 和 Run Manifest 下载。

UI 只调用 FastAPI 和消费 SSE，不直接调用 LLM、搜索服务或数据库。

### 15.2 公开限额

- Replay Showcase 默认开放，无付费调用。
- Live 使用 medium 预算上限：8 次搜索、12 页、40k Token、5 分钟。
- 按 IP/session 限流，并设置总日费用上限和同时运行数。
- 访客不能向浏览器或 Graph State 提交 Provider key。

单实例 MVP 使用进程内 IP/session 限流和并发 semaphore；总日费用与已用额度持久化到 Postgres，服务重启也不能绕过费用硬上限。

### 15.3 部署拓扑

MVP 使用：

```text
Streamlit process
FastAPI process with in-process RunManager
Postgres
Persistent artifact volume
```

本地使用 SQLite 和本地 artifact 目录。在线进程退出时将运行标记为 `interrupted`；用户可从最近 checkpoint 恢复。不引入独立 worker，除非单进程并发或恢复已经成为测量到的瓶颈。

Public Live 默认最多同时运行两个 run；Search 全局并发为 4，同一域名并发为 2。服务启动时将数据库中遗留的 `running` run 原子更新为 `interrupted`，不会自动产生新的付费调用。优雅关闭先停止接收新 run，等待当前节点最多 20 秒，然后取消任务、保存 checkpoint 并标记 interrupted。

## 16. 仓库结构

```text
Multi-Agent-DeepResearch/
├── src/deepresearch/
│   ├── domain/
│   ├── planning/
│   ├── evidence/
│   ├── retrieval/
│   ├── workflow/
│   ├── providers/
│   ├── storage/
│   └── runtime/
├── apps/
│   ├── api/
│   ├── ui/
│   └── cli/
├── benchmarks/
│   ├── datasets/
│   ├── evaluators/
│   ├── configs/
│   └── scripts/
├── experiments/
├── tests/
│   ├── unit/
│   ├── contracts/
│   ├── integration/
│   └── fixtures/
├── docs/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 17. 4–6 周里程碑

### 第 1 周：可重放基线

- 建立领域模型、Provider Protocol、RunBudget 和 Run Manifest。
- 完成 Replay Providers。
- 完成 P1 + R1 基线和 CLI 闭环。
- 建立 Unit/Contract/Replay 测试骨架。

退出条件：一条命令可以在固定 fixture 上生成带引用的基线报告。

### 第 2 周：检索与证据基础

- 完成 Tavily Search、HTTPX Fetch、Trafilatura HTML Parse 和 PyMuPDF PDF Parse。
- 完成规范化、切分、去重、缓存和 Evidence Store。
- 完成 citation offset/hash 验证。

退出条件：任意已引用 claim 可回溯到固定原文片段和 SourceDocument。

### 第 3 周：Evidence Ranker

- 完成两阶段 Ranker 和 R0/R1/R2。
- 完成 Claim Extractor、Evidence Judge 和 Citation Guard。
- 完成冲突证据 fixture 与 Ranker 独立指标。

退出条件：可在固定候选池上单独运行 Ranker 消融并生成 score breakdown。

### 第 4 周：Adaptive Planner

- 完成 Coverage Ledger、Query Scheduler 和停止策略。
- 完成 P0/P1/P2。
- 完成开发集调参，冻结配置并运行内部 30 题封存集的 medium A/B/C/D。
- 在预注册 20 题子集完成三档预算与质量—成本脚本。

退出条件：四种配置在相同冻结 corpus/index、模型和预算协议下完成可比运行，并产出结构化结果与置信区间。

### 第 5 周：服务与 Demo

- 完成 FastAPI、SSE、恢复、取消和 Streamlit。
- 完成 Replay Showcase、受限 Live 和安全/限额测试。
- 完成本地 Docker 运行。

退出条件：浏览器可展示完整计划、证据、预算、停止原因和报告。

### 第 6 周：外部验证与求职交付

- 运行稳定性子集、精简外部 benchmark 和人工评测。
- 生成结果表、Pareto 曲线、失败案例和局限说明。
- 部署公开 Demo，完成 CI、README 和演示视频。

退出条件：仓库首页包含可复现命令、真实数据、置信区间、Demo 和明确限制。

四周是 **Core Research** 交付：核心包、Replay、内部 60 题、A/B/C/D 和本地 CLI；不承诺公开 Live、外部 benchmark、人工盲评或演示视频。六周是 **Portfolio Full** 交付：在 Core 上增加 API/UI、受限 Live、外部验证、人工评测、公开部署和视频。若时间不足，按外部评测 → 公开自定义 Live → UI 装饰的顺序缩减，Replay Showcase 和实验结果不删除。

## 18. 完成标准

四周 Core Research 完成标准：

- 一条命令可重放完整研究轨迹。
- A/B/C/D 消融可由配置运行并生成结构化结果。
- 所有事实 claim 有 EvidenceSpan 链路或明确的不确定标记。
- CLI 可展示或导出计划、查询、证据筛选、预算、停止原因和报告。
- README 包含内部主实验的真实指标、置信区间、失败案例、成本和限制。
- Unit、Contract、Replay Integration、Lint 和类型检查通过。
- 不以漂亮个例、未锁定 Judge 或单一总分替代完整评测。

六周 Portfolio Full 还必须满足：

- FastAPI、SSE、Streamlit、Docker build 和 Replay Showcase 通过端到端验证。
- 公开页面展示计划、证据图、预算、停止原因和最终报告。
- 完成精简外部 benchmark、稳定性子集和目标三人盲评；无法招募三人时明确降级并把它列为限制，不能伪称完整人工实验。
- 公开受限 Live 已配置服务端密钥、持久费用上限、速率限制和并发上限。
- README 包含可访问 Demo、架构图、2×2 表格、Pareto 曲线和演示视频。

## 19. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 搜索排序与网页内容漂移 | snapshot ID、时间戳、hash、Replay 和刷新后新实验配置 |
| LLM Judge 漂移 | 锁定模型、prompt、版本和日期；人工抽查 |
| Benchmark 泄漏 | Gold 与 Agent 环境隔离；封存测试；补充晚于模型知识截止的题目 |
| Ranker 提升来自更大预算 | 固定候选池、Writer、模型和预算；报告 Pareto 曲线 |
| 证据评分被误解为事实概率 | 输出分项和 relation；保留冲突；明确总分仅用于排序 |
| 在线 Demo 成本或滥用 | Replay 默认、硬预算、并发/速率/日费用上限 |
| 长任务因进程重启中断 | checkpoint、幂等 artifact、interrupted 状态和手动恢复 |
| 网页 prompt injection / SSRF | 不可信内容边界、URL 校验、无脚本执行、安全 fixture |
| 4–6 周范围失控 | 严格非目标和砍项顺序；以算法与实验优先 |

## 20. 后续扩展

只有 MVP、实验和公开 Demo 完成后才考虑：

- 学习型 Query Policy 或停止策略。
- SFT / GRPO Planner。
- 训练或蒸馏 Evidence Ranker。
- 第二搜索后端或论文专用检索。
- FAISS/向量数据库和长期研究记忆。
- 分布式任务队列和多实例部署。
- React 前端替换 Streamlit。

## 21. 参考资料

- LangGraph Overview: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Streaming: https://docs.langchain.com/oss/python/langgraph/streaming
- LangGraph Fault Tolerance: https://docs.langchain.com/oss/python/langgraph/fault-tolerance
- LangChain Open Deep Research: https://github.com/langchain-ai/open_deep_research
- GPT Researcher: https://github.com/assafelovic/gpt-researcher
- Stanford STORM: https://github.com/stanford-oval/storm
- LiveDRBench: https://github.com/microsoft/LiveDRBench
- FRAMES: https://huggingface.co/datasets/google/frames-benchmark
- DeepResearch Bench: https://github.com/Ayanami0730/deep_research_bench
- DeepResearch Bench II: https://github.com/SawyerCooper/DeepResearchBench2
- ScholarQABench: https://github.com/AkariAsai/ScholarQABench
- ALCE: https://github.com/princeton-nlp/ALCE
- FActScore: https://aclanthology.org/2023.emnlp-main.741/
- FastAPI Server-Sent Events: https://fastapi.tiangolo.com/tutorial/server-sent-events/
- HTTPX Clients: https://www.python-httpx.org/advanced/clients/
- Trafilatura: https://github.com/adbar/trafilatura
- PyMuPDF: https://pymupdf.readthedocs.io/
- Tavily Search API: https://docs.tavily.com/
- Multilingual MiniLM model card: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
