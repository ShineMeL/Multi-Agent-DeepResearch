# Demo 使用说明

## 启动

在项目根目录或包含本次代码的 worktree 中打开 PowerShell，执行：

```powershell
python -m uv sync --extra dev --frozen
python -m uv run python -m scripts.run_demo
```

打开 <http://127.0.0.1:8501>；接口文档在 <http://127.0.0.1:8000/docs>。
保持启动终端运行；Ctrl+C 关闭本次启动的两个服务，不会删除研究记录。
运行记录、报告、会话密钥和 SQLite 数据保存在 `artifacts/demo/`，不进入 Git。
默认仅监听本机。端口冲突时先停止旧 Demo，或传 `--api-port 18000 --ui-port 18501`。

## 离线示例

选择「离线示例」并点击「开始研究」。问题固定为
`Compare planner strategies`，英语报告、中等预算、seed 0。
这是真实工作流读取已校验的录制数据，不联网，也不产生 API 费用；不是任意问题问答。

页面显示运行事件、搜索记录、证据、带引用的报告和下载按钮。
完成后应显示「信息已充分，研究正常完成（SUFFICIENT）」。
`SUFFICIENT` 是「充分」，不是「不充分」。

其他状态仍如实保留：

- 正在运行：尚无停止理由，等待事件与最终状态。
- `PLATEAU`：连续搜索新增信息不足，返回部分结果。
- `BUDGET_EXHAUSTED`：达到搜索、Token 或时间等预算上限。
- `BLOCKED`：无法继续满足证据需求。固定 P1 计划执行完仍不充分时，保留已有报告并明确标记为部分结果。
- `REPLAY_MISS`：请求不匹配录制数据；使用原始样例，或改用在线模式。

证据不足不能通过改标签变成充分；本次没有降低证据充分性阈值。
Windows 的回放文件必须保持 LF，不能通过重新生成哈希来掩盖 CRLF 变化。

## 接入 Kimi 与 Tavily

如果 `.env.demo` 不存在，将 `.env.demo.example` 复制为 `.env.demo`，在编辑器中填写：

```dotenv
MODEL_PROVIDER=kimi-instant
MODEL_BASE_URL=https://api.moonshot.cn/v1
MODEL_ID=kimi-k2.6
MODEL_API_KEY=在本地填入新生成的模型密钥
SEARCH_API_KEY=在本地填入Tavily搜索密钥
```

不要把上面的说明文字当密钥运行；不要把真实密钥写入命令行参数、网页或 Git。
之前公开过的密钥应在供应商后台撤销并重新生成。
Kimi 只提供本方案的模型调用；网页检索使用 Tavily，需要独立的搜索密钥。

`.env.demo` 仅由 API 启动配置读取；已有进程环境变量优先。
也支持 `MOONSHOT_API_KEY` / `KIMI_API_KEY` 和 `TAVILY_API_KEY` 别名。
旧 CLI 的 `.env` 与本 Web Demo 的 `.env.demo` 是不同入口，不会隐式混用。
密钥变更后重启启动命令，再点击页面「刷新服务能力」。

```powershell
python -m uv run python -m scripts.run_demo --check
```

此检查仅输出配置是否存在，不显示密钥、不调用付费服务，也不证明余额或权限有效。
当两项均配置后，「在线 API」允许输入自己的问题，例如：

> 比较 ReAct 与 Plan-and-Execute 在研究型 Agent 中的优缺点，并给出原始资料引用。

选择中文或英文报告及预算后提交。模型生成计划和搜索 Query，Tavily 检索，
安全抓取器读取网页，R1 筛选证据，模型生成带引用的报告。

Kimi 默认采用独立的 `kimi-instant` 适配器，支持 `kimi-k2.5` / `kimi-k2.6`：
关闭思考模式、使用供应商固定采样设置，且不声称支持随机种子重现。
有效温度与 seed 在缓存和审计前规范化；用量保留缓存 Token。
参数依据 [Kimi K2.6 文档](https://platform.kimi.com/docs/guide/kimi-k2-6-quickstart)
和 [Chat Completions API](https://platform.kimi.com/docs/api/chat)；搜索接口见
[Tavily Search API](https://docs.tavily.com/api-reference/endpoint/search)。

其他兼容 Chat Completions 的服务可设置 `MODEL_PROVIDER=openai-compatible`，
并显式配置它的 `MODEL_BASE_URL`、`MODEL_ID` 和密钥；并不保证所有厂商扩展协议都兼容。

## 边界与排错

- 在线模式目前演示 `baseline-v1 / P1 / R1`；离线支持 `research-v1 / P1 / R1`。
  P2/R2 优化、模型训练和正式 Benchmark 结论不在本次 Demo 交付中。
- 在线本地 R1 使用轻量词项哈希相似度（含中文二元词项），无需下载模型。
  它不是神经语义嵌入，也不代表跨语言语义检索或已证明的 Ranker 优化效果。
- 本地未定价在线模式仍限制搜索、页面、Token、重试和时间：低预算为
  4 次搜索、8 页、20,000 Token、180 秒；中预算为 8 次搜索、12 页、40,000 Token、300 秒。
  USD 显示 Unknown，不能据此认为免费或保证美元费用上限。公开服务和 Benchmark 仍要求完整定价。
- `AUTHENTICATION`：检查密钥是否有效、区域/域名是否匹配。`RATE_LIMITED`：检查配额并稍后重试。
- `PLAN_INVALID`：模型计划未通过结构/可执行性校验；缩小问题后重试。
- 网页不可访问或证据不足时可能只得到部分报告；这应显式呈现，不能伪造引用。
- 真正的中断续跑仍有限制；页面上的继续能力以服务器响应为准。
- 本文的接口回归测试使用受控 HTTP 响应，不能替代真实密钥/额度的在线验收。
  Docker/PostgreSQL、公网部署和正式结果封存也不由本地 Demo 运行来证明。
