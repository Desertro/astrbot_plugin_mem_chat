# 常驻协议与语义工具路由：设计草案

本文从旧 README 移出，保留当时的方案讨论。文中的推荐接口、配置项、云玩家参与流程与迁移步骤属于设计草案，不代表已经全部实现。当前能力见 [首页](../README.md)，配置以 [_conf_schema.json](../_conf_schema.json) 为准。

## 常驻协议 + 语义工具路由（Spec / Draft）

本节定义两个目标：
1) 将“每次都发送的协议/系统约束”升级为可版本化的常驻协议，减少提示词冗余并提升稳定性。
2) 将群聊管理/搜图/TTS 等非高频能力改为“向量库语义召回 + 门控”的按需注入，降低工具耦合与格式混乱。

### 1. 术语
- **常驻协议 (Resident Protocol)**：一份版本化的 system 级约束（JSON-only、字段约束、互斥规则、群聊规则等），有唯一 `protocol_id`。
- **全量注入**：把协议完整文本注入 system prompt。
- **轻量注入**：只注入 `protocol_id` + 少量关键硬约束（用于节省 token）。
- **语义工具路由 (Semantic Tool Router)**：通过向量召回挑选本轮可能需要的工具子集，再注入 tools schema。
- **门控 (Gating)**：对高风险动作施加“明确触发条件”，避免语义误触。

### 2. 常驻协议（版本化与注入策略）

#### 2.1 协议结构
- `protocol_id`: `mem_chat.protocol.vX`（X 为版本号）
- `protocol_text`: 协议全文（当前 README/LLMClient 中的 JSON-only 规则、互斥规则、群聊规则等）
- `fingerprint`: 对 `protocol_text` 的 hash（用于 drift 检测）

#### 2.2 注入策略
- **首轮/冷启动**：全量注入（保证模型“学会”协议）。
- **稳定轮**：轻量注入（仅保留 JSON-only、必须字段、互斥规则 + `protocol_id`）。
- **漂移检测**：当出现任一情况，立即切回全量注入 N 轮：
  - StructuredResponse 解析失败或字段缺失
  - 输出混入自然语言/Markdown/思维链
  - 违反互斥规则（TTS 与 @/引用/表情冲突等）

#### 2.3 兼容性原则
- 协议升级必须向后兼容：新增字段必须可选；已有字段含义不变。
- 解析以 schema 为准：协议文字仅用于引导模型，不作为程序逻辑判断依据。

### 3. 工具向量索引（建立索引）

#### 3.1 工具文档（被索引对象）
对每个工具生成一条 ToolDoc（用于 embedding）：

```json
{
  "name": "tool_name",
  "description": "人类可读描述",
  "parameters": {"type":"object", "properties":{}},
  "tags": ["search", "admin", "tts"],
  "risk": "low|medium|high",
  "gate": {
    "requires_explicit": true,
    "explicit_patterns": ["禁言", "mute", "/mute"],
    "allow_in_states": ["AT", "FOCUS"],
    "deny_in_states": ["READ_AIR"]
  }
}
```

原则：ToolDoc 必须能从工具自身 schema 与少量静态配置生成，避免写死在对话编排代码里。

#### 3.2 索引存储
- 存储位置：复用 `mem_chat_embeddings.db`。
- 存储方式：在向量表中新增/使用 `type = "tool"` 作为工具索引桶。
- 分组策略：默认 `group_id = "global"`；如需群特化能力，可按 group 分桶。

#### 3.3 索引构建触发
- **启动时**：加载工具集合后构建/更新 tool 索引。
- **热更新时**：当工具集合（name/description/parameters）变化，重建 tool 索引。

### 4. 语义召回与门控

#### 4.1 召回输入
- Query 由“消息块文本 + 最后一条用户文本 + 显式关键词（如搜图关键词）”拼接而成。

#### 4.2 召回输出
- `top_k` 个工具名候选（可带分数）。
- 与 Persona 显式指定工具做并集（Persona 是硬约束）。

#### 4.3 门控规则（防误触）
- **高风险工具（群管：禁言/踢人/@all 等）**：必须满足 `requires_explicit`，即出现明确指令/命令形式；仅语义相近不允许触发。
- **状态门控**：READ_AIR 默认禁止高消耗/高风险工具；AT/FOCUS 允许在门控通过后启用。
- **输出门控**：即使模型产出 control.mute，也必须二次校验“是否显式触发”，否则降级为 reply_only/no_reply。

### 5. 运行时数据流（端到端）
- 输入消息块 → 意图检测（含显式指令识别）
- 决定协议注入模式（全量/轻量/纠偏）
- 语义召回工具子集（tool 索引）→ 门控过滤
- 向 LLM 注入：system(协议) + memory_context + instructions + tools(子集)
- 得到 StructuredResponse → 解析校验 → Action 落地（禁言/@/TTS/图片等）

### 5.1 云玩家参与（外部资讯 + 多模态摘要 → Talker）

适用场景：群聊正在讨论一个 LLM 训练集缺失的特定游戏；机器人需要通过特定网站获取资讯，像“云玩家”一样参与聊天。

#### 5.1.1 两阶段模型（Pre-Pipeline → Talker）
- **Pre-Pipeline（不含人格）**：负责触发、路由、检索、解析与多模态摘要，产出可注入的事实包（ContextPack）。
- **Talker（含人格）**：只消费 ContextPack + 最近对话 + 人格设定，生成结构化 JSON 回复（StructuredResponse）。

目标：把“事实获取”与“群聊表达”解耦，避免人格污染检索 query，同时控制成本与误触发。

#### 5.1.2 模块与职责
- **Trigger（筛选器）**：低成本判断是否进入 Pre-Pipeline（规则/embedding/阈值），避免每条消息都检索。
- **Router（轻量路由器）**：小模型/短上下文，仅输出 RetrievalPlan（要不要查、查什么、去哪查、查多新）。
- **Context Builder（上下文构建器）**：把群聊文本 + 图片 OCR/Caption 压缩成检索 query 与约束。
- **Retrieval Layer（检索层）**：站点白名单 search/fetch/parse，输出标准化证据卡片（EvidenceCards）。
- **Multimodal Summarizer（多模态摘要器）**：把网页证据与图片信息压缩成可注入要点。
- **Domain Memory（领域记忆/向量库）**：存储证据摘要、名词解释、版本变更，按群/按游戏分桶并带时间衰减。
- **Talker（对话模型）**：生成最终回复与控制字段（reply_only/no_reply、images、tts 等）。

#### 5.1.3 数据流（推荐）
- 群聊消息块进入 → Trigger 快筛
- 图片存在时：先 OCR/Caption 得到 `image_notes`
- Router 仅看：最新消息 + `image_notes` + 极短历史摘要 → 产出 `RetrievalPlan`
- 执行 Plan：站点 search/fetch/parse → EvidenceCards → Summarizer 压缩
- 写入/召回 Domain Memory（可异步）
- 组装 `ContextPack` 注入给 Talker → Talker 生成 StructuredResponse

#### 5.1.4 接口契约（Draft）

RouterInput：
```json
{
  "trace_id": "uuid",
  "group_id": "xxx",
  "message": {
    "text": "最新消息块精简文本",
    "images": [{"id":"...", "caption":"...", "ocr":"..."}],
    "mentions": [{"qq":"..."}]
  },
  "dialogue_hint": {
    "topic_candidates": ["游戏名/简称"],
    "short_history_summary": "不超过200字"
  }
}
```

RetrievalPlan：
```json
{
  "version": "plan.v1",
  "actions": [
    {
      "type": "web_search",
      "provider": "site_x",
      "query": "游戏名 + 关键词 + 版本/机制",
      "limit": 5,
      "constraints": {"lang":"zh", "freshness_hours": 168}
    },
    {
      "type": "web_fetch",
      "provider": "site_x",
      "urls": ["https://..."],
      "extract": {"mode":"article", "max_chars": 8000}
    }
  ]
}
```

ContextPack（给 Talker 的事实包）：
```json
{
  "topic": {"game":"xxx", "subtopic":"配装/版本/剧情"},
  "evidence": [
    {
      "title": "...",
      "source_url": "...",
      "published_at": "...",
      "key_points": ["...", "..."],
      "confidence": 0.7
    }
  ],
  "image_notes": ["截图显示的BOSS名是...", "界面数值为..."],
  "domain_memory": ["该游戏xx机制：...", "上个版本改动：..."],
  "limitations": ["信息可能过期", "仅来自站点X"]
}
```

#### 5.1.5 门控与成本上限（建议）
- 站点白名单与限流：避免抓取泛滥与不可控来源。
- READ_AIR 默认不触发高成本检索；AT/FOCUS 在 Trigger 命中后才进入 Pre-Pipeline。
- 每个消息块限定最大抓取次数与摘要次数；超出时降级为“等待更多信息/仅基于已知事实参与”。

### 6. 推荐的最小配置项（WebUI）
- `enable_resident_protocol`: 是否启用常驻协议策略（默认 true）
- `resident_protocol_id`: 当前协议版本号（如 `mem_chat.protocol.v1`）
- `resident_protocol_repair_rounds`: 漂移后全量注入轮数（默认 3）
- `enable_semantic_tool_router`: 是否启用语义工具路由（默认 false，先灰度）
- `semantic_tool_top_k`: 工具召回数量（默认 8~12）
- `semantic_tool_threshold`: 召回阈值（默认 0.30~0.45，按 embedding 模型校准）
- `semantic_tool_always_keep`: 永久保留工具列表（如基础 web_search/fetch_url）

### 7. 迁移顺序（低风险）
- 先只做“工具索引构建 + 召回日志”（不改变注入行为），用于离线评估 top_k 命中率。
- 再灰度启用语义工具路由（先对白名单群/Persona 生效）。
- 最后把群管/TTS 的触发从“工具混用”收敛为 StructuredResponse 的 control/response 字段落地。
