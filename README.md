# mem_chat 使用说明

**简介**
- 在群聊环境实现"被 @ 即回复""专注模式持续参与""读空气自主判断"的对话助手。
- 结合向量知识库、记忆与人物印象，自动注入上下文以提升连贯度与个性化。
- 支持图片检索下载（Pixiv、Twitter/X），并在候选给出但未成功下载时触发恢复流程。
- **架构**：采用结构化 JSON 响应，结合向量知识库、记忆与人物印象，自动注入上下文以提升连贯度与个性化。

## 2026-03-20 更新 (Target Awareness + Chat Time Markers)
- 新增 `【对话对象判断】` 区块，用中性事实描述当前消息最可能是在对谁说话。
- 群聊历史行从单纯的 `发送者: 内容` 升级为 `发送者 -> 对象: 内容`，帮助模型理解 A 在对 B 说话，还是在对机器人/群聊说话。
- 历史上下文会像常见聊天软件一样自动插入时间分隔线，例如 `[时间] 今天 14:23`，帮助模型感知话题是否连续。
- `raw_group_messages` 新增结构化字段，记录 `reply_to_*`、`at_targets_json`、`at_all` 以及 `explicit_target_*`。
- 这些增强只用于上下文理解，不会直接触发真实的 QQ `Reply` 或 `@` 发送。

## 2026-03-13 更新 (Focus 双周期优化)
- `@` 触发时，仍会先执行**小周期奖励**：按 `focus_mode_cycle_at_reward_ratio` 缩短下一次专注回复间隔。
- 同时将**大周期**从“固定时长重置”改为“按剩余时长增量续期”：
  - 新专注时长 = `max(当前剩余时长, focus_duration_sec)` + `增量`
  - `增量` 基于专注最小周期，并受 `focus_duration_step_max_sec` 限幅。
  - 当当前剩余时长已经很高时，增量会自动衰减，避免高频 @ 把专注时长快速堆满上限。

## 2026-01-22 更新 (Poke)
- 支持戳一戳事件作为唤醒信号，视为等价 @ 触发调度。
- 戳一戳会进入消息文本抽取，用于记忆与上下文构建。
- 结构化 JSON 支持 response.pokes 主动戳一戳输出。
- OneBot 适配器可发送 response.pokes 的戳一戳。
- 群管撤回协议扩展支持批量 targets 字段，可一次撤回多条候选消息，并在提示词层要求在给出可疑列表时默认撤回不少于 50% 的条目。
- 常规回复协议对齐结构化多段输出：当存在多人 @ 或多条需要分别回应的上下文时，LLM 可通过 response.segments 输出多条回复，每个 segment 对应一条消息，后端已按 segments 顺序逐条发送。

## 2026-02-05 更新 (Segments Delay)
- 当使用 `response.segments` 分段回复时，后端会在**每条消息发送完成后**按字数添加延迟再发送下一条：
  - 延迟范围：0.3s ~ 2.0s
  - 计数规则：按该段文本去除空白后的字符数计数，最多按 400 字计
  - 近似公式：`delay = clamp(0.3, 2.0, 0.3 + (min(chars, 400) * 1.7 / 400))`

## 2026-02-12 更新 (Impression V3 Stream)
- 人物印象新增 **Impression V3 流式架构**：
  - `read_air` 轮可输出 `impression_delta`（0-2 条可验证短事实）。
  - `focus` 结束轮可输出 `session_impression_summary`（1-2 句总结 + tags）。
  - 两者都会写入新库并做向量嵌入，支持语义召回注入。
- 新增数据库表：
  - `impression_profiles_v3`：每人摘要、标签、置信度、来源。
  - `impression_facts_v3`：可累积事实（evidence_count/last_seen/source）。
  - `impression_embeddings_v3`：profile/fact 向量条目（独立于旧 `memory_embeddings`）。
- 上下文注入优先读取 V3：
  - `【人物印象】` 先取 `impression_profiles_v3 + impression_facts_v3`。
  - `【人物印象-相关人物】` 优先走 `impression_embeddings_v3` 语义召回，再回退旧 IMPR2。
- 旧版“每 50 条消息批处理画像”已默认关闭：
  - 配置 `legacy_periodic_profile_enabled=false`（默认）。
  - 如需兼容旧流程，可开启 `legacy_periodic_profile_enabled=true` 并设置 `legacy_periodic_profile_every_n_msgs`。
- 旧 `impressions / user_profiles / profile_facts / IMPR2(memory_embeddings.type=impression)` 进入兼容读取阶段；当 V3 数据稳定后可按运维窗口清理旧表与旧向量条目。
- 关键配置：
  - `impression_v3_enabled`
  - `impression_stream_enabled`
  - `impression_stream_read_air_delta`
  - `impression_stream_focus_delta`
  - `impression_stream_focus_final`
  - `impression_focus_end_threshold_sec`

## 2026-02-12 更新 (Prompt Context Slim)
- 新增非 `AT` 场景的上下文限流（用于 `focus/read_air`）：
  - `max_injected_memories_non_at`：限制 `【相关记忆】` 注入条数。
  - `max_injected_knowledge_non_at`：限制 `【相关知识】` 注入条数。
  - `history_lines_limit_focus`：限制 `focus` 的 `【历史对话】` 条数。
  - `history_lines_limit_read_air`：限制 `read_air` 的 `【历史对话】` 条数。
  - `history_time_gap_minutes`：历史消息时间差超过该值时插入时间分隔线。
  - `history_time_marker_every_n`：每累计 N 条历史消息补一个时间分隔线。
- 保持 `AT` 场景上下文完整，避免影响显式点名的处理稳定性。
- `【相关记忆】` 增加重复行去重，减少相同语义文本重复注入。

## 2026-02-26 更新 (Raw Message Store)
- 新增结构化原始消息表：`raw_group_messages`（位于 `mem_chat_embeddings.db`）。
- 主要字段：
  - `group_id`
  - `message_id`（主键维度：`group_id + message_id`）
  - `message_seq`
  - `sender_id` / `sender_name`
  - `text`（处理后的文本，含图片转写）
  - `raw_text`（原始 message_str）
  - `reply_to_msg_id / reply_to_sender_id / reply_to_sender_name`
  - `at_targets_json / at_all`
  - `explicit_target_type / explicit_target_id / explicit_target_name`
  - `created_at` / `updated_at`
- 新消息实时写入：`on_message` 流程会并行写入 `raw_group_messages`，用于后续画像、统计、审计、检索扩展。
- 历史回填支持：启动时可将 `memory_embeddings` 中可解析历史一次性回填到 `raw_group_messages`，并写入 `system_meta.raw_group_messages_backfill_v1_done=1` 防止重复全量扫描。
- 新配置项：
  - `raw_messages_backfill_on_startup`（默认 `true`）
  - `raw_messages_backfill_batch_size`（默认 `2000`）
  - `raw_messages_backfill_force`（默认 `false`）

## 2026-02-12 更新 (人物印象按最新参与者聚合)
- 新增“按最新消息参与者聚合人物印象”：
  - 每次处理最新消息时，会提取本轮消息块中出现过的用户并注入到同一个 `【人物印象】` 区块。
  - 每个用户下同时展示：
    - 关系倾向（`affinity/trust/bias`）
    - 人物印象（V3 summary/tags/facts，或旧档案回退）
- 该模式默认开启，并可通过配置调整：
  - `impression_merge_recent_participants`
  - `impression_recent_max_persons`

## 2026-02-09 更新 (State Prompt / no_reply / Quote Stability)
- **状态化核心规则改造**：
  - 核心规则改为运行时注入到 `【人物印象/设定】` 第一行，按 `task_type + mood_state` 选择，不再单独注入 `【当前状态】` 区块。
  - 状态规则外置到 `personas/state_rules.txt`，并支持新增状态（如 `focus:软懦`）。
- **no_reply 消息标记与隔离**：
  - 当 LLM 返回 `control.decision="no_reply"` 时，系统会记录该轮消息 ID，并在后续 `pending` 构建与历史注入中排除这些消息，避免反复被处理。
  - 会话历史中对命中的用户消息写入 `_no_reply=true` 标记，后续读取历史时自动跳过。
- **提示词结构去重**：
  - `【最新消息上下文】` 与 `【需要你优先处理的最新一条】` 增加去重，减少同一条消息重复注入导致的“复读/误判”。
- **引用策略收敛（不强制）**：
  - 引用保持可选，不强制引用最新消息。
  - 提示词新增明确约束：仅在直接回应某条具体消息时填写 `quote_index`；`segments` 场景仅对应段可填 `segment.quote_index`，其余留空。

## 2026-02-10 更新 (Affinity V1)
- 新增独立关系库：`affinity_state` / `affinity_events`，用于管理好感与变化日志。
- 旧 `impressions` 表继续保留为“人物印象来源”，不再作为好感主数据。
- 在提示词上下文新增 `【关系倾向】` 区块（明显喜欢/偏喜欢/中性/偏反感/明显反感），用于控制回复语气强弱。
- `personas/personas.txt` 定义好感反应规则：高好感更主动亲昵和暧昧，低好感更冷淡疏离；好感只调节距离和语气，不覆盖事实、任务与安全边界。
- 新增 mood 驱动的好感更新链路：回复完成后可根据 `mood.state/intensity/text` 写入好感变化（带冷却限频与单次限幅）。
- 新增“自然回归”机制：长时间无互动时，好感会按配置步长缓慢回到中性。
- 好感波动改为“阶段化”：
  - 前 10 次有效变更：允许较大波动（单次最多约 40）。
  - 第 10-49 次：中等波动（单次最多约 10）。
  - 第 50 次后：小幅波动（单次最多约 5）。
  该分段用于实现“前期关系变化快，后期印象成型后变化慢”。

## 2026-02-12 更新 (Affinity LLM-led)
- 好感加减方向改为由 LLM 在结构化 JSON 顶层字段 `affinity` 中给出：
  - `affinity.delta`：本轮应加分/扣分（正=加分，负=扣分）。
  - `affinity.reason`：简短原因。
  - `affinity.target_qq`：可选，默认当前发言人。
- 本地逻辑仅负责“新遇到用户”系数补正（只缩放幅度，不改方向）。
- 新配置项：
  - `affinity_llm_signal_max_age_sec`
  - `affinity_llm_fallback_to_mood`
  - `affinity_new_user_correction_enabled`
  - `affinity_new_user_window_events`
  - `affinity_new_user_factor_start`
  - `affinity_new_user_factor_end`

## 2026-02-10 更新 (Impression Semantic Spec V1)
- 人物印象注入改为分层：
  - `【人物印象】`：优先固定注入“当前发言人”印象（可配置）。
  - `【人物印象-相关人物】`：从 impression 向量库按语义召回相关人物（可配置）。
  - `【关系倾向】`：来自 `affinity_state`，仅作用于语气强弱。
- 语义召回默认开启“活跃用户过滤”：只保留当前发言人和本轮消息中实际出现过的用户（`name/id:`、`<at:id>`、`@id`），避免未参与对话的人物印象被误注入。
- 新增相关人物预算控制：支持人数上限、分数阈值、字符预算，防止 prompt 被人物印象区块挤爆。

## 2026-02-01 更新 (Profile V2)
- 人物印象升级为 **Profile V2**：以 `group_id + user_id(QQ号)` 作为唯一标识，昵称仅用于展示，避免同昵称覆盖与错配。
- 向量印象条目改为 `IMPR2 gid=... uid=... name=... :: ...` 格式，召回时直接解析 uid/name，不再通过 summary 反查映射。
- 新增数据表：`user_profiles`（结构化画像/摘要）与 `profile_facts`（可累积的事实片段）。
- 新增迁移脚本：`scripts/migrate_impressions_v2.py`，可从旧 `impressions` 表迁移到 V2，并可选写入 IMPR2 向量条目（支持 `--dry-run/--group/--force/--skip-embedding`）。

## 2026-02-03 更新 (Profile V3)
- 人物印象升级为 **Profile V3**：在 `profile_json` 中新增 `impression_summary`（短摘要）与 `tags`（短标签），并用更稳定的 `canonical_text` 模板生成可注入文本（`印象/标签/特征/偏好/...`）。
- `user_profiles.version` 写入为 3；旧版 V2 数据仍可读取，注入时会自动规整为自然语言片段。
- 迁移脚本 `scripts/migrate_impressions_v2.py` 已升级为可迁移到 V3：
  - 迁移来源：旧 `impressions` 表、以及 `user_profiles(version<3)`。
  - 向量一致性：可选重算 embedding；否则会将 embedding 置空，运行时首次触发会自动重建。
  - 支持 `--reset-faiss` 清理旧 FAISS 索引文件，强制重建（更一致，但首次召回会更依赖重建进度）。
  - 常用示例：
    - 仅预演统计：`python scripts/migrate_impressions_v2.py --dry-run`
    - 迁移并让运行时重建向量：`python scripts/migrate_impressions_v2.py --skip-embedding --reset-faiss`
    - 迁移并在脚本内重算向量：确保配置中已设置 embedding_*，再运行 `python scripts/migrate_impressions_v2.py`。

## 2026-01-20 更新 (Prompt Placement)
- 群管协议不再注入系统上下文，改为追加在【回复要求】之后，避免与知识库内容混排。
- 群管协议仅从 admin_protocols/ 目录加载（与 knowledge 目录并列），避免与普通知识混用。

## 2026-01-21 更新 (Group Admin)
- 群管协议新增消息撤回控制字段说明，并支持通过 JSON 控制执行撤回。

## 2026-01-21 更新 (Knowledge Sync)
- 修复知识库哈希读取未 await 导致每次启动都重复增量入库的问题。
- FAISS 迁移完成后写入标记，避免重复迁移带来的额外写入。


## 2026-01-21 更新 (Impression Prompt)
- 人物印象如为 JSON 存储，注入提示词前会自动规整为自然语言片段。
- 增加人物印象整理流程日志，便于确认定期压缩是否生效。

## 2026-01-12 更新 (Decoupling & Media Reliability)
- **架构解耦与推送优化**：重构了图片推送流程。现在 `mem_chat` 承担了统一的图片下载与缓存职责。外部插件（如 `trending_images`）只需提供原始 URL，由 `mem_chat` 在发送前自动完成本地化。
- **X/Twitter 图片深度适配**：在 `llm_client.py` 中增加了对 X/Twitter 图片链接的自动解析逻辑（如自动添加 `format=jpg&name=large`），确保抓取的是最高质量原图。
- **多级缓存机制**：图片下载增加 MD5 缓存校验。同一图片（URL）在 `data/temp/mem_chat_images` 下仅下载一次，显著提升二次发送速度并节省流量。
- **WSL2 代理支持**：同步适配了宿主机代理自动检测逻辑，确保在复杂网络环境下（如抓取 X 图片）的连接稳定性。

## 2026-01-06 更新 (Stability & Performance)
- **并发安全性优化**：引入 `asyncio.Lock` 保护 SQLite 写入操作，彻底解决多线程/多协程下的 `database is locked` 风险。
- **异步性能提升**：将耗时的阻塞操作（SQLite 写入、FAISS 索引保存、PIL 图像处理）全部迁移至线程池（`asyncio.to_thread`），显著降低插件主线程阻塞概率。
- **安全性增强 (SSRF 防护)**：为图片下载逻辑增加 SSRF 防护，自动拦截内网、回环及保留地址。
- **代码健壮性修复**：修复了 `main.py` 中部分异步方法（如 `set_meta` / `delete_meta`）未被正确 await 的问题。

## 技术细节 (Librarian Agent Note)

### 并发控制
插件使用 `asyncio.Lock` 确保对 SQLite 数据库的串行写入。在 `database.py` 中，所有执行 `INSERT` 或 `UPDATE` 的方法均被锁保护，避免了 SQLite 在并发写入时的锁定错误。

### 异步非阻塞设计
为了不阻塞 AstrBot 的主事件循环，插件采用了“厚异步，薄同步”的策略：
- **CPU 密集型任务**：如 `FAISS` 索引的序列化与反序列化、图像的缩放与转换，均通过 `asyncio.to_thread` 委派给线程池。
- **IO 密集型任务**：数据库操作和网络请求均使用异步驱动。

### 安全与资源
- **SSRF 检查**：下载图片前会解析 URL 并检查 IP 地址是否属于私有或保留网段。
- **自动清理**：使用 `TemporaryFileContext` 确保 `temp/` 目录下的中间产物在任务结束后自动删除。

## 安装与配置

## 2026-01-23 更新 (Vision Split)
- 视觉媒体能力（抽图、压缩/GIF 抽帧、视觉 LLM 描述、图片描述缓存）已拆分为独立插件 `vision_media`（目录：`astrbot_plugin_vision_media`）。
- `mem_chat` 会优先调用 `vision_media`：生成 `text_with_images`（把图片替换为 `[图片内容: ...]` 或 `[图片]`）并抽取 `image_urls` 供多模态输入使用；若未安装/未启用则自动降级，不影响纯文本聊天。
- 视觉相关配置请在 `vision_media` 插件内配置（WebUI 中进入该插件配置）：`vision_llm.*`、`vision_image_cache_*`、`mem_chat_llm_image_*`。

## 2026-01-24 更新 (QQ 引用索引增强)
- 【最新消息上下文】中的 `[REF:n]` 候选池现在覆盖：最近消息、当前最新消息，以及这些消息里引用到的消息（Reply 目标），用于更稳定的引用/撤回/群精华等动作目标选择。
- `quote_index` / `control.recall.quote_index` / `control.set_group_essence.quote_index` 统一使用同一套候选映射，避免“引用对了但撤回/精华错了”。如遇套娃引用，仍可用 `use_replied_target=true` 兜底。
- 新增配置项：`ref_context_window`、`ref_context_include_referenced`、`reply_targets_*`、`reply_summaries_*`（用于控制候选窗口与引用缓存清理）。

## 2026-01-24 更新 (Prompt 结构精简 & 群管协议注入)
- 提示词结构精简：移除重复的 JSON 强约束块与 AT 双重历史注入，减少 token 并降低旧话题干扰。
- 群管协议不再做语义召回：改为确定性注入（`admin_protocols_inject_mode=auto/off/always`）。如需实际执行群管操作，需显式开启 `admin_actions_enabled=true`（默认关闭）。
- 转发/卡片消息的上下文展示已做脱敏：不再把 `Json` 消息段的原始 payload 直接塞进 [REF]，会以“(不可见) [转发消息]/[卡片消息]”占位，避免模型误把其内容当作可读文本并胡乱引用。
- 新增“心境变量”：LLM 每次输出顶层 `mood.text`（20-30 字），下一轮会注入 `【上一轮心境】` 供模型延续情绪（`mood_enabled=true/false`）。

## 2026-02-04 更新 (Moderation Score)
- 撤回策略改为“打分触发”：要求 LLM 在 `control.recall.score` 或各 `targets[].score` 中给出 0-100 分的黄暴/敏感评分，仅当分数达到阈值才会执行撤回。
- 新增配置：`recall_nsfw_score_threshold`（默认 99）。分数低于阈值时不撤回；若只给出部分候选的高分项，仅撤回这些高分条目。
- 管理协议文本已更新：在 `admin_protocols/admin_protocols.txt` 的【消息撤回指令】中补充了 `score` 字段及触发规则说明。

## 2026-02-15 更新 (工具白名单限制)
- 新增 `llm_tools_restrict_enabled`（默认开启）：对 `mem_chat` 的 Function Calling 做白名单限制。
- 新增 `llm_tools_allowed_patterns`：按“工具名包含关键词”放行；为空时不允许任何工具。
- 这可以避免模型误调用高风险工具；若当前没有匹配到的工具，`mem_chat` 本轮将不会执行任何 tool_call。

### 安装
1. 将 `astrbot_plugin_mem_chat` 目录放置在 AstrBot 的 `data/plugins/` 目录下。
2. 重启 AstrBot，插件将自动加载并初始化数据库。

### 配置
请在 AstrBot WebUI 中配置以下核心项：
- `custom_llm_base_url`: 你的 LLM API 地址。
- `custom_llm_api_key`: API 密钥。
- `llm_model`: 使用的模型（如 `gpt-4o` 或 `gemini-1.5-pro`）。
- `enable_llm_tools`: 是否启用工具调用（建议关闭以节省 Token）。
- `profile_v2_enabled`: 是否启用人物印象 V2（建议开启）。

## 2026-01-05 更新 (Testability)
- `llm_client_parts` 的 `ImageService` / `TtsService` 支持依赖注入（`get_day` / `get_cwd` / `get_data_path`），可在无 AstrBot 运行时下独立实例化与单测。
- 新增单元测试：`/tests/test_mem_chat_llm_client_parts.py`（覆盖图片 key、每日去重、视觉载荷 data-url 生成、wav 合并）。
- `MemChatState` 支持依赖注入（`now` / `rand`），可独立测试焦点状态、去重逻辑、pending 队列与中断标记。
- 新增单元测试：`/tests/test_mem_chat_state.py`。
- 修复连续回复：当平台事件缺失 `sender_id` 时直接忽略，避免 bot 自身消息被当作新消息入队。
- 新增单元测试：`/tests/test_mem_chat_sender_ignore.py`。
- `main.py` 兼容以脚本方式导入（无包上下文）以便本地单测运行。
- 增强调度器异常日志：任务执行失败时输出完整堆栈与任务上下文，便于定位线上报错。
- 修复 LLM 工具调用兼容性：支持执行 `@llm_tool` 异步生成器（yield 多条 MessageEventResult）的工具（如 `reminder`），避免 `object async_generator can't be used in 'await' expression`。
- 新增配置项 `enable_llm_tools`（默认关闭）：关闭后不再向对话 LLM 注入 tools schema，也不会执行任何 tool_call，以减少提示词长度与 token 消耗。
- 修复专注周期竞态：当对话任务已入队/处理中时，不再提前启动新的专注/读空气计时；专注计时以“上次成功发送回复时间”为基准。
- 支持结构化 JSON 多段回复：提示词允许使用 `response.segments` 输出多条消息（适配多人上下文），并增加 JSON-only 强约束以提高解析稳定性。
- 恢复 GIF 处理：当输入图片为动图时，自动抽帧为多张 JPEG 供视觉模型使用，并按总预算压缩以控制体积。

## 2026-01-03 更新 (State Machine Refactor)
- **架构重构：状态机 (State Machine)**：
  - **核心状态**：系统现仅运行在两种状态之一：
    - **READ_AIR**（默认）：静默观察模式，仅在心跳检测满足条件时被动回复，**禁止**任何主动搜图。
    - **FOCUS**（专注）：活跃对话模式，允许规则驱动的轻量级搜图，维持对话节奏。
  - **事件驱动**：`AT` 指令不再作为一种“任务类型”，而是作为**高优先级事件**。收到 AT 会立即打断当前任务，强制切换/刷新至 `FOCUS` 状态，并触发 Plan B（LLM 规划）级搜图。
- **搜图逻辑收敛与权限隔离**：
  - **Plan B (LLM 规划)**：仅在 **AT 事件** 中触发。LLM 会分析整个消息块上下文来规划搜图关键词（高精度，高消耗）。
  - **Plan A (规则匹配)**：仅在 **FOCUS 状态** 下触发。使用正则提取关键词进行快速检索（低延迟，维持热度）。
  - **READ_AIR**：彻底禁止搜图，防止在挂机模式下因误判（如表情包）而发送图片。
  - **误触优化**：意图识别正则已更新，明确排除了“发表情包”等非搜图指令，并自动忽略消息块中的图片组件。
- **Bug 修复**：
  - **双重回复**：修复了 AstrBot 核心与插件同时响应 AT 的问题（通过 `event.should_call_llm(False)` 抑制核心）。
  - **旧消息重复处理**：修复了“决定不回复”时未清理消息缓存的问题。现在，LLM 的“静默”决策也被视为一次成功的处理，会正常清理积压消息。
  - **旧话题回跳**：修复了在队列/专注周期中因消息积压导致“突然回复几分钟前/十几分钟前话题”的问题。现在会对待处理消息做过期裁剪/丢弃，并按“本轮消费条数”清理，避免残留反复注入上下文。
  - **JSON 提取鲁棒性**：增强了从 LLM 输出中提取 JSON 的逻辑。现在能有效过滤思维链（Chain of Thought）、Markdown 标记以及 JSON 前后的杂质文本，支持从多个潜在块中精准定位有效的结构化响应。
  - **强制 JSON 输出（可选）**：新增配置项 `force_llm_json_output`。启用后会在自定义 LLM 请求层尝试开启 JSON-only 模式（OpenAI 兼容：`response_format=json_object`；Gemini：`responseMimeType=application/json`），以减少思维链/解释性文字干扰（部分接口不支持会报 400，请关闭）。

## 2025-12-31 更新
- **逻辑简化与对齐**：
  - **消息块级搜图计划**：在收到 AT 指令触发搜图意图时，系统现在会将**整个消息块（Message Block）**发送给 LLM 进行关键词提取（Pixiv Search Plan）。这允许 LLM 结合前后文（如 A、B 的对话内容）来更精准地理解用户的搜图指令。
  - **精准意图识别**：虽然发送给 LLM 的是整个消息块，但系统在识别搜图意图时依然会精准扫描块内的 AT 消息。
- **回复节奏优化**：
  - **避免连续回复**：修正了 `Read Air` 和 `AT` 任务完成后立即触发后续回复的问题。现在只有在 `FOCUS` 周期任务完成后，如果仍有积压消息才会自动触发下一个周期。`Read Air` 和 `AT` 完成后将等待下一条新消息来启动新的 `FOCUS` 延迟周期。
  - **周期稳定性**：确保了“回复完等待一个完整延迟周期再提交 LLM”的逻辑，消除了任务链条过于紧密导致的连续刷屏。
- **静默失败处理与消息保留**：
  - 移除了 LLM 超时或负载高时的回退文本提示（“上游模型当前负载较高...”），此类情况现在将静默处理。
  - **超时消息保留**：修正了 LLM 超时或失败时消息被错误标记为已读的问题。现在，只有在 LLM 成功生成回复后才会清理已处理的消息。如果发生超时或调用失败，消息将保留在待处理队列中，并在下一个周期（如心跳触发）中重新尝试处理，确保不会遗漏用户指令。
- **分场景处理**：
  - **读空气模式 + AT**：立即触发回复并进入专注模式。
  - **专注模式 + AT**：应用周期奖赏（缩短 5s 延迟），刷新持续时间，并执行消息块级搜图计划。
  - **专注模式（无 AT）**：按正常的专注周期处理消息块，仅使用规则进行基础搜图识别。
- **强制结构化图片输出**：统一要求模型通过 `response.images` 数组字段返回图片 URL。
- **任务调度优化**：优化了任务插队与中断机制，确保 @at 消息的最高优先级。

## 2025-12-30 更新
- **入站消息去重**：同时按 `message_id` 与文本指纹做短窗去重，避免平台重投递导致重复触发。
- **调整专注模式机制**：
  - 非专注模式下，收到 AT 消息立即触发回复。
  - 专注模式下，AT 消息具有高优先级，触发即时回复周期，并应用回复周期奖赏。
  - **动态周期算法**：
    - 基准周期：由 WebUI 配置项 `focus_mode_cycle`（专注模式周期）控制。
    - 频率抑制：每次在 **FOCUS（无 AT）** 下完成一次回复后，下次专注周期将被设置为 `focus_mode_cycle + focus_mode_cycle_after_reply_add`（例如 30s → 50s），防止无人 @ 时刷屏。
    - @ 奖励：收到 @ 后，专注周期按比例减少 `focus_mode_cycle * focus_mode_cycle_at_reward_ratio`。
      - 若未配置 `focus_mode_cycle_at_reward_ratio`，则使用 `focus_mode_cycle_at_reduce / focus_mode_cycle` 推导比例（例如 20/30 = 0.666667）。
    - @ 续期：收到 @ 后，专注持续时长按“剩余时长 + 增量”刷新，而不是直接重置为固定值。
    - 下限：专注周期不会低于 `focus_mode_cycle_min`（默认 10s）。
    - **周期同步**：专注模式下会等待上一次 LLM 返回并发送后，再进入下一次计时，确保回复不重叠。
- **配置项精简**：
  - 移除了冗余的 `task_timeout_sec_at` 配置项，AT 任务将不再受此超时限制（因为其具有最高优先级且立即执行）。
- **优化图片处理逻辑**：
  - 过滤消息文本中的 `[图片]` 占位符，防止 LLM 因看到该字符串而误触发搜图工具。
- **修复中断竞态**：
  - 修正中断触发逻辑：仅在新的 @ 消息到来时中断当前生成，修复专注模式下误判中断导致消息“没发出去”的问题。
- **任务调度优化**：
  - AT 支持 debounce（连续 @ 只触发一次）。
  - 读空气触发增加消息量阈值：支持 `decision_msg_count_min/max`。
  - 新增任务等待超时：AT/FOCUS/READ_AIR 支持 `task_timeout_sec_*`。
- **其他改进**：
  - 新增群白名单 `group_whitelist`。
  - 视觉辅助增强：支持图片压缩与动图首帧提取，视觉 LLM 格式对齐。

**主要特性**
- **主动图片检索**：
  - **AT 模式（B 方案）**：正则识别到搜图语义后，先调用一次 LLM 提炼 `{keywords,count}`，再对多个关键词分别进行 Pixiv 搜索并注入候选；最终由模型在 `response.images` 中返回需要发送的图片 URL 列表。
  - **FOCUS 模式（A 方案）**：为控制性能，仅使用规则清洗出的关键词进行 Pixiv 预检索并注入候选（可选用自定义 LLM 进一步精炼关键词，取决于配置）。
  - **READ_AIR**：默认不进行 Pixiv 预检索以保证性能。
- **结构化 JSON 响应**：
  - 输出统一的结构化 JSON 格式，包含普通回复、表情、TTS 调用、引用等。
  - **emotion 对齐**：emotion 字段与 meme_chat 插件完全兼容，支持 `[happy]`、`(sad)`、`&&angry&&` 等格式。
- **图片处理优化**：
  - **本地下载后发送**：图片完全下载到本地后才发送消息，发送后立即删除本地文件。
  - **自动恢复**：未找到图片或 404 错误时，尝试从上下文候选 URL 中恢复。

**知识库调用逻辑 (RAG)**
1. **自动化同步 (增量更新)**
   - 插件自动监听 `knowledge/` 目录下的 `.txt` 和 `.md` 文件。
   - 在启动或执行 `/mem_chat reload` 时，插件会对文件内容计算 MD5 哈希。
   - 若哈希变化，则执行**增量更新**：自动对比数据库与本地文件的差异，仅对新增的文本块（以双换行 `\n\n` 分隔）进行向量化入库，并删除已不存在的文本块。
   - 默认使用配置中的 Embedding 模型（如 `text-embedding-3-small`）进行向量化。

2. **多维语义召回**
   - 每次处理消息时，系统会将当前消息 (Query) 向量化，并从向量数据库 (`mem_chat_embeddings.db`) 中进行多维度召回：
     - **相关知识 (`knowledge`)**: 从 `knowledge/` 目录导入的全局知识库中检索最相关的 Top 5 片段。
     - **长期记忆 (`memory`)**: 从历史群聊记录中检索相关的往期对话。
     - **人物印象 (`impression`)**: 检索关于当前对话参与者的结构化印象摘要。

3. **上下文注入**
   - 召回的信息会被格式化为以下结构块并注入到 System Prompt 中：
     - `【人物印象/设定】`: 包含 `personas/personas.txt` 中的全局设定。
     - `【相关知识】`: 注入检索到的本地知识库片段。
     - `【相关记忆】`: 注入检索到的历史记忆。
     - `【人物印象】`: 注入当前发言人的个性化印象。
     - `【历史对话】`: 注入最近 N 条消息作为背景参考，保持一定连贯性。
   - LLM 将优先回复【当前最新对话】；【历史对话】仅作参考，避免被旧话题/旧 @ 干扰当前回复。

**工作原理**
- 模块职责：
  - `MemChat`：插件入口、指令注册、文件增量导入、消息入口。
  - `Scheduler`（`src/core/`）：AT/FOCUS/READ_AIR 周期与优先队列管理、任务调度。
  - `LLMClient`（`src/`）：对外门面（Facade），负责结构化提示词、JSON 解析、LLM 调用与多媒体编排；图像/TTS/自定义 LLM 细节拆到 `llm_client_parts/`。
  - `DatabaseManager`（`src/db/`）：SQLite 数据层与向量召回（知识、记忆、印象）。
  - `vision_media`（外部插件）：图片抽取/压缩/GIF 抽帧/视觉描述与缓存服务。
  - `MemChatState`（`src/core/`）：运行态管理（专注模式、待处理消息、会话状态等）。

- 目录结构（树形）：
  ```
  astrbot_plugin_mem_chat/
  |
  |____ main.py                 # 插件入口、指令注册、生命周期接入点
  |____ _conf_schema.json       # WebUI 配置 schema
  |____ README.md
  |
  |____ src/
  |     |
  |     |____ core/
  |     |     |____ scheduler.py   # AT/FOCUS/READ_AIR 周期与队列调度
  |     |     |____ state.py       # 运行态（pending/focus/heartbeat 等）
  |     |     |____ models.py      # ReplyTask / DummyEvent 等数据结构
  |     |
  |     |____ db/
  |     |     |____ database.py    # SQLite + 向量召回（knowledge/memory/impression）
  |     |
  |     |____ llm/
  |     |     |____ prompt_builder.py
  |     |     |____ response_schemas.py
  |     |
  |     |____ llm_client.py        # LLMClient 门面：提示词/解析/编排/调用
  |     |____ llm_client_parts/
  |     |     |____ image_service.py  # 图片 key/每日去重/视觉载荷压缩
  |     |     |____ tts_service.py    # TTS 清洗/分句/替换音频/合并 wav
  |     |     |____ custom_llm.py     # 自定义 LLM 请求（OpenAI/Gemini）
  |     |     |____ logger_proxy.py
  |     |
  |     |____ database.py          # 兼容转发：DatabaseManager
  |     |____ scheduler.py         # 兼容转发：Scheduler
  |     |____ models.py            # 兼容转发：ReplyTask/DummyEvent
  |     |____ state.py             # 兼容转发：MemChatState
  |
  |____ knowledge/              # 本地知识库（增量入库）
  |____ personas/               # 人物设定
  |____ scripts/                # 维护脚本
  |____ temp/                   # 图片临时目录（发送后清理）
  |____ faiss/                  # 可选的 FAISS 索引持久化文件
  |____ mem_chat_embeddings.db  # SQLite 向量库（运行时生成）
  ```

- 兼容层（历史路径转发）：
  - `src/scheduler.py` / `src/database.py` / `src/models.py` / `src/state.py` 仅做薄转发导入，避免旧引用路径在重构后失效。

- 运行时数据（插件目录下）：
  - `mem_chat_embeddings.db`：SQLite 向量库（knowledge/memory/impression）。
  - `faiss/*.faiss`：可选的 FAISS 索引持久化文件（按 group/model/type 分桶）。
  - `knowledge/`：本地知识库文件（启动或 `#mem_chat reload` 时增量入库）。
  - `personas/personas.txt`：全局人物设定注入源。
  - `temp/`：图片下载临时目录（发送后清理）。

**结构化响应格式**
- **完整 JSON 结构**：
  ```json
  {
    "response": {
      "text": "直接回复用户的文本内容",
      "emotion": "[happy]",  // 对齐 meme_chat 插件格式
      "images": ["https://..."], // 可选：图片 URL 列表（推荐用于 AT 模式搜图）
      "tts": {
        "enabled": true,
        "voice": "default",
        "speed": 1.0
      },
      "mentions": []  // 引用和艾特信息
    },
    "control": {
      "decision": "reply_only" // 或 "no_reply"
    }
  }
  ```
- **容错说明**：当模型输出扁平化键（如 `control.decision`、`response.text`、`response.tts.enabled`）时，会在解析阶段自动归并为嵌套结构；仍建议优先输出标准嵌套结构以避免误判。

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

**使用方法**
- 群聊中直接 @ 机器人，或在专注/读空气周期触发下由插件自动回复。
- 图片请求：系统会自动识别搜图意图并提供候选图片，模型只需在回复中包含图片 URL，系统即可自动解析下载并发送。
- 指令：
  - `/mem_chat help|clear|stats|focus|forget|reload`
  - 管理：`/mc_add [type] <content>`、`/mc_list [type] [limit]`

**外部内容人格分享（供其他插件调用）**
- `MemChat.persona_share(target_umo, items, intro=None, query=None)`：让 mem_chat 以当前人物设定口吻生成分享文案并发送（要求 items 至少包含可用的 `image_url` 或 `image` 字段）。
- `target_umo` 会话格式：`platform:group:群号`（示例：`qq:group:123456`）。

**常用配置项（WebUI）**
- `group_whitelist`：群聊白名单（留空=不过滤）。
- `decision_msg_count_min / decision_msg_count_max`：读空气触发的消息量区间。
- `read_air_enable`：是否启用读空气模式（关闭后仅 @/专注触发回复）。
- `task_timeout_sec_at / task_timeout_sec_focus / task_timeout_sec_read_air`：任务队列等待超时（秒）。
- `stale_pending_max_age_sec`：待处理消息的最大保留时间（秒，超时会被裁剪/丢弃，避免旧话题回跳）。也可按任务覆盖：`stale_pending_max_age_sec_focus` / `stale_pending_max_age_sec_read_air`。
- `focus_pending_max_msgs / at_pending_max_msgs`：FOCUS/AT 任务拼接上下文时最多保留的待处理条数（避免上下文过长）。
- `history_time_gap_minutes`：历史上下文中插入时间分隔线的最小间隔（分钟）。
- `history_time_marker_every_n`：历史上下文中每累计多少条消息补一个时间分隔线。
- `custom_llm_format / llm_format`：对话 LLM 载荷格式（OpenAI / Google Gemini）。
- 视觉相关配置已迁移到 `vision_media` 插件内配置。
- `focus_mode_cycle`：专注模式基准回复周期（秒）。
- `focus_cycle_min_sec / focus_cycle_max_sec`：专注回复周期上下限（秒）。
- `focus_cycle_step_max_sec`：专注回复周期单次最大变动（秒）。
- `focus_duration_min_sec / focus_duration_max_sec`：专注持续时长上下限（秒）。
- `focus_duration_step_max_sec`：专注持续时长单次最大变动（秒）；同时作为 `@` 触发时单次续期增量的上限。
- `focus_mode_cycle_at_reward_ratio`：@ 奖励比例（例如 0.666667）。
- `focus_mode_cycle_at_reduce`：用于推导奖励比例的旧参数（比例=该值/`focus_mode_cycle`），不配置 ratio 时生效。
- `focus_non_at_use_decision_prompt`：专注周期内若没有任何消息显式 @ 机器人，则改用读空气判定模板（允许输出 no_reply）。
- `recall_nsfw_score_threshold`：黄暴/敏感内容撤回分数阈值（0-100，仅分数达到阈值才撤回）。
- `reply_quote_max_age_sec`：普通回复可引用消息的最大年龄（秒，默认 1800）；超时会忽略 quote，避免胡乱引用很久前消息。
- `mood_negative_ttl_cap_enabled`：是否启用负面心境 TTL 上限。
- `mood_persistent_states`：常驻心境状态列表，默认 `["色色"]`；列表内状态会忽略 LLM 输出的 `mood.ttl_sec`，不因 TTL 自动回到中性。
- `mood_ttl_cap_sarcasm_sec / mood_ttl_cap_angry_sec / mood_ttl_cap_timid_sec`：仅对 `讥讽/生气/软懦` 的最大持续时长（秒）。
- `affinity_enabled`：启用关系好感系统。
- `affinity_update_require_at`：仅在点名机器人时写入好感变化（默认开启）。
- `affinity_delta_cooldown_sec`：同一用户好感写入冷却（秒）。
- `affinity_decay_enabled / affinity_decay_interval_sec / affinity_decay_step / affinity_decay_idle_sec`：自然回归参数。
- `affinity_llm_signal_max_age_sec`：LLM 输出 affinity 信号的最大有效时长（秒）。
- `affinity_llm_fallback_to_mood`：当本轮 LLM 未输出 affinity 时，是否回退旧的 mood 映射。
- `affinity_new_user_correction_enabled`：是否启用“新遇到用户”系数补正。
- `affinity_new_user_window_events / affinity_new_user_factor_start / affinity_new_user_factor_end`：新用户前 N 次事件的线性系数窗口（只缩放幅度，不改变加减方向）。
- `affinity_poke_bonus_force / affinity_poke_min_delta`：对机器人“戳一戳”视为强正向互动，强制追加加分并设置最小正增量。
- `affinity_poke_ignore_cooldown`：戳一戳时是否忽略通用冷却（默认开启，保证互动反馈及时）。
- `affinity_poke_preapply_before_prompt`：戳一戳消息先写入好感，再组织 prompt；默认开启。
- `impression_semantic_enabled`：是否启用人物印象向量语义召回。
- `impression_active_only`：是否只注入“本轮活跃用户”印象（默认开启，建议保持）。
- `impression_sender_pinned`：是否固定注入当前发言人印象（默认开启）。
- `impression_stream_focus_delta`：专注对话中也持续写入人物印象增量；开启后更容易及时记住外号、偏好、互动边界。
- `impression_related_max_persons`：语义召回相关人物数量上限。
- `impression_related_score_min`：语义召回最小分数阈值。
- `impression_token_budget`：`【人物印象-相关人物】` 区块字符预算（0 为不限制）。

**消息处理流程**
1. 入口 `on_message`：过滤私聊，抽取文本与图片，更新状态。
2. 触发周期：
   - AT：收到 @ 消息后触发。
   - FOCUS：处于专注模式时定期触发。
   - READ_AIR：心跳触发，根据消息量判断是否回复。
3. 任务调度：`Scheduler` 将任务入队并按优先级处理。
4. LLM 调用：`LLMClient` 组装提示词（含人物设定、相关记忆、历史对话等），调用 LLM 获取结构化响应。
5. 结果处理：解析 JSON，下载图片，发送消息链，清理临时文件。
