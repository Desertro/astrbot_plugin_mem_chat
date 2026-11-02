# 旧版使用与技术参考

本文保留原 README 中的介绍、技术说明、安装配置、调用示例与参数清单，便于追溯；内容未按当前实现逐项重写。旧模型示例、视觉配置入口和会话标识格式可能已过时，请勿直接照抄。

当前安装入口见 [首页](../README.md#快速开始)，当前配置字段见 [_conf_schema.json](../_conf_schema.json)。图片处理既涉及可选的 `vision_media` 插件，也支持本插件的独立 `meme_vision` 配置；以实际启用的路径为准。历史更新见 [更新记录](../CHANGELOG.md)，未落地方案见 [设计草案](DESIGN_DRAFT.md)。

以下为原文归档。

# mem_chat 使用说明

**简介**
- 在群聊环境实现定向消息优先关注、专注模式持续参与、读空气自主判断的对话助手。
- 结合向量知识库、记忆与人物印象，自动注入上下文以提升连贯度与个性化。
- 支持图片检索下载（Pixiv、Twitter/X），并在候选给出但未成功下载时触发恢复流程。
- **架构**：采用结构化 JSON 响应，结合向量知识库、记忆与人物印象，自动注入上下文以提升连贯度与个性化。

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
- `affinity_update_require_at`：仅在点名机器人时写入好感变化（默认关闭）；关闭后只要爱丽丝真的发出回复，就会按 LLM 判断更新当前发言人的关系分。
- `affinity_delta_cooldown_sec`：同一用户好感写入冷却（秒）。
- `affinity_decay_enabled / affinity_decay_interval_sec / affinity_decay_step / affinity_decay_idle_sec`：自然回归参数。
- `affinity_llm_signal_max_age_sec`：LLM 输出 affinity 信号的最大有效时长（秒）。
- `affinity_llm_fallback_to_mood`：当本轮 LLM 未输出 affinity 时，是否回退旧的 mood 映射。
- `affinity_new_user_correction_enabled`：是否启用“新遇到用户”系数补正。
- `affinity_new_user_window_events / affinity_new_user_factor_start / affinity_new_user_factor_end`：新用户前 N 次事件的线性系数窗口（默认从 0.8 起步，只缩放幅度，不改变加减方向）。
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


表情包发送会先用向量检索召回候选，再由现有对话模型查看最多 8 张候选原图，结合图中文字、动作和本轮意图选择；全部不合适或核对失败时不强行配图。此步骤要求对话模型支持图片输入，配置项为 `meme_visual_selection_enabled`（默认开启）。准确指定图中文字的请求仍可直接匹配。动图会抽取多帧供识别，发送记录使用本轮实际识图结果，避免沿用错误的旧描述。额外识图通常需要数秒，超时上限为 20 秒。

可通过 `meme_vision` 单独配置 OpenAI 兼容识图模型，用于聊天图片补充描述、保存表情时的识别和发送前看图选择；对话模型与 embedding 模型保持独立。支持显式备用站点，只有 `fallback_enabled=true` 才调用，失败或无贴切候选仍不强行发表情。
