# mem_chat 使用说明

**简介**
- 在群聊环境实现"被 @ 即回复""专注模式持续参与""读空气自主判断"的对话助手。
- 结合向量知识库、记忆与人物印象，自动注入上下文以提升连贯度与个性化。
- 支持图片检索下载（Pixiv、Twitter/X），并在候选给出但未成功下载时触发恢复流程。
- **架构**：采用结构化 JSON 响应，结合向量知识库、记忆与人物印象，自动注入上下文以提升连贯度与个性化。

## 2026-01-06 更新 (Stability & Performance)
- **并发安全性优化**：引入 `asyncio.Lock` 保护 SQLite 写入操作，彻底解决多线程/多协程下的 `database is locked` 风险。
- **异步性能提升**：将耗时的阻塞操作（SQLite 写入、FAISS 索引保存、PIL 图像处理）全部迁移至线程池（`asyncio.to_thread`），显著降低插件主线程阻塞概率。
- **安全性增强 (SSRF 防护)**：为图片下载逻辑增加 SSRF 防护，自动拦截内网、回环及保留地址。
- **资源管理优化**：实现基于 `asynccontextmanager` 的临时文件清理机制，确保图片发送后立即释放磁盘空间。
- **代码健壮性修复**：修复了 `main.py` 中部分异步方法（如 `set_meta` / `delete_meta`）未被正确 await 的问题。

## 技术细节 (Librarian Agent Note)

### 并发控制
插件使用 `asyncio.Lock` 确保对 SQLite 数据库的串行写入。在 `database.py` 中，所有执行 `INSERT` 或 `UPDATE` 的方法均被锁保护，避免了 SQLite 在并发写入时的锁定错误。

### 异步非阻塞设计
为了不阻塞 AstrBot 的主事件循环，插件采用了“厚异步，薄同步”的策略：
- **CPU 密集型任务**：如 `FAISS` 索引的序列化与反序列化、图像的缩放与转换，均通过 `asyncio.to_thread` 委派给线程池。
- **IO 密集型任务**：数据库操作和网络请求均使用异步驱动。

### 安全与资源
- **SSRF 检查**：在 `media/manager.py` 中，下载图片前会解析 URL 并检查 IP 地址是否属于私有或保留网段。
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
- 新增群聊重复图片视觉描述缓存（非 @ 消息）：优先复用缓存的 `[图片内容: ...]`，并发场景下同一图片仅触发一次视觉 LLM 调用（配置：`vision_image_cache_*`）。

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
  - `MediaManager`（`src/media/manager.py`）：图片下载与视觉辅助。
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
  |     |____ media/
  |     |     |____ manager.py     # 图片下载与视觉辅助
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
  - `knowledge/`：本地知识库文件（启动或 `/mem_chat reload` 时增量入库）。
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

**使用方法**
- 群聊中直接 @ 机器人，或在专注/读空气周期触发下由插件自动回复。
- 图片请求：系统会自动识别搜图意图并提供候选图片，模型只需在回复中包含图片 URL，系统即可自动解析下载并发送。
- 指令：
  - `/mem_chat help|clear|stats|focus|forget|reload`
  - 管理：`/mc_add [type] <content>`、`/mc_list [type] [limit]`

**常用配置项（WebUI）**
- `group_whitelist`：群聊白名单（留空=不过滤）。
- `decision_msg_count_min / decision_msg_count_max`：读空气触发的消息量区间。
- `task_timeout_sec_at / task_timeout_sec_focus / task_timeout_sec_read_air`：任务队列等待超时（秒）。
- `stale_pending_max_age_sec`：待处理消息的最大保留时间（秒，超时会被裁剪/丢弃，避免旧话题回跳）。也可按任务覆盖：`stale_pending_max_age_sec_focus` / `stale_pending_max_age_sec_read_air`。
- `focus_pending_max_msgs / at_pending_max_msgs`：FOCUS/AT 任务拼接上下文时最多保留的待处理条数（避免上下文过长）。
- `custom_llm_format / llm_format`：对话 LLM 载荷格式（OpenAI / Google Gemini）。
- `vision_llm.format`：视觉 LLM 载荷格式（OpenAI / Google Gemini）。
- `focus_mode_cycle`：专注模式基准回复周期（秒）。
- `focus_mode_cycle_after_reply_add`：FOCUS（无 AT）回复后，下次周期额外增加的秒数（默认 20）。
- `focus_mode_cycle_min`：专注周期下限（默认 10）。
- `focus_mode_cycle_at_reward_ratio`：@ 奖励比例（例如 0.666667）。
- `focus_mode_cycle_at_reduce`：用于推导奖励比例的旧参数（比例=该值/`focus_mode_cycle`），不配置 ratio 时生效。

**消息处理流程**
1. 入口 `on_message`：过滤私聊，抽取文本与图片，更新状态。
2. 触发周期：
   - AT：收到 @ 消息后触发。
   - FOCUS：处于专注模式时定期触发。
   - READ_AIR：心跳触发，根据消息量判断是否回复。
3. 任务调度：`Scheduler` 将任务入队并按优先级处理。
4. LLM 调用：`LLMClient` 组装提示词（含人物设定、相关记忆、历史对话等），调用 LLM 获取结构化响应。
5. 结果处理：解析 JSON，下载图片，发送消息链，清理临时文件。
