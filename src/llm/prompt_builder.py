import re


def _get(cfg, key, default):
    try:
        if isinstance(cfg, dict):
            v = cfg.get(key, default)
        else:
            v = getattr(cfg, key, default)
    except Exception:
        v = default
    return v


def _append_structured_json_constraints(content: str) -> str:
    suffix = (
        "\n\n【输出格式强约束】\n"
        "你必须只输出一个 JSON 对象，禁止输出任何解释、Markdown 代码块或前后缀。\n"
        "必须包含字段：control.decision（no_reply/reply_only）与 response。\n"
        'JSON 字符串内部不要写未转义的英文双引号；需要强调词语时优先使用中文引号「」或单引号，必须使用英文双引号时写成 \\"。\n'
        "当 control.decision 为 reply_only：必须满足其一：\n"
        "- response.text 为非空字符串\n"
        "- response.segments 为非空数组，且每个 segment.text 为非空字符串\n"
        "多段回复能力：当回复天然包含多个发送节奏时，请优先使用 response.segments（每个 segment 表示一条实际发送的消息），不要把所有内容硬塞进 response.text。\n"
        "优先使用 response.segments 的场景：需要分别回应多人；需要分别回应多条消息/多个问题；情绪反应与实用信息应分开发送；内容超过约80个汉字且确有两个独立意图。\n"
        "单句能说清时使用 response.text，不要为了拆段而拆段；只有确有多个独立意图时才使用 segments。每段直接回答或作出反应，不要复述问题或只用语气词铺垫。使用 segments 时，response.text 建议留空，避免与 segments 重复。\n"
        "【艾特约束】对于同一个用户，单次回复（包含所有 segments）中只能艾特一次。请在 response.mentions 或 segment.mentions 中填入 QQ 号，并在对应的 text 中使用 `<at:QQ号>` 占位符指定插入位置。\n"
        "引用是可选能力：仅当你在直接回应某一条具体消息时，才填写 response.quote_index 或 segment.quote_index（对应【最新消息上下文】里的 [REF:n]）；否则留空。\n"
        "若使用 response.segments，仅允许正在直接回应某条消息的那个 segment 填写 segment.quote_index，其他 segment 必须留空。\n"
        '如需表情包：单条回复可写 response.emotion；多段回复请把表情标签写进对应 segment.text（例如 "好哦 [happy]"），系统会自动移除标签并只在该段发送表情。不要把同一个表情放在顶层 response.emotion 后再使用 segments。\n'
    )
    return (content or "").rstrip() + suffix


def _normalize_line_for_compare(line: str) -> str:
    s = str(line or "").strip()
    s = re.sub(r"^\[[0-9]{2}:[0-9]{2}\]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _dedupe_history_lines_against_target(
    history_lines: list[str] | None, target: str
) -> list[str]:
    if not history_lines:
        return []
    t = _normalize_line_for_compare(target)
    out: list[str] = []
    for ln in history_lines:
        n = _normalize_line_for_compare(ln)
        if not n:
            continue
        if t and n == t:
            continue
        out.append(ln)
    return out


def build_decision_prompt(
    umo: str,
    cfg=None,
    last_msg: str | None = None,
    reply_line: str | None = None,
    history_lines: list[str] | None = None,
    mood_text: str | None = None,
    target_block: str | None = None,
) -> str:
    tmpl = _get(
        cfg,
        "decision_prompt_template",
        "",
    )
    base = tmpl.format(umo=umo, last_msg=last_msg or "", reply_line=reply_line or "")
    target = reply_line or (last_msg or "")
    clean_history = _dedupe_history_lines_against_target(history_lines, target)
    parts = []
    if target_block:
        parts.append(str(target_block).strip())
    if clean_history:
        parts.append("【最新消息上下文】\n" + "\n".join(clean_history))
    if target:
        extra = ""
        if mood_text:
            extra = f"\n[上一轮mood.text] {str(mood_text).strip()}"
        parts.append("【需要你优先处理的最新一条】\n" + target + extra)
    parts.append(base)
    out = "\n".join([p for p in parts if p]).strip()
    return out


def build_focus_prompt(
    umo: str,
    cfg=None,
    last_msg: str | None = None,
    reply_line: str | None = None,
    history_lines: list[str] | None = None,
    mood_text: str | None = None,
    target_block: str | None = None,
) -> str:
    tmpl = _get(
        cfg,
        "focus_prompt_template",
        "",
    )
    base = tmpl.format(umo=umo, last_msg=last_msg or "", reply_line=reply_line or "")
    target = reply_line or (last_msg or "")
    clean_history = _dedupe_history_lines_against_target(history_lines, target)
    parts = []
    if target_block:
        parts.append(str(target_block).strip())
    if clean_history:
        parts.append("【最新消息上下文】\n" + "\n".join(clean_history))
    if target:
        extra = ""
        if mood_text:
            extra = f"\n[上一轮mood.text] {str(mood_text).strip()}"
        parts.append("【需要你优先处理的最新一条】\n" + target + extra)
    parts.append(base)
    out = "\n".join([p for p in parts if p]).strip()
    return out
