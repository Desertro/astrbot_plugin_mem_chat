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
        "当 control.decision 为 reply_only：必须满足其一：\n"
        "- response.text 为非空字符串\n"
        "- response.segments 为非空数组，且每个 segment.text 为非空字符串\n"
        "当需要对多人/多条上下文分别回应时，请优先使用 response.segments（每个 segment 表示一条要发送的消息）。\n"
        "【艾特约束】对于同一个用户，单次回复（包含所有 segments）中只能艾特一次。请在 response.mentions 或 segment.mentions 中填入 QQ 号，并在对应的 text 中使用 `<at:QQ号>` 占位符指定插入位置。\n"
        "如需引用，请在 response.quote_index 或 segment.quote_index 填写整数，对应【最新消息上下文】里的 [REF:n]。\n"
        '如需表情包：将表情标签写入 response.emotion（例如 "[happy]"），不要把 "[happy]" 写进 response.text。\n'
    )
    return (content or "").rstrip() + suffix


def build_decision_prompt(
    umo: str,
    cfg=None,
    last_msg: str | None = None,
    reply_line: str | None = None,
    history_lines: list[str] | None = None,
) -> str:
    tmpl = _get(
        cfg,
        "decision_prompt_template",
        "",
    )
    base = tmpl.format(umo=umo, last_msg=last_msg or "", reply_line=reply_line or "")
    parts = []
    if history_lines:
        parts.append("【最新消息上下文】\n" + "\n".join(history_lines))
    target = reply_line or (last_msg or "")
    if target:
        parts.append("【需要你优先处理的最新一条】\n" + target)
    parts.append(base)
    out = "\n".join([p for p in parts if p]).strip()
    return _append_structured_json_constraints(out)


def build_focus_prompt(
    umo: str,
    cfg=None,
    last_msg: str | None = None,
    reply_line: str | None = None,
    history_lines: list[str] | None = None,
) -> str:
    tmpl = _get(
        cfg,
        "focus_prompt_template",
        "",
    )
    base = tmpl.format(umo=umo, last_msg=last_msg or "", reply_line=reply_line or "")
    parts = []
    if history_lines:
        parts.append("【最新消息上下文】\n" + "\n".join(history_lines))
    target = reply_line or (last_msg or "")
    if target:
        parts.append("【需要你优先处理的最新一条】\n" + target)
    parts.append(base)
    out = "\n".join([p for p in parts if p]).strip()
    return _append_structured_json_constraints(out)
