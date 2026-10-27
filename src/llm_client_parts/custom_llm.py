import asyncio
import copy
import json
import re
import time

import aiohttp

from .logger_proxy import logger


def _is_deepseek_openai_target(base_url: str, model: str) -> bool:
    haystack = f"{base_url or ''} {model or ''}".lower()
    return "deepseek" in haystack


def _get_int_conf(self, key: str, default: int) -> int:
    try:
        value = self._get_conf(key, default)
        if value in (None, ""):
            return int(default)
        parsed = int(value)
        return parsed if parsed > 0 else int(default)
    except Exception:
        return int(default)


def _openai_supports_image_url(base_url: str, model: str) -> bool:
    # The current DeepSeek Flash alias and the legacy vision model accept images.
    if _is_deepseek_openai_target(base_url, model):
        model_name = str(model or "").lower()
        return model_name == "deepseek-flash" or "vision" in model_name
    return True


def _json_mode_system_suffix() -> str:
    return (
        "\n\n【JSON Output 协议】\n"
        "你必须只输出一个合法 JSON object，不要 Markdown、代码块、解释文字或前后缀。"
        'JSON 字符串内部不要写未转义的英文双引号；需要强调词语时优先使用中文引号「」或单引号，必须使用英文双引号时写成 \\"。'
        "若回复适合拆成多条消息，请优先使用 response.segments；"
        "例如多人/多问题/先接话再补充/情绪与信息分离时，segments 建议 2-4 段，每段一个意图，response.text 可留空。"
        "多段回复如需表情包，请把表情标签写进对应 segment.text，不要放在顶层 response.emotion。"
        "JSON 示例："
        '{"control":{"decision":"reply_only"},"response":{"text":"收到","tts":{"enabled":false}},'
        '"mood":{"text":"认真处理当前对话","state":"中性","ttl_sec":60,"intensity":0.2},'
        '"affinity":{"delta":0,"reason":"正常互动"}}'
    )


def _strip_unsupported_strict_keywords(schema):
    if isinstance(schema, dict):
        out = copy.deepcopy(schema)
        for key in ("minLength", "maxLength", "minItems", "maxItems"):
            out.pop(key, None)
        if out.get("type") == "object":
            props = out.get("properties")
            if isinstance(props, dict):
                out["properties"] = {
                    k: _strip_unsupported_strict_keywords(v) for k, v in props.items()
                }
                out["required"] = list(props.keys())
            out["additionalProperties"] = False
        if out.get("type") == "array" and "items" in out:
            out["items"] = _strip_unsupported_strict_keywords(out.get("items"))
        if isinstance(out.get("anyOf"), list):
            out["anyOf"] = [
                _strip_unsupported_strict_keywords(item) for item in out["anyOf"]
            ]
        for defs_key in ("$def", "$defs"):
            defs = out.get(defs_key)
            if isinstance(defs, dict):
                out[defs_key] = {
                    k: _strip_unsupported_strict_keywords(v) for k, v in defs.items()
                }
        return out
    if isinstance(schema, list):
        return [_strip_unsupported_strict_keywords(item) for item in schema]
    return schema


async def call_custom_llm(
    self,
    base_url,
    api_key,
    model,
    system_prompt,
    prompt,
    contexts,
    tools=None,
    image_urls=None,
    temperature=None,
    fmt=None,
    tag: str = "",
):
    fmt = str(fmt or self._get_conf("custom_llm_format", "openai")).lower()
    if fmt == "google":
        return await call_custom_llm_google(
            self,
            base_url,
            api_key,
            model,
            system_prompt,
            prompt,
            contexts,
            tools,
            image_urls,
            temperature,
            tag=tag,
        )
    deepseek_strict_tools = bool(
        self._get_bool_conf("deepseek_tool_strict_mode", False)
        and _is_deepseek_openai_target(base_url, model)
    )
    if deepseek_strict_tools:
        base = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
        if not base.endswith("/beta"):
            base = base.rstrip("/") + "/beta"
        base_url = base
    elif not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    json_mode = self._get_bool_conf("force_llm_json_output", False)
    if json_mode:
        system_prompt = (system_prompt or "") + _json_mode_system_suffix()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for ctx in contexts:
        msg = copy.deepcopy(ctx)
        messages.append(msg)

    user_content = []
    if prompt:
        user_content.append({"type": "text", "text": prompt})

    if image_urls:
        if _openai_supports_image_url(base_url, model):
            try:
                llm_imgs = await self._prepare_llm_image_contents(image_urls)
                if _is_deepseek_openai_target(base_url, model):
                    for item in llm_imgs:
                        image_url = (
                            item.get("image_url") if isinstance(item, dict) else None
                        )
                        if isinstance(image_url, dict):
                            image_url.setdefault("detail", "original")
                user_content.extend(llm_imgs)
            except Exception:
                for img_url in image_urls:
                    image_url = {"url": img_url}
                    if _is_deepseek_openai_target(base_url, model):
                        image_url["detail"] = "original"
                    user_content.append({"type": "image_url", "image_url": image_url})
        else:
            logger.info(
                f"[mem_chat] Skipping image_url payload for text-only OpenAI target model={model}; relying on prompt/history image descriptions."
            )
            user_content.append(
                {
                    "type": "text",
                    "text": f"【系统提示】本轮包含 {len(image_urls)} 张图片，但当前 LLM 接口不支持 image_url 输入；请仅依据上文已有图片描述、引用与聊天文本判断。",
                }
            )
    if user_content:
        messages.append({"role": "user", "content": user_content})
    payload = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        try:
            payload["temperature"] = float(temperature)
        except Exception:
            pass
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
        payload["max_tokens"] = _get_int_conf(self, "llm_max_tokens", 3072)
    if "grok" in model.lower():
        logger.info(
            f"[mem_chat] Detected Grok model '{model}', disabling native search via search_parameters."
        )
        payload["search_parameters"] = {"mode": "off"}
    if tools and hasattr(tools, "tools") and tools.tools:
        openai_tools = []
        for t in tools.tools:
            params = t.parameters or {"type": "object", "properties": {}}
            if deepseek_strict_tools:
                params = _strip_unsupported_strict_keywords(params)
            tool_schema = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": params,
                },
            }
            if deepseek_strict_tools:
                tool_schema["function"]["strict"] = True
            openai_tools.append(tool_schema)
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"
            try:
                tool_names = [tool["function"]["name"] for tool in openai_tools]
                logger.info(
                    f"[mem_chat] Custom LLM payload using OpenAI tools protocol; tools: {tool_names}"
                )
            except Exception:
                logger.info(
                    "[mem_chat] Custom LLM payload attached OpenAI tools schema."
                )
    try:
        total_chars = 0
        inline_image_chars = 0
        inline_image_count = 0
        image_url_count = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                total_chars += len(c)
                continue
            if isinstance(c, list):
                for seg in c:
                    if not isinstance(seg, dict):
                        total_chars += len(str(seg))
                        continue
                    if seg.get("type") == "text":
                        total_chars += len(str(seg.get("text", "")))
                    elif seg.get("type") == "image_url":
                        u = (seg.get("image_url") or {}).get("url") or ""
                        total_chars += len(u)
                        image_url_count += 1
                        if isinstance(u, str) and u.startswith("data:image/"):
                            inline_image_count += 1
                            inline_image_chars += len(u)
                    else:
                        total_chars += len(json.dumps(seg, ensure_ascii=False))
        tag_part = f" tag={tag}" if tag else ""
        logger.info(
            f"[mem_chat] Custom LLM request stats:{tag_part} model={model} msgs={len(messages)} total_chars={total_chars} image_urls={image_url_count} inline_images={inline_image_count} inline_image_chars={inline_image_chars}"
        )
    except Exception:
        pass
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            max_attempts = 2 if json_mode else 1
            for attempt in range(max_attempts):
                if attempt > 0:
                    retry_messages = copy.deepcopy(messages)
                    retry_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "上一轮 JSON Output 返回为空。请立即重试，只输出一个合法 JSON object，"
                                        "至少包含 control.decision 与 response。"
                                    ),
                                }
                            ],
                        }
                    )
                    payload["messages"] = retry_messages
                async with session.post(
                    url, headers=headers, json=payload, timeout=120
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choices = data.get("choices", [])
                        if not choices:
                            return None
                        choice = choices[0]
                        message = choice.get("message", {})
                        content = message.get("content", "")
                        tool_calls = message.get("tool_calls", [])
                        if tool_calls and isinstance(tool_calls, list):
                            tool_calls = [
                                tc
                                for tc in tool_calls
                                if tc.get("function", {}).get("name")
                            ]

                        if (
                            json_mode
                            and not (content or "").strip()
                            and not tool_calls
                            and attempt + 1 < max_attempts
                        ):
                            logger.warning(
                                "[mem_chat] Custom LLM returned empty JSON content; retrying once with stricter JSON prompt."
                            )
                            continue

                        if tool_calls:
                            try:
                                names_ids = [
                                    (tc.get("function", {}).get("name"), tc.get("id"))
                                    for tc in tool_calls
                                ]
                                logger.info(
                                    f"[mem_chat] Custom LLM returned tool_calls: {names_ids}"
                                )
                            except Exception:
                                logger.info(
                                    "[mem_chat] Custom LLM returned non-empty tool_calls."
                                )

                        class CustomLLMResponse:
                            def __init__(
                                self,
                                role,
                                text,
                                tool_names=None,
                                tool_args=None,
                                tool_ids=None,
                            ):
                                self.role = role
                                self.completion_text = text
                                self.tools_call_name = tool_names or []
                                self.tools_call_args = tool_args or []
                                self.tools_call_ids = tool_ids or []
                                self.result_chain = None
                                self.reasoning_content = (
                                    message.get("reasoning_content") or ""
                                )
                                self.raw_completion = tool_calls

                        role = message.get("role", "assistant")
                        t_names = []
                        t_args = []
                        t_ids = []
                        if tool_calls:
                            role = "assistant"
                            for tc in tool_calls:
                                func = tc.get("function", {})
                                t_names.append(func.get("name"))
                                try:
                                    t_args.append(
                                        json.loads(func.get("arguments", "{}"))
                                    )
                                except Exception:
                                    t_args.append(func.get("arguments", ""))
                                t_ids.append(tc.get("id"))
                        return CustomLLMResponse(role, content, t_names, t_args, t_ids)
                    text = await resp.text()
                    logger.error(
                        f"[mem_chat] Custom LLM call failed: {resp.status} - {text}"
                    )
                    return None
            return None
    except Exception as e:
        logger.error(f"[mem_chat] Custom LLM error: {e}")
        return None


async def call_custom_llm_google(
    self,
    base_url,
    api_key,
    model,
    system_prompt,
    prompt,
    contexts,
    tools=None,
    image_urls=None,
    temperature=None,
    tag: str = "",
):
    base = base_url.rstrip("/")
    url = (
        f"{base}/v1beta/models/{model}:generateContent"
        if not model.startswith("models/")
        else f"{base}/v1beta/{model}:generateContent"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _convert_to_gemini_tools(openai_tools):
        if not openai_tools:
            return None

        declarations = []
        for t in openai_tools:
            if hasattr(t, "model_dump"):
                t_dict = t.model_dump()
            elif isinstance(t, dict):
                t_dict = t
            else:
                continue

            func = t_dict.get("function") or t_dict
            name = func.get("name")
            desc = func.get("description")
            params = func.get("parameters")

            if name:
                decl = {"name": name, "description": desc or ""}
                if params:
                    decl["parameters"] = params
                declarations.append(decl)

        if not declarations:
            return None
        return [{"function_declarations": declarations}]

    def _to_parts_from_content(c):
        if isinstance(c, str):
            return [{"text": c}]
        if isinstance(c, list):
            parts = []
            for seg in c:
                if isinstance(seg, dict):
                    if seg.get("type") == "text":
                        parts.append({"text": seg.get("text", "")})
                    elif seg.get("type") == "image_url":
                        u = (seg.get("image_url") or {}).get("url") or ""
                        u = str(u or "").strip().strip("`").strip()
                        if u.startswith("data:image/"):
                            try:
                                head, b64 = u.split(",", 1)
                                mime = head.split(";")[0].split(":")[1]
                                parts.append(
                                    {"inline_data": {"mime_type": mime, "data": b64}}
                                )
                            except Exception:
                                parts.append({"text": "[图片]"})
                        else:
                            parts.append({"text": "[图片]"})
            return parts
        return [{"text": str(c)}]

    system_instruction = None
    if system_prompt:
        system_instruction = {"parts": [{"text": system_prompt}]}

    raw_items = []
    for ctx in contexts:
        r = ctx.get("role")
        c = ctx.get("content")
        if r == "assistant":
            raw_items.append({"role": "model", "parts": _to_parts_from_content(c)})
        elif r == "user":
            raw_items.append({"role": "user", "parts": _to_parts_from_content(c)})
        elif r == "system":
            if isinstance(c, str) and c.strip():
                raw_items.append({"role": "user", "parts": [{"text": c.strip()}]})
        elif r == "tool":
            tool_name = ctx.get("name") or ctx.get("tool_call_id") or "tool"
            tool_text = f"【工具结果】{tool_name}\n{c}"
            raw_items.append({"role": "user", "parts": [{"text": str(tool_text)}]})

    user_parts = []
    if prompt:
        user_parts.append({"text": prompt})
    if image_urls:
        try:
            llm_imgs = await self._prepare_llm_image_contents(image_urls)
            user_parts.extend(_to_parts_from_content(llm_imgs))
        except Exception:
            for img_url in image_urls:
                user_parts.append({"text": img_url})
    if user_parts:
        raw_items.append({"role": "user", "parts": user_parts})

    contents = []
    if raw_items:
        merged = []
        for item in raw_items:
            if not merged:
                merged.append(item)
            else:
                if merged[-1]["role"] == item["role"]:
                    merged[-1]["parts"].extend(item["parts"])
                else:
                    merged.append(item)

        if merged and merged[0]["role"] == "model":
            merged.insert(0, {"role": "user", "parts": [{"text": "Continue"}]})

        contents = merged

    payload = {"contents": contents}
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    gemini_tools = _convert_to_gemini_tools(tools)
    if gemini_tools:
        payload["tools"] = gemini_tools

    gen = {}
    if temperature is not None:
        try:
            gen["temperature"] = float(temperature)
        except Exception:
            pass
    if self._get_bool_conf("force_llm_json_output", False):
        gen["responseMimeType"] = "application/json"
        gen["maxOutputTokens"] = _get_int_conf(self, "llm_max_tokens", 3072)
    if gen:
        payload["generationConfig"] = gen

    try:
        total_chars = 0
        inline_image_chars = 0
        inline_image_count = 0
        for c in contents:
            parts = c.get("parts") or []
            for p in parts:
                if not isinstance(p, dict):
                    total_chars += len(str(p))
                    continue
                if "text" in p:
                    total_chars += len(str(p.get("text") or ""))
                elif "inline_data" in p:
                    d = (p.get("inline_data") or {}).get("data") or ""
                    total_chars += len(str(d))
                    inline_image_chars += len(str(d))
                    inline_image_count += 1
                else:
                    total_chars += len(json.dumps(p, ensure_ascii=False))
        tag_part = f" tag={tag}" if tag else ""
        tool_count = (
            len(gemini_tools[0]["function_declarations"]) if gemini_tools else 0
        )
        logger.info(
            f"[mem_chat] Custom Gemini request stats:{tag_part} model={model} contents={len(contents)} tools={tool_count} total_chars={total_chars} inline_images={inline_image_count} inline_image_chars={inline_image_chars}"
        )
    except Exception:
        pass

    retryable_statuses = {429, 500, 502, 503, 504}
    max_attempts = 4
    attempt = 0
    last_error_text = ""
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            while attempt < max_attempts:
                try:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            cands = data.get("candidates", [])
                            if not cands:
                                logger.warning(
                                    f"[mem_chat] Gemini returned no candidates: {data}"
                                )
                                return None
                            cand = cands[0]
                            parts = (cand.get("content") or {}).get("parts") or []

                            texts = []
                            tool_calls_name = []
                            tool_calls_args = []
                            tool_calls_ids = []

                            for p in parts:
                                if "text" in p:
                                    texts.append(p["text"])
                                elif "functionCall" in p:
                                    fc = p["functionCall"]
                                    tool_calls_name.append(fc.get("name"))
                                    tool_calls_args.append(fc.get("args") or {})
                                    tool_calls_ids.append(
                                        f"call_{int(time.time())}_{len(tool_calls_ids)}"
                                    )

                            text_out = "\n".join(texts).strip()

                            class CustomLLMResponse:
                                def __init__(
                                    self, role, text, tc_names, tc_args, tc_ids
                                ):
                                    self.role = role
                                    self.completion_text = text
                                    self.tools_call_name = tc_names
                                    self.tools_call_args = tc_args
                                    self.tools_call_ids = tc_ids
                                    self.result_chain = None
                                    self.reasoning_content = ""
                                    self.raw_completion = None

                            return CustomLLMResponse(
                                "assistant",
                                text_out,
                                tool_calls_name,
                                tool_calls_args,
                                tool_calls_ids,
                            )

                        last_error_text = await resp.text()
                        tag_part = f" ({tag})" if tag else ""
                        if (
                            resp.status in retryable_statuses
                            and attempt < max_attempts - 1
                        ):
                            retry_after = resp.headers.get("Retry-After")
                            delay = None
                            if isinstance(retry_after, str):
                                try:
                                    delay_val = float(retry_after.strip())
                                    if delay_val > 0:
                                        delay = delay_val
                                except Exception:
                                    delay = None
                            if delay is None:
                                delay = min(20.0, 4**attempt)
                            logger.warning(
                                f"[mem_chat] Custom Gemini call retrying{tag_part}: {resp.status} (attempt {attempt + 1}/{max_attempts}) sleep={delay:.2f}s | error={last_error_text[:200]}"
                            )
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue

                        logger.error(
                            f"[mem_chat] Custom Gemini call failed{tag_part}: {resp.status} - {last_error_text}"
                        )
                        return None
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    tag_part = f" ({tag})" if tag else ""
                    if attempt < max_attempts - 1:
                        delay = min(12.0, 2**attempt)
                        logger.warning(
                            f"[mem_chat] Custom Gemini call network retrying{tag_part}: {type(e).__name__} (attempt {attempt + 1}/{max_attempts}) sleep={delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    logger.error(f"[mem_chat] Custom Gemini error{tag_part}: {e}")
                    return None
    except Exception as e:
        tag_part = f" ({tag})" if tag else ""
        if last_error_text:
            logger.error(
                f"[mem_chat] Custom Gemini error{tag_part}: {e} | last_text={last_error_text}"
            )
        else:
            logger.error(f"[mem_chat] Custom Gemini error{tag_part}: {e}")
        return None
