import argparse
import json
import os
import urllib.error
import urllib.request


def estimate_tokens(text: str) -> int:
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_count = len(text) - ascii_count
    return int(ascii_count / 4 + non_ascii_count)

def load_input(json_path: str | None) -> dict:
    if json_path:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "results": [
            {
                "title": "花帆是谁？角色与作品背景速览",
                "url": "https://example.com/wiki/%E8%8A%B1%E5%B8%86",
                "content": """花帆为作品中的主要角色，常见别名与原名检索需要结合日文名与相关系列上下文进行。
涉及多个子系列与角色关联，条目较长，包含相当多的文本说明，用于测试压缩效果。"""
            },
            {
                "title": "氷上メルル与花帆的关系梳理",
                "url": "https://example.com/blog/%E6%B0%B7%E4%B8%8A%E3%83%A1%E3%83%AB%E3%83%AB-%E8%8A%B1%E5%B8%86",
                "content": """氷上メルル（冰上梅露露）相关作品在 Pixiv 上标签通常同时出现。若仅使用中文昵称检索，可能无法命中高质量作品，需要通过网页搜索解析原名。"""
            },
            {
                "title": "日文原名与别名的检索策略",
                "url": "https://example.com/guide/nickname-search",
                "content": """当用户仅给出昵称时，建议先使用网页搜索解析本名/日文原名/全称，再在 Pixiv 或 Twitter/X 进行图片检索，提升命中率。"""
            },
            {
                "title": "Pixiv 热门排序与作品检索",
                "url": "https://example.com/pixiv/search",
                "content": """Pixiv 搜索支持 popular_desc，需高级会员。无高级会员可用 date_desc 或排行榜接口近似。检索时组合中日文关键词更稳定。"""
            },
            {
                "title": "Tavily 搜索源的稳定性",
                "url": "https://example.com/search/tavily",
                "content": """Tavily 在国内网络下表现稳定，返回标题、链接与简要摘要。压缩这些结果可显著减少第二轮 LLM 输入体量。"""
            },
            {
                "title": "工具调用两段式的性能影响",
                "url": "https://example.com/tools/latency",
                "content": """在函数调用场景下，第一段用于检索，第二段用于整合与生成文本。减少第二段输入体量可降低超时概率。"""
            }
        ]
    }

def load_tavily_key_from_cmd_config() -> str | None:
    # Try import AstrBotConfig
    try:
        from astrbot.core.config.astrbot_config import AstrBotConfig
        cfg = AstrBotConfig()
        keys = cfg.get("provider_settings", {}).get("websearch_tavily_key", [])
        if isinstance(keys, list) and keys:
            return keys[0]
    except Exception:
        pass
    # Fallback to direct file
    for candidate in [
        os.path.join(os.getcwd(), "data", "cmd_config.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cmd_config.json"),
    ]:
        try:
            with open(candidate, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            keys = cfg.get("provider_settings", {}).get("websearch_tavily_key", [])
            if isinstance(keys, list) and keys:
                return keys[0]
        except Exception:
            continue
    return None

def tavily_search(query: str, max_results: int = 10, depth: str = "basic") -> dict:
    api_key = os.environ.get("TAVILY_API_KEY") or load_tavily_key_from_cmd_config()
    if not api_key:
        raise RuntimeError("Tavily API key not found. Set TAVILY_API_KEY or configure provider_settings.websearch_tavily_key.")
    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": depth,
        "topic": "general",
        "include_images": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            txt = resp.read().decode("utf-8")
            return json.loads(txt)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"HTTP {e.code}: {body}",
        ) from e
    except Exception as e:
        raise RuntimeError(f"Request error: {e}") from e

def build_variants(data: dict) -> dict:
    results = data.get("results") or []
    full = []
    title_url = []
    urls = []
    title_only = []
    title_summary = []
    for r in results:
        t = str(r.get("title", ""))
        u = str(r.get("url", ""))
        c = str(r.get("content", ""))
        full.append("\n".join([t, u, c]).strip())
        title_url.append((t + " " + u).strip())
        urls.append(u)
        title_only.append(t)
        summary = c.strip()
        if len(summary) > 200:
            summary = summary[:200]
        title_summary.append((t + "\n" + summary).strip())
    full_text = "\n\n".join(full)
    title_url_text = "\n".join(title_url[:5])
    urls_text = "\n".join(urls[:5])
    title_only_text = "\n".join(title_only[:5])
    title_summary_text = "\n\n".join(title_summary[:5])
    compressed_current = title_url_text
    compressed_no_urls = title_only_text
    return {
        "raw_json": json.dumps(data, ensure_ascii=False),
        "full": full_text,
        "title_url": title_url_text,
        "urls_only": urls_text,
        "title_only": title_only_text,
        "title_summary": title_summary_text,
        "compressed_current": compressed_current,
        "compressed_no_urls": compressed_no_urls,
    }

def summarize(name: str, text: str) -> dict:
    return {
        "name": name,
        "chars": len(text),
        "tokens": estimate_tokens(text),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--query", default="花帆是谁")
    args = parser.parse_args()
    if args.live:
        data = tavily_search(args.query, max_results=10, depth="basic")
    else:
        data = load_input(args.json)
    variants = build_variants(data)
    rows = [summarize(k, v) for k, v in variants.items()]
    base = next((r for r in rows if r["name"] == "full"), None)
    for r in rows:
        rel = ""
        if base:
            dc = r["chars"] - base["chars"]
            dt = r["tokens"] - base["tokens"]
            pc = (dc / base["chars"] * 100) if base["chars"] else 0
            pt = (dt / base["tokens"] * 100) if base["tokens"] else 0
            rel = f" | Δchars={dc} ({pc:.1f}%), Δtokens={dt} ({pt:.1f}%)"
        name = r["name"]
        chars = r["chars"]
        tokens = r["tokens"]
        print(f"{name}: chars={chars}, tokens={tokens}{rel}")

if __name__ == "__main__":
    main()
