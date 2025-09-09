import os
import asyncio
from typing import List, Dict, Any
import httpx

S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"

USER_AGENT = "CleantrustDemo/1.0 (research app; contact@example.org)"  # 建议改成带你邮箱

def _headers(extra: Dict[str, str] | None = None) -> Dict[str, str]:
    h = {"User-Agent": USER_AGENT}
    if extra:
        h.update(extra)
    return h

async def _get_with_retries(url: str, *, params: dict, headers: dict, timeout: float = 30.0, retries: int = 3):
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True, limits=limits) as client:
        for i in range(retries):
            try:
                r = await client.get(url, params=params, headers=headers)
                if r.status_code in (429, 502, 503, 504):
                    # 限流/网关问题：退避重试
                    await asyncio.sleep(1.5 * (2 ** i))
                    continue
                return r
            except httpx.ReadTimeout:
                await asyncio.sleep(1.5 * (2 ** i))
            except Exception as e:
                print(f"[httpx] soft-exception: {e!r}")
                await asyncio.sleep(1.0 * (2 ** i))
        return None

async def search_semantic_scholar(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Graph API 搜索。失败/限流返回 []（不抛异常）。"""
    q = (query or "").strip()
    if not q:
        return []
    # ❗用 externalIds（不要用 `doi` 字段）
    fields = "title,abstract,year,venue,url,citationCount,authors.name,externalIds"
    headers = _headers()
    if os.getenv("S2_API_KEY"):
        headers["x-api-key"] = os.getenv("S2_API_KEY")

    params = {"query": q, "limit": limit, "fields": fields}
    r = await _get_with_retries(S2_ENDPOINT, params=params, headers=headers)
    if r is None:
        print("[S2] soft-fail: no response after retries")
        return []
    if r.status_code >= 400:
        print(f"[S2] soft-fail {r.status_code}: {r.text[:200]}")
        return []

    items = []
    try:
        data = r.json()
        for p in data.get("data", []):
            ext = p.get("externalIds") or {}
            items.append({
                "title": p.get("title"),
                "abstract": p.get("abstract") or "",
                "year": p.get("year"),
                "doi": ext.get("DOI"),
                "url": p.get("url"),
                "venue": p.get("venue"),
                "citations": p.get("citationCount", 0),
                "source": "Semantic Scholar",
            })
    except Exception as e:
        print(f"[S2] parse soft-fail: {e!r}")
    return items

async def search_crossref(query: str, rows: int = 8) -> List[Dict[str, Any]]:
    """Crossref 搜索。失败/超时返回 []。"""
    q = (query or "").strip()
    if not q:
        return []
    params = {"query": q, "rows": rows}
    r = await _get_with_retries(CROSSREF_ENDPOINT, params=params, headers=_headers(), timeout=35.0)
    if r is None:
        print("[Crossref] soft-fail: no response after retries")
        return []
    if r.status_code >= 400:
        print(f"[Crossref] soft-fail {r.status_code}: {r.text[:200]}")
        return []

    items: List[Dict[str, Any]] = []
    try:
        data = r.json()
        for it in data.get("message", {}).get("items", []):
            title = (it.get("title") or [""])[0]
            abstract = it.get("abstract") or ""
            doi = it.get("DOI")
            url = it.get("URL")
            year = (it.get("published-print") or it.get("published-online") or {}).get("date-parts", [[None]])[0][0]
            items.append({
                "title": title,
                "abstract": abstract,
                "year": year,
                "doi": doi,
                "url": url,
                "venue": (it.get("container-title") or [""])[0],
                "citations": it.get("is-referenced-by-count", 0),
                "source": "Crossref",
            })
    except Exception as e:
        print(f"[Crossref] parse soft-fail: {e!r}")
    return items

async def search_papers_combined(query: str, limit_total: int = 10) -> List[Dict[str, Any]]:
    """并行检索 + 软失败；任何一侧失败都不会抛异常。"""
    s2_task = asyncio.create_task(search_semantic_scholar(query, limit=limit_total))
    cr_task = asyncio.create_task(search_crossref(query, rows=limit_total))
    s2, cr = await asyncio.gather(s2_task, cr_task)

    # 合并去重（优先 DOI，其次标题）
    seen = set()
    merged: List[Dict[str, Any]] = []
    for lst in (s2, cr):
        for p in lst:
            key = (p.get("doi") or "").lower() or (p.get("title") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(p)
    return merged[:limit_total]