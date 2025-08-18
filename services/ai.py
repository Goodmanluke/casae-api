import os
import json
from typing import Any, Dict, List, Optional

import asyncio
import httpx

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ---- Internals --------------------------------------------------------------

async def _call_openai(messages: List[Dict[str, str]], want_json: bool = False, timeout_s: int = 10) -> Optional[str]:
    """
    Returns the assistant message content or None on failure.
    Uses JSON mode when want_json=True (the model should produce a JSON string).
    """
    if not OPENAI_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if want_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(OPENAI_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None

def _safe_json(s: Optional[str], default: Dict[str, Any]) -> Dict[str, Any]:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

# ---- Public API -------------------------------------------------------------

async def rank_comparables(subject: Dict[str, Any], comps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns: {"comps": [<subset/reordered comps>], "reasoning": "<short text>"}
    Fallback: top 3 original comps with generic reasoning.
    """
    default = {"comps": comps[:3], "reasoning": "AI unavailable – using top 3 raw comps."}
    prompt = f"""
Subject: {json.dumps(subject)}
Raw comps: {json.dumps(comps)}

Task: pick the best 3–5 comps for a CMA. Consider proximity, size, beds/baths, year, style, and recency.
Return strict JSON: {{"comps": <list of comps from input>, "reasoning": <short string>}}.
Do not invent fields; reuse the input comp structures.
"""
    content = await _call_openai(
        [
            {"role": "system", "content": "You are a real estate CMA assistant."},
            {"role": "user", "content": prompt},
        ],
        want_json=True,
    )
    return _safe_json(content, default)

async def compute_adjusted_cma(subject: Dict[str, Any], comps: List[Dict[str, Any]], adjustments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns: {"value": <float>, "reasoning": "<text>", "comps": <optional new list>}
    Fallback: baseline avm_value or median comp price.
    """
    baseline = subject.get("avm_value")
    if baseline is None and comps:
        try:
            prices = [c.get("raw_price") for c in comps if c.get("raw_price")]
            prices = [float(p) for p in prices]
            if prices:
                baseline = sorted(prices)[len(prices)//2]
        except Exception:
            pass
    if baseline is None:
        baseline = 0.0

    default = {"value": baseline, "reasoning": "AI unavailable – baseline used.", "comps": comps}
    prompt = f"""
Subject (pre-adjustments): {json.dumps(subject)}
Current comps: {json.dumps(comps)}
User adjustments: {json.dumps(adjustments)}

Task: estimate the adjusted market value. If adjustments reclassify the subject (e.g., +bedrooms/+sqft),
you MAY suggest a replacement comp list (otherwise leave comps as is).
Return JSON: {{"value": <number>, "reasoning": <string>, "comps": <list or null>}}.
"""
    content = await _call_openai(
        [
            {"role": "system", "content": "You are a real estate CMA assistant."},
            {"role": "user", "content": prompt},
        ],
        want_json=True,
    )
    return _safe_json(content, default)

async def generate_cma_summary(subject: Dict[str, Any], comps: List[Dict[str, Any]], adjustments: Dict[str, Any], value: float) -> str:
    """
    Returns a short, professional CMA summary paragraph.
    Fallback: static disclaimer.
    """
    prompt = f"""
Write a 3–5 sentence CMA summary for a report.
Subject: {json.dumps(subject)}
Comps (top 3–5): {json.dumps(comps[:5])}
Adjustments: {json.dumps(adjustments)}
Final value: {value}
Explain why comps were chosen and how adjustments affected the value.
"""
    content = await _call_openai(
        [
            {"role": "system", "content": "You write concise, professional CMA summaries."},
            {"role": "user", "content": prompt},
        ],
        want_json=False,
    )
    return content or "This CMA was generated automatically. Please review comps and adjustments."
