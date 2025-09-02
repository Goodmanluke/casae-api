from openai import AsyncOpenAI
import os
import json
from typing import List, Dict, Any

# Create a single async client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _safe_json(content: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        return default

async def rank_comparables(subject: Dict[str, Any], comps: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = f"""
    Subject Property: {subject}
    Raw Comparables: {comps}

    Task: Select the top 3–5 comparables and rank them by similarity (location, style, upgrades, condition).
    Respond in JSON: {{ "comps": <list>, "reasoning": <short explanation> }}
    """
    default_result = {"comps": comps[:3], "reasoning": "AI unavailable – fallback comps used."}
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a real estate CMA assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=10,
        )
        return _safe_json(response.choices[0].message.content, default_result)
    except Exception:
        return default_result

async def compute_adjusted_cma(subject: Dict[str, Any], comps: List[Dict[str, Any]], adjustments: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
    Subject Property: {subject}
    Current Comparables: {comps}
    Adjustments: {adjustments}

    Task: Estimate the adjusted market value considering the adjustments.
    Respond in JSON: {{ "value": <adjusted value>, "reasoning": <explanation>, "comps": <list or null> }}
    """
    default_result = {"value": 0, "reasoning": "AI unavailable – baseline estimate used.", "comps": comps}
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a real estate CMA assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=10,
        )
        return _safe_json(response.choices[0].message.content, default_result)
    except Exception as e:
        print(f"AI function error: {e}")
        return default_result

async def generate_cma_summary(subject: Dict[str, Any], comps: List[Dict[str, Any]], adjustments: Dict[str, Any], value: int) -> str:
    prompt = f"""
    Subject Property: {subject}
    Comparables: {comps}
    Adjustments: {adjustments}
    Final Value: {value}

    Write a professional CMA summary explaining why these comps were chosen, how adjustments impacted the value, and the current market context. Use 3–5 sentences.
    """
    fallback = "This CMA report was generated automatically. Please review the comps and adjustments."
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a real estate CMA writer."},
                {"role": "user", "content": prompt},
            ],
            timeout=10,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return fallback
