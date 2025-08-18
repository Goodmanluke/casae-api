import os
import json
import openai
from typing import List, Dict, Any

# Configure the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def _safe_json(content: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Safely parse JSON from AI responses; return default on failure."""
    try:
        return json.loads(content)
    except Exception:
        return default

async def rank_comparables(subject: Dict[str, Any], comps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use GPT-4o-mini to select and rank the most relevant comparables.
    Returns a dict with keys 'comps' and 'reasoning'.
    Falls back to the first three comps if the AI fails.
    """
    prompt = f"""
    Subject Property: {subject}
    Raw Comparables: {comps}

    Task: Select the top 3–5 comparables and rank them by similarity (location, style, upgrades, condition).
    Respond in JSON: {{ "comps": <list>, "reasoning": <short explanation> }}
    """
    default_result = {"comps": comps[:3], "reasoning": "AI unavailable – fallback comps used."}
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a real estate CMA assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format="json",
            timeout=10,
        )
        return _safe_json(response.choices[0].message["content"], default_result)
    except Exception:
        return default_result

async def compute_adjusted_cma(subject: Dict[str, Any], comps: List[Dict[str, Any]], adjustments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask GPT-4o-mini to compute an adjusted property value given a baseline subject, its comps, and user adjustments.
    Returns a dict with keys 'value', 'reasoning', and optionally 'comps'.
    Falls back to the baseline estimate if the AI fails.
    """
    prompt = f"""
    Subject Property: {subject}
    Current Comparables: {comps}
    Adjustments: {adjustments}

    Task: Estimate the adjusted market value considering the adjustments. If adjustments change the property class (e.g., number of bedrooms), suggest new comps.
    Respond in JSON: {{ "value": <adjusted value>, "reasoning": <explanation>, "comps": <list or null> }}
    """
    default_result = {
        "value": subject.get("avm_value", 0),
        "reasoning": "AI unavailable – baseline estimate used.",
        "comps": comps,
    }
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a real estate CMA assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format="json",
            timeout=10,
        )
        return _safe_json(response.choices[0].message["content"], default_result)
    except Exception:
        return default_result

async def generate_cma_summary(subject: Dict[str, Any], comps: List[Dict[str, Any]], adjustments: Dict[str, Any], value: int) -> str:
    """
    Generate a human-readable summary for the CMA report using GPT-4o-mini.
    Returns a string summary. Falls back to a static disclaimer if the AI fails.
    """
    prompt = f"""
    Subject Property: {subject}
    Comparables: {comps}
    Adjustments: {adjustments}
    Final Value: {value}

    Write a professional CMA summary explaining why these comps were chosen, how adjustments impacted the value, and the current market context. Use 3–5 sentences.
    """
    fallback = "This CMA report was generated automatically. Please review the comps and adjustments."
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a real estate CMA writer."},
                {"role": "user", "content": prompt},
            ],
            timeout=10,
        )
        return response.choices[0].message["content"].strip()
    except Exception:
        return fallback
