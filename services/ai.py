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

async def adjust_comp_prices_for_condition_and_renovations(
    subject_condition: str,
    subject_renovations: List[str],
    comps: List[Dict[str, Any]],
    default_condition: str = "Good",
    default_renovations: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Use OpenAI to adjust comparable property prices based on condition and renovation differences.
    
    Args:
        subject_condition: The condition of the subject property (Poor, Fair, Good, Excellent)
        subject_renovations: List of renovations for subject (kitchen, bath, flooring, roof, windows)
        comps: List of comparable properties with their raw_price values
        default_condition: Default condition to compare against ("Good")
        default_renovations: Default renovations to compare against (all selected if None)
    
    Returns:
        List of comparables with adjusted raw_price values
    """
    if default_renovations is None:
        default_renovations = ["kitchen", "bath", "flooring", "roof", "windows"]
    
    # Check if adjustments are needed
    condition_changed = subject_condition != default_condition
    renovations_changed = set(subject_renovations) != set(default_renovations)
    
    if not condition_changed and not renovations_changed:
        # No adjustments needed, return original comps
        return comps
    
    missing_renovations = set(default_renovations) - set(subject_renovations)
    extra_renovations = set(subject_renovations) - set(default_renovations)
    
    prompt = f"""
    REAL ESTATE APPRAISAL TASK: Adjust comparable sale prices for a subject property.
    
    BASELINE: Comparable sales assume {default_condition} condition with renovations: {', '.join(default_renovations)}
    SUBJECT: Actually has {subject_condition} condition with renovations: {', '.join(subject_renovations) if subject_renovations else 'none'}
    
    COMPARABLES:
    {json.dumps(comps, indent=2)}
    
    ADJUSTMENT RULES:
    - Subject BETTER than baseline = INCREASE comparable prices
    - Subject WORSE than baseline = DECREASE comparable prices
    
    SPECIFIC ADJUSTMENTS:
    1. Condition: {subject_condition} vs {default_condition}
       - Excellent > Good: ADD 5% 
       - Good = Good: No change
       - Fair < Good: SUBTRACT 5%
       - Poor < Good: SUBTRACT 10%
    
    2. Missing renovations: {list(missing_renovations) if missing_renovations else 'none'}
       - Each missing renovation: SUBTRACT $15,000-30,000 depending on home value
    
    JSON Response:
    {{
        "adjusted_comps": [
            {{
                "id": "comp_id",
                "original_price": original_price,
                "adjusted_price": adjusted_price,
                "adjustment_amount": difference,
                "adjustment_reasoning": "Brief reason"
            }}
        ]
    }}
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional real estate appraiser. CRITICAL: When subject property has BETTER condition (Excellent > Good), the subject should be worth MORE than baseline comps, so INCREASE the prices. When subject has WORSE condition (Poor < Good), DECREASE the prices. When subject has FEWER renovations than baseline, DECREASE the prices."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=100,
        )
        
        ai_result = _safe_json(response.choices[0].message.content, {"adjusted_comps": [], "overall_reasoning": "AI adjustment failed"})
        
        if not ai_result.get('adjusted_comps') or len(ai_result['adjusted_comps']) == 0:
            return comps  # Return original if no valid adjustments
        
        adjusted_comps = []
        adjustment_applied = False
        
        for i, comp in enumerate(comps):
            original_comp = comp.copy()
            
            ai_adjustment = None
            if i < len(ai_result.get("adjusted_comps", [])):
                ai_adjustment = ai_result["adjusted_comps"][i]
            
            if ai_adjustment and "adjusted_price" in ai_adjustment:
                new_price = ai_adjustment["adjusted_price"]
                if new_price != original_comp.get("raw_price"):
                    original_comp["raw_price"] = new_price
                    original_comp["ai_adjustment"] = {
                        "original_price": ai_adjustment.get("original_price", comp.get("raw_price")),
                        "adjusted_price": new_price,
                        "adjustment_amount": ai_adjustment.get("adjustment_amount", 0),
                        "reasoning": ai_adjustment.get("adjustment_reasoning", "AI price adjustment")
                    }
                    adjustment_applied = True
            
            adjusted_comps.append(original_comp)
        
        if not adjustment_applied:
            return comps  # Return original comps if no changes were made
        
        print(f"[AI Adjustment] Successfully adjusted {len(adjusted_comps)} comps")
        
        return adjusted_comps
        
    except Exception as e:
        print(f"[AI Adjustment] Error adjusting comp prices: {e}")
        # Return original comps if AI adjustment fails
        return comps


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
