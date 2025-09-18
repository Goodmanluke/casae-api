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
    
    # Prepare the prompt for OpenAI
    prompt = f"""
    You are a professional real estate appraiser. I need you to adjust comparable property prices based on condition and renovation differences.
    
    SUBJECT PROPERTY DETAILS:
    - Condition: {subject_condition}
    - Renovations: {', '.join(subject_renovations) if subject_renovations else 'None'}
    
    BASELINE (DEFAULT) ASSUMPTIONS:
    - Default Condition: {default_condition}
    - Default Renovations: {', '.join(default_renovations)}
    
    COMPARABLE PROPERTIES TO ADJUST:
    {json.dumps(comps, indent=2)}
    
    TASK:
    Adjust the raw_price of each comparable property to reflect what it would sell for if it had the same condition and renovations as the subject property.
    
    CONDITION ADJUSTMENT GUIDELINES:
    - Poor condition: Typically 10-20% below market value
    - Fair condition: Typically 5-10% below market value  
    - Good condition: Market value baseline
    - Excellent condition: Typically 5-15% above market value
    
    RENOVATION ADJUSTMENT GUIDELINES:
    - Kitchen renovation: +/- $15,000-$30,000 depending on scope
    - Bathroom renovation: +/- $8,000-$15,000 per bathroom
    - Flooring renovation: +/- $5,000-$15,000 depending on materials
    - Roof renovation: +/- $10,000-$25,000 depending on size
    - Windows renovation: +/- $8,000-$20,000 depending on quantity
    
    Consider the size and value of each property when applying adjustments. Higher-value homes should receive proportionally higher adjustment amounts.
    
    RESPOND IN JSON FORMAT:
    {{
        "adjusted_comps": [
            {{
                "id": "comp_id",
                "address": "comp_address", 
                "original_price": original_raw_price,
                "adjusted_price": adjusted_raw_price,
                "adjustment_amount": difference,
                "adjustment_reasoning": "Brief explanation of why this adjustment was made"
            }}
        ],
        "overall_reasoning": "Summary of the adjustment methodology used"
    }}
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional real estate appraiser skilled at making precise property value adjustments based on condition and renovation differences."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            timeout=15,
        )
        
        ai_result = _safe_json(response.choices[0].message.content, {"adjusted_comps": [], "overall_reasoning": "AI adjustment failed"})
        
        # Apply the AI adjustments to the original comps
        adjusted_comps = []
        for i, comp in enumerate(comps):
            original_comp = comp.copy()
            
            # Find the corresponding AI adjustment
            ai_adjustment = None
            if i < len(ai_result.get("adjusted_comps", [])):
                ai_adjustment = ai_result["adjusted_comps"][i]
            
            if ai_adjustment and "adjusted_price" in ai_adjustment:
                # Apply the AI-suggested price adjustment
                original_comp["raw_price"] = ai_adjustment["adjusted_price"]
                original_comp["ai_adjustment"] = {
                    "original_price": ai_adjustment.get("original_price", comp.get("raw_price")),
                    "adjusted_price": ai_adjustment["adjusted_price"],
                    "adjustment_amount": ai_adjustment.get("adjustment_amount", 0),
                    "reasoning": ai_adjustment.get("adjustment_reasoning", "AI price adjustment")
                }
            
            adjusted_comps.append(original_comp)
        
        print(f"[AI Adjustment] Successfully adjusted {len(adjusted_comps)} comps for condition: {subject_condition}, renovations: {subject_renovations}")
        print(f"[AI Adjustment] Overall reasoning: {ai_result.get('overall_reasoning', 'N/A')}")
        
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
