import json
import os
from typing import Any, Dict, List
from litellm import completion
from pydantic import BaseModel, ValidationError, Field

# You can replace these with other models as needed but this is the one we suggest for this lab.
MODEL = "groq/llama-3.3-70b-versatile"

class ItinerarySchema(BaseModel):
    destination: str = Field(min_length=1, max_length=120)
    price_range: str
    ideal_visit_times: List[str]
    top_attractions: List[str]
    
def get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return key

def _extract_text(resp: Any) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        pass
    try:
        return resp["choices"][0]["text"]
    except Exception:
        raise ValueError("Unexpected LLM response format; cannot extract output text.")

def parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Model returned empty output")
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}")

def get_itinerary(destination: str) -> Dict[str, Any]:
    """
    Returns a JSON-like dict with keys:
      - destination
      - price_range
      - ideal_visit_times
      - top_attractions
    """
    # implement litellm call here to generate a structured travel itinerary for the given destination

    # See https://docs.litellm.ai/docs/ for reference.

    key = get_api_key()
    system_msg = (
        "Return ONLY valid JSON. Do not include markdown, code fences, or extra text. "
        "The JSON must match this schema exactly:\n"
        '{\n'
        '  "destination": string,\n'
        '  "price_range": string,\n'
        '  "ideal_visit_times": [string, ...],\n'
        '  "top_attractions": [string, ...]\n'
        '}\n'
        "Constraints: ideal_visit_times = 3-6 items, top_attractions = 5-8 items."
    )
    user_msg = f"Destination: {destination}"
    resp = completion(
        model=MODEL,
        api_key=key,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    )
    text = _extract_text(resp)
    obj = parse_json(text)
    try:
        validated = ItinerarySchema.model_validate(obj)
    except ValidationError as e:
        raise ValueError(f"Model output did not match expected schema: {e}")
    return validated.model_dump()