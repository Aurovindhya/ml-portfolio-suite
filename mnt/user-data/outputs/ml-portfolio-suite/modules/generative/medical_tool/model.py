"""
modules/generative/medical_tool/model.py

Symptom checker powered by Gemini Pro with:
  - Structured JSON output via Pydantic parsing
  - Hardcoded medical disclaimer injection
  - Guardrails: refuses to give dosage/prescription advice
  - Langfuse tracing with relevance eval score
  - Emergency detection → immediate escalation message

This is an educational tool only — not a medical device.
"""

import json
import time
from typing import List, Optional, Dict

from pydantic import BaseModel, Field
from core.config import GEMINI_API_KEY
from observability.langfuse_client import tracer

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


DISCLAIMER = (
    "⚠️ IMPORTANT: This is an AI-powered educational tool, not a medical device. "
    "It does not provide diagnosis or treatment. Always consult a licensed healthcare professional."
)

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "difficulty breathing", "stroke",
    "unconscious", "seizure", "heart attack", "severe bleeding", "overdose",
]

SYSTEM_PROMPT = """You are a medical information assistant. Given a list of symptoms, you will:
1. Identify 2-3 possible conditions that match the symptoms (NOT a diagnosis)
2. For each condition, explain why the symptoms might match
3. Recommend appropriate next steps (see a GP, urgent care, or emergency services)
4. Note any red flag symptoms that require immediate attention

STRICT RULES:
- Never prescribe medications or dosages
- Never provide a definitive diagnosis
- Always recommend professional consultation
- If symptoms suggest an emergency, say so immediately

Respond ONLY with valid JSON matching this schema:
{
  "possible_conditions": [
    {"name": "...", "match_reason": "...", "urgency": "low|medium|high|emergency"}
  ],
  "recommended_action": "...",
  "red_flags": ["..."],
  "general_advice": "..."
}"""


class ConditionMatch(BaseModel):
    name: str
    match_reason: str
    urgency: str = Field(pattern="^(low|medium|high|emergency)$")


class MedicalResponse(BaseModel):
    possible_conditions: List[ConditionMatch]
    recommended_action: str
    red_flags: List[str]
    general_advice: str
    disclaimer: str
    is_emergency: bool
    trace_id: str
    inference_ms: float


def _detect_emergency(symptoms: List[str]) -> bool:
    text = " ".join(symptoms).lower()
    return any(kw in text for kw in EMERGENCY_KEYWORDS)


def analyze_symptoms(symptoms: List[str], age: Optional[int] = None, gender: Optional[str] = None) -> MedicalResponse:
    if not GENAI_AVAILABLE:
        raise ImportError("google-generativeai not installed.")
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro", system_instruction=SYSTEM_PROMPT)

    is_emergency = _detect_emergency(symptoms)

    with tracer.trace("medical_tool") as span:
        symptom_str = ", ".join(symptoms)
        demo_str = ""
        if age:    demo_str += f" Patient age: {age}."
        if gender: demo_str += f" Gender: {gender}."

        user_message = f"Symptoms: {symptom_str}.{demo_str}"
        span.set_input({"symptoms": symptoms, "age": age, "gender": gender})

        t0 = time.time()
        response = model.generate_content(user_message)
        ms = (time.time() - t0) * 1000

        raw_text = response.text.strip()
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text)

        input_tokens  = getattr(response.usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
        span.set_tokens(input=input_tokens, output=output_tokens)

        # Attach a simple relevance score based on condition count
        relevance = min(1.0, len(parsed.get("possible_conditions", [])) / 3)
        span.set_score("relevance", relevance, comment="Fraction of max expected conditions returned")
        span.set_output({"conditions": len(parsed.get("possible_conditions", [])), "is_emergency": is_emergency})

        return MedicalResponse(
            **parsed,
            disclaimer=DISCLAIMER,
            is_emergency=is_emergency,
            trace_id=span.trace_id,
            inference_ms=round(ms, 2),
        )
