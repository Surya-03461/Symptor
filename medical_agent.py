"""
Agentic AI Medical Reasoner
Uses IBM Granite via IBM watsonx.ai REST API (direct HTTP — no ibm-cos-sdk dependency).
Applies tool-use patterns (symptom lookup, urgency assessment) and
generates structured, educational health guidance.
"""

import os
import re
import json
import requests
from symptom_db import (
    SYMPTOM_DATABASE,
    URGENCY_LEVELS,
    EMERGENCY_KEYWORDS,
    DISCLAIMER,
)
from guardrails import run_output_guardrails


# ─────────────────────────────────────────
# IBM watsonx.ai REST endpoints
# ─────────────────────────────────────────
_IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
_GENERATE_URL  = "{url}/ml/v1/text/generation?version=2023-05-29"


# ─────────────────────────────────────────
# Tool Functions (Agentic Tool-Use Pattern)
# ─────────────────────────────────────────

def tool_lookup_symptoms(symptom_text: str) -> dict:
    """Tool: Search curated WHO/CDC/NIH symptom database for keyword matches."""
    text_lower = symptom_text.lower()
    return {key: data for key, data in SYMPTOM_DATABASE.items() if key in text_lower}


def tool_check_emergency(symptom_text: str) -> bool:
    """Tool: Scan input for 20+ emergency red-flag keyword phrases."""
    text_lower = symptom_text.lower()
    return any(kw in text_lower for kw in EMERGENCY_KEYWORDS)


def tool_assess_urgency(matched_symptoms: dict) -> str:
    """Tool: Determine highest urgency across all matched symptoms."""
    priority = {"emergency": 4, "high": 3, "moderate": 2, "low": 1}
    highest = "low"
    for _, data in matched_symptoms.items():
        level = data.get("urgency", "low")
        if priority.get(level, 0) > priority.get(highest, 0):
            highest = level
    return highest


def tool_detect_language(text: str) -> str:
    """Tool: Heuristic language detection via Unicode ranges."""
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    if any(w in text.lower() for w in ["je", "le", "les", "une", "avec", "merci"]):
        return "fr"
    if any(w in text.lower() for w in ["hola", "tengo", "dolor", "fiebre", "gracias"]):
        return "es"
    return "en"


# ─────────────────────────────────────────
# IBM IAM token (cached in-process)
# ─────────────────────────────────────────

_iam_token_cache: dict = {"token": None, "expires_at": 0}


def _get_iam_token(api_key: str) -> str:
    """Exchange IBM Cloud API key for a short-lived IAM Bearer token (cached)."""
    import time
    now = time.time()
    if _iam_token_cache["token"] and now < _iam_token_cache["expires_at"] - 60:
        return _iam_token_cache["token"]

    resp = requests.post(
        _IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    _iam_token_cache["token"] = data["access_token"]
    _iam_token_cache["expires_at"] = now + int(data.get("expires_in", 3600))
    return _iam_token_cache["token"]


# ─────────────────────────────────────────
# IBM Granite Agent
# ─────────────────────────────────────────

class HealthSymptomAgent:
    """
    Agentic AI Health Symptom Checker powered by IBM Granite via watsonx.ai REST API.
    ReAct (Reason + Act) loop:
      1. detect_language    — identify user's language
      2. check_emergency    — scan for red-flag keywords
      3. lookup_symptoms    — search WHO/CDC/NIH database
      4. assess_urgency     — classify severity level
      5. IBM Granite REST   — generate enriched health guidance
      6. Output Guardrails  — safety post-processing
    """

    def __init__(self):
        self._api_key    = os.getenv("IBM_API_KEY", "")
        self._project_id = os.getenv("IBM_PROJECT_ID", "")
        self._url        = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")
        self._model_id   = os.getenv("IBM_MODEL_ID", "ibm/granite-3-1-8b-base")
        self._ready      = bool(self._api_key and self._project_id)

        if self._ready:
            print("IBM Granite REST client configured successfully.")
        else:
            print("IBM credentials not set. Running in fallback (database-only) mode.")

    # ------------------------------------------------------------------
    def _call_granite(self, prompt: str) -> str:
        """Call IBM watsonx.ai text generation REST endpoint directly."""
        token = _get_iam_token(self._api_key)
        endpoint = _GENERATE_URL.format(url=self._url)

        payload = {
            "model_id": self._model_id,
            "project_id": self._project_id,
            "input": prompt,
            "parameters": {
                "decoding_method": "sample",
                "max_new_tokens": 1024,
                "min_new_tokens": 50,
                "temperature": 0.3,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "stop_sequences": ["Human:", "User:"],
            },
        }

        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        # watsonx.ai returns: {"results": [{"generated_text": "..."}]}
        return result["results"][0]["generated_text"].strip()

    # ------------------------------------------------------------------
    def _build_prompt(self, user_input: str, matched_symptoms: dict, urgency: str) -> str:
        """Build structured Granite instruction-format prompt."""
        db_summary = ""
        for sym, data in matched_symptoms.items():
            conditions = ", ".join(data["conditions"][:4])
            db_summary += f"\n- Symptom '{sym}': possible conditions include {conditions}."

        urgency_info = URGENCY_LEVELS.get(urgency, URGENCY_LEVELS["low"])

        return (
            "<|system|>\n"
            "You are an empathetic, knowledgeable AI Health Symptom Checker. "
            "Provide educational health information only — NOT medical diagnosis or treatment.\n\n"
            "RULES:\n"
            "1. Educational information only — never diagnose.\n"
            "2. Always recommend consulting a licensed healthcare professional.\n"
            "3. Be empathetic, clear, and non-alarmist.\n"
            "4. Structure: Possible Conditions, Urgency Assessment, Home Care Tips, When to See a Doctor.\n"
            "5. Always include the medical disclaimer at the end.\n\n"
            f"Verified context (WHO/CDC/NIH):{db_summary if db_summary else ' No exact DB match — give general guidance.'}\n\n"
            f"Urgency assessed: {urgency_info['label']} — {urgency_info['description']}\n"
            "<|user|>\n"
            f"{user_input}\n"
            "<|assistant|>\n"
        )

    # ------------------------------------------------------------------
    def _fallback_response(self, matched_symptoms: dict, urgency: str, is_emergency: bool) -> str:
        """Build structured response from local DB when Granite is unavailable."""
        urgency_info = URGENCY_LEVELS.get(
            "emergency" if is_emergency else urgency,
            URGENCY_LEVELS["low"]
        )
        conditions, home_care, see_doctor, sources = [], [], [], []

        for data in matched_symptoms.values():
            conditions.extend(data.get("conditions", []))
            home_care.extend(data.get("home_care", []))
            see_doctor.extend(data.get("see_doctor_if", []))
            if data.get("source"):
                sources.append(data["source"])

        conditions = list(dict.fromkeys(conditions))
        home_care  = list(dict.fromkeys(home_care))
        see_doctor = list(dict.fromkeys(see_doctor))

        if not conditions:
            return (
                "I couldn't find a specific match for your symptoms in my database. "
                "Please describe your symptoms in more detail (e.g., 'I have fever and sore throat'), "
                "or consult a healthcare professional for a proper evaluation."
            )

        lines = [
            "**Possible Conditions:**\n" + "\n".join(f"• {c}" for c in conditions[:6]),
            f"\n**Urgency:** {urgency_info['label']}\n{urgency_info['description']}",
        ]
        if home_care:
            lines.append("\n**Home Care Recommendations:**\n" + "\n".join(f"• {h}" for h in home_care[:5]))
        if see_doctor:
            lines.append("\n**When to See a Doctor:**\n" + "\n".join(f"• {s}" for s in see_doctor[:5]))
        if sources:
            lines.append("\n**Sources:** " + " | ".join(sources))
        lines.append(f"\n\n{DISCLAIMER}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def analyze(self, user_input: str) -> dict:
        """Main ReAct entry point — run tools then call Granite or fallback."""
        language     = tool_detect_language(user_input)
        is_emergency = tool_check_emergency(user_input)
        matched      = tool_lookup_symptoms(user_input)
        urgency      = "emergency" if is_emergency else tool_assess_urgency(matched)
        urgency_info = URGENCY_LEVELS.get(urgency, URGENCY_LEVELS["low"])

        response_text = None
        source        = "symptom_database"

        if self._ready:
            try:
                prompt        = self._build_prompt(user_input, matched, urgency)
                response_text = self._call_granite(prompt)
                source        = "ibm_granite"
            except Exception as e:
                print(f"IBM Granite REST error: {e} — falling back to DB mode.")

        if not response_text:
            response_text = self._fallback_response(matched, urgency, is_emergency)

        # Apply output guardrails
        response_text, guardrail_warnings = run_output_guardrails(
            response_text, is_emergency, DISCLAIMER
        )

        return {
            "response":          response_text,
            "urgency":           urgency,
            "urgency_info":      urgency_info,
            "matched_symptoms":  list(matched.keys()),
            "is_emergency":      is_emergency,
            "language":          language,
            "source":            source,
            "guardrail_warnings": guardrail_warnings,
        }
