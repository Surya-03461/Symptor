"""
Agentic AI Medical Reasoner
Uses IBM Granite via IBM watsonx.ai to reason over symptoms,
apply tool-use patterns (symptom lookup, urgency assessment),
and generate structured, educational health guidance.
"""

import os
import json
import re
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from symptom_db import (
    SYMPTOM_DATABASE,
    URGENCY_LEVELS,
    EMERGENCY_KEYWORDS,
    DISCLAIMER,
    MULTILINGUAL_GREETINGS,
)
from guardrails import run_output_guardrails


# ─────────────────────────────────────────
# Tool Functions (Agentic Tool-Use Pattern)
# ─────────────────────────────────────────

def tool_lookup_symptoms(symptom_text: str) -> dict:
    """
    Tool: Searches the local curated symptom database for matching conditions.
    Returns matched symptom entries with conditions, urgency, home care, and referral triggers.
    """
    symptom_text_lower = symptom_text.lower()
    results = {}
    for key, data in SYMPTOM_DATABASE.items():
        if key in symptom_text_lower:
            results[key] = data
    return results


def tool_check_emergency(symptom_text: str) -> bool:
    """
    Tool: Scans input for emergency keyword phrases.
    Returns True if any emergency red-flag is detected.
    """
    text_lower = symptom_text.lower()
    return any(kw in text_lower for kw in EMERGENCY_KEYWORDS)


def tool_assess_urgency(matched_symptoms: dict) -> str:
    """
    Tool: Determines the highest urgency level across all matched symptoms.
    Priority: emergency > high > moderate > low
    """
    priority = {"emergency": 4, "high": 3, "moderate": 2, "low": 1}
    highest = "low"
    for _, data in matched_symptoms.items():
        level = data.get("urgency", "low")
        if priority.get(level, 0) > priority.get(highest, 0):
            highest = level
    return highest


def tool_detect_language(text: str) -> str:
    """
    Tool: Simple heuristic language detection based on Unicode character ranges.
    Returns ISO 639-1 language code (defaults to 'en').
    """
    if re.search(r'[\u0900-\u097F]', text):
        return "hi"
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    if any(word in text.lower() for word in ["je", "le", "les", "une", "avec", "merci"]):
        return "fr"
    if any(word in text.lower() for word in ["hola", "tengo", "dolor", "fiebre", "gracias"]):
        return "es"
    return "en"


# ─────────────────────────────────────────
# IBM Granite Agent
# ─────────────────────────────────────────

class HealthSymptomAgent:
    """
    Agentic AI Health Symptom Checker powered by IBM Granite via watsonx.ai.
    Uses a ReAct-style (Reason + Act) loop:
      1. Detect language & emergency flags (tools)
      2. Look up symptom database (tool)
      3. Assess urgency (tool)
      4. Call IBM Granite for enriched, empathetic health guidance
      5. Compose structured response
    """

    def __init__(self):
        self.model = None
        self._init_ibm_client()

    def _init_ibm_client(self):
        """Initialize IBM watsonx.ai client using credentials from .env"""
        api_key = os.getenv("IBM_API_KEY")
        project_id = os.getenv("IBM_PROJECT_ID")
        url = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        if not api_key or not project_id:
            print("⚠️  IBM credentials not set. Running in fallback (database-only) mode.")
            self.model = None
            return

        try:
            credentials = Credentials(url=url, api_key=api_key)
            client = APIClient(credentials)

            self.model = ModelInference(
                model_id=os.getenv("IBM_MODEL_ID", "ibm/granite-13b-instruct-v2"),
                api_client=client,
                project_id=project_id,
                params={
                    GenParams.MAX_NEW_TOKENS: 1024,
                    GenParams.MIN_NEW_TOKENS: 50,
                    GenParams.TEMPERATURE: 0.3,
                    GenParams.TOP_P: 0.9,
                    GenParams.REPETITION_PENALTY: 1.1,
                    GenParams.STOP_SEQUENCES: ["Human:", "User:"],
                },
            )
            print("✅ IBM Granite model initialized successfully.")
        except Exception as e:
            print(f"⚠️  IBM Granite initialization failed: {e}")
            self.model = None

    def _build_granite_prompt(
        self,
        user_input: str,
        matched_symptoms: dict,
        urgency: str,
        language: str,
    ) -> str:
        """Builds a structured prompt for IBM Granite following its instruction format."""

        db_summary = ""
        for sym, data in matched_symptoms.items():
            conditions = ", ".join(data["conditions"][:4])
            db_summary += f"\n- Symptom '{sym}': possible conditions include {conditions}."

        urgency_info = URGENCY_LEVELS.get(urgency, URGENCY_LEVELS["low"])

        prompt = f"""<|system|>
You are an empathetic, knowledgeable AI Health Symptom Checker. Your role is to provide educational health information based on symptoms described by users.

IMPORTANT RULES:
1. You provide EDUCATIONAL information only — NOT medical diagnosis or treatment.
2. Always recommend consulting a licensed healthcare professional.
3. Be empathetic, clear, and non-alarmist.
4. Structure your response with: Possible Conditions, Urgency Assessment, Home Care Tips, When to See a Doctor.
5. End every response with the medical disclaimer.
6. Respond in the same language the user used if possible.

Verified medical context from trusted database (WHO/CDC/NIH sources):
{db_summary if db_summary else "No exact matches — provide general guidance based on symptoms described."}

Urgency level assessed: {urgency_info['label']} — {urgency_info['description']}
<|user|>
{user_input}
<|assistant|>
"""
        return prompt

    def _build_fallback_response(
        self,
        user_input: str,
        matched_symptoms: dict,
        urgency: str,
        is_emergency: bool,
    ) -> dict:
        """
        Fallback response builder using the symptom database directly
        when IBM Granite is unavailable.
        """
        urgency_info = URGENCY_LEVELS.get(urgency, URGENCY_LEVELS["low"])

        if is_emergency:
            urgency = "emergency"
            urgency_info = URGENCY_LEVELS["emergency"]

        conditions = []
        home_care = []
        see_doctor = []
        sources = []

        for sym, data in matched_symptoms.items():
            conditions.extend(data.get("conditions", []))
            home_care.extend(data.get("home_care", []))
            see_doctor.extend(data.get("see_doctor_if", []))
            if data.get("source"):
                sources.append(data["source"])

        # Deduplicate
        conditions = list(dict.fromkeys(conditions))
        home_care = list(dict.fromkeys(home_care))
        see_doctor = list(dict.fromkeys(see_doctor))

        if not conditions:
            response_text = (
                "I couldn't find a specific match for your symptoms in my database. "
                "Please describe your symptoms in more detail (e.g., 'I have fever and sore throat'), "
                "or consult a healthcare professional for a proper evaluation."
            )
        else:
            lines = []
            lines.append(f"**Possible Conditions:**\n" + "\n".join(f"• {c}" for c in conditions[:6]))
            lines.append(f"\n**Urgency:** {urgency_info['label']}\n{urgency_info['description']}")
            if home_care:
                lines.append("\n**Home Care Recommendations:**\n" + "\n".join(f"• {h}" for h in home_care[:5]))
            if see_doctor:
                lines.append("\n**When to See a Doctor:**\n" + "\n".join(f"• {s}" for s in see_doctor[:5]))
            if sources:
                lines.append("\n**Sources:** " + " | ".join(sources))
            lines.append(f"\n\n{DISCLAIMER}")
            response_text = "\n".join(lines)

        return {
            "response": response_text,
            "urgency": urgency,
            "urgency_info": urgency_info,
            "matched_symptoms": list(matched_symptoms.keys()),
            "is_emergency": is_emergency,
            "source": "symptom_database",
        }

    def analyze(self, user_input: str) -> dict:
        """
        Main agentic entry point.
        Runs tool chain then calls IBM Granite for enriched response.
        """
        # Step 1: Detect language
        language = tool_detect_language(user_input)

        # Step 2: Check for emergency keywords
        is_emergency = tool_check_emergency(user_input)

        # Step 3: Look up symptom database
        matched_symptoms = tool_lookup_symptoms(user_input)

        # Step 4: Assess urgency
        urgency = "emergency" if is_emergency else tool_assess_urgency(matched_symptoms)

        urgency_info = URGENCY_LEVELS.get(urgency, URGENCY_LEVELS["low"])

        # Step 5: Call IBM Granite or use fallback
        if self.model:
            try:
                prompt = self._build_granite_prompt(user_input, matched_symptoms, urgency, language)
                result = self.model.generate_text(prompt=prompt)
                granite_text = result.strip() if isinstance(result, str) else str(result)

                # Apply output guardrails (diagnosis filter, prescription filter,
                # disclaimer enforcement, hedging check, emergency escalation)
                granite_text, guardrail_warnings = run_output_guardrails(
                    granite_text, is_emergency, DISCLAIMER
                )

                return {
                    "response": granite_text,
                    "urgency": urgency,
                    "urgency_info": urgency_info,
                    "matched_symptoms": list(matched_symptoms.keys()),
                    "is_emergency": is_emergency,
                    "language": language,
                    "source": "ibm_granite",
                    "guardrail_warnings": guardrail_warnings,
                }
            except Exception as e:
                print(f"IBM Granite inference error: {e} — falling back to DB mode.")

        # Fallback — apply output guardrails on database response too
        result = self._build_fallback_response(user_input, matched_symptoms, urgency, is_emergency)
        cleaned_response, guardrail_warnings = run_output_guardrails(
            result["response"], is_emergency, DISCLAIMER
        )
        result["response"] = cleaned_response
        result["guardrail_warnings"] = guardrail_warnings
        result["language"] = language
        return result
