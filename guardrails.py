"""
Guardrails for the AI Health Symptom Checker
=============================================
Implements layered safety controls relevant to a medical AI assistant:

  INPUT GUARDRAILS
  1. Input Sanitization    — strip injection attempts, control chars, HTML
  2. Topic Relevance       — block non-health / off-topic queries
  3. Prompt Injection Guard — detect adversarial instruction overrides
  4. Rate Limiting         — cap requests per session to prevent abuse
  5. PII Detection         — warn & strip personally identifiable information

  OUTPUT GUARDRAILS
  6. Diagnosis Language Filter  — block definitive diagnostic statements
  7. Prescription Filter        — block specific drug dosage recommendations
  8. Disclaimer Enforcement     — guarantee disclaimer is always present
  9. Hallucination Hedge Check  — ensure hedging language is present
 10. Emergency Escalation Check — verify emergency response is included when needed
"""

import re
import time
from collections import defaultdict
from typing import Tuple


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

# Phrases that indicate the model is diagnosing rather than educating
DIAGNOSIS_PHRASES = [
    r"\byou have\b(?! a cold| a cough| symptoms)",   # "you have diabetes" etc.
    r"\byou are diagnosed\b",
    r"\bmy diagnosis is\b",
    r"\bthe diagnosis is\b",
    r"\bconfirmed case of\b",
    r"\bdefinitely have\b",
    r"\bcertainly have\b",
    r"\btest confirms\b",
]

# Prescription-style phrases the model must not output
PRESCRIPTION_PHRASES = [
    r"\btake \d+\s*(mg|ml|mcg|g)\b",          # "take 500mg"
    r"\bprescribe\b",
    r"\bprescription for\b",
    r"\bdosage of \d+",
    r"\binject\b.*\binsulin\b",
    r"\badminister\b.*\bmedication\b",
]

# Off-topic categories — the model should not engage with these at all
OFF_TOPIC_PATTERNS = [
    r"\b(write|create|generate|code|program|script|function|class|sql|python|javascript)\b",
    r"\b(recipe|cook|bake|ingredient)\b",
    r"\b(stock|invest|crypto|bitcoin|finance|money|loan)\b",
    r"\b(weather|temperature outside|forecast)\b",
    r"\b(politics|election|president|government policy)\b",
    r"\b(religion|god|allah|jesus|prayer)\b",
    r"\b(joke|funny|meme|entertainment)\b",
    r"\b(homework|essay|assignment|exam answer)\b",
]

# Prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|prior|all)\s+instructions?",
    r"disregard\s+(your|the)\s+(instructions?|rules?|guidelines?)",
    r"you are now\s+(a|an)\s+(?!health|medical)",   # role-switch except health roles
    r"act as\s+(?!a\s+health|a\s+medical|an?\s+AI health)",
    r"forget\s+(everything|your|the)\s+(training|instructions?|rules?)",
    r"jailbreak",
    r"dan mode",
    r"developer mode",
    r"<\|system\|>",          # direct system prompt injection
    r"<\|user\|>",
    r"\[INST\]",              # Llama-style injection
    r"###\s*instruction",
]

# PII patterns — detected and redacted before storage / logging
PII_PATTERNS = {
    "phone":   r"\b(\+?[\d\-\(\)\s]{10,15})\b",
    "email":   r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "ssn":     r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "dob":     r"\b(0[1-9]|[12]\d|3[01])[-/](0[1-9]|1[0-2])[-/](19|20)\d{2}\b",
    "credit_card": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",   # India Aadhaar
}

# Minimum hedging phrases the output must contain (at least one)
HEDGING_PHRASES = [
    "may", "might", "could", "possible", "possibly", "likely", "suggest",
    "consult", "doctor", "healthcare", "professional", "educational",
    "not a diagnosis", "not medical advice",
]

DISCLAIMER_MARKER = "MEDICAL DISCLAIMER"

# Rate limiting: max requests per session within a rolling window
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# In-memory rate limit store: { session_id: [timestamp, ...] }
_rate_limit_store: dict = defaultdict(list)


# ─────────────────────────────────────────────────────────────────
# Guardrail Result
# ─────────────────────────────────────────────────────────────────

class GuardrailResult:
    """Carries the outcome of a guardrail check."""

    def __init__(
        self,
        passed: bool,
        block_reason: str = "",
        sanitized_text: str = "",
        warnings: list = None,
    ):
        self.passed = passed
        self.block_reason = block_reason          # human-readable reason if blocked
        self.sanitized_text = sanitized_text      # cleaned text (may differ from input)
        self.warnings = warnings or []            # non-blocking advisory messages

    def __bool__(self):
        return self.passed


# ─────────────────────────────────────────────────────────────────
# INPUT GUARDRAILS
# ─────────────────────────────────────────────────────────────────

def guardrail_sanitize_input(text: str) -> GuardrailResult:
    """
    Guardrail 1 — Input Sanitization
    - Strip HTML/XML tags
    - Remove control characters
    - Collapse excessive whitespace
    - Enforce minimum meaningful length
    """
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", text)
    # Remove null bytes and control characters (except newline/tab)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    if len(cleaned) < 3:
        return GuardrailResult(
            passed=False,
            block_reason="Input is too short. Please describe your symptoms in more detail.",
            sanitized_text=cleaned,
        )

    return GuardrailResult(passed=True, sanitized_text=cleaned)


def guardrail_topic_relevance(text: str) -> GuardrailResult:
    """
    Guardrail 2 — Topic Relevance
    Blocks queries that are clearly off-topic for a health assistant.
    """
    text_lower = text.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return GuardrailResult(
                passed=False,
                block_reason=(
                    "I'm a Health Symptom Checker and can only assist with health and medical "
                    "symptom queries. Please describe any symptoms or health concerns you have."
                ),
                sanitized_text=text,
            )
    return GuardrailResult(passed=True, sanitized_text=text)


def guardrail_prompt_injection(text: str) -> GuardrailResult:
    """
    Guardrail 3 — Prompt Injection Detection
    Detects attempts to override system instructions or jailbreak the model.
    """
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                block_reason=(
                    "Your message contains instructions that cannot be processed. "
                    "I'm here to help with health symptom queries only. "
                    "Please describe your symptoms."
                ),
                sanitized_text=text,
            )
    return GuardrailResult(passed=True, sanitized_text=text)


def guardrail_rate_limit(session_id: str) -> GuardrailResult:
    """
    Guardrail 4 — Rate Limiting (per session)
    Allows up to RATE_LIMIT_MAX_REQUESTS per RATE_LIMIT_WINDOW_SECONDS.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    timestamps = _rate_limit_store[session_id]

    # Prune old timestamps outside the window
    timestamps[:] = [t for t in timestamps if t > window_start]

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        return GuardrailResult(
            passed=False,
            block_reason=(
                f"Too many requests. Please wait a moment before sending another message. "
                f"(Limit: {RATE_LIMIT_MAX_REQUESTS} requests per minute)"
            ),
            sanitized_text="",
        )

    timestamps.append(now)
    return GuardrailResult(passed=True, sanitized_text="")


def guardrail_pii_detection(text: str) -> GuardrailResult:
    """
    Guardrail 5 — PII Detection & Redaction
    Detects and redacts phone numbers, emails, SSN, Aadhaar, DOB, credit cards.
    Non-blocking: warns user and returns redacted text.
    """
    cleaned = text
    warnings = []
    pii_found = []

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, cleaned)
        if matches:
            pii_found.append(pii_type)
            # Redact matches
            cleaned = re.sub(pattern, f"[{pii_type.upper()} REDACTED]", cleaned)

    if pii_found:
        warnings.append(
            f"Your message appeared to contain personal information "
            f"({', '.join(pii_found)}) which has been removed for your privacy. "
            f"There is no need to share personal identifiers to get health guidance."
        )

    return GuardrailResult(passed=True, sanitized_text=cleaned, warnings=warnings)


# ─────────────────────────────────────────────────────────────────
# OUTPUT GUARDRAILS
# ─────────────────────────────────────────────────────────────────

def guardrail_no_diagnosis(response_text: str) -> GuardrailResult:
    """
    Guardrail 6 — Diagnosis Language Filter
    Strips or rewrites definitive diagnostic language in AI output.
    """
    warnings = []
    cleaned = response_text

    for pattern in DIAGNOSIS_PHRASES:
        if re.search(pattern, cleaned, re.IGNORECASE):
            # Replace with hedged language
            cleaned = re.sub(
                pattern,
                lambda m: m.group().replace("you have", "your symptoms may suggest")
                                   .replace("You have", "Your symptoms may suggest")
                                   .replace("you are diagnosed", "a healthcare provider might evaluate")
                                   .replace("the diagnosis is", "one possibility worth discussing with a doctor is")
                                   .replace("my diagnosis is", "based on the described symptoms")
                                   .replace("confirmed case of", "possible presentation of")
                                   .replace("definitely have", "may be experiencing")
                                   .replace("certainly have", "may be experiencing")
                                   .replace("test confirms", "symptoms may indicate"),
                cleaned,
                flags=re.IGNORECASE,
            )
            warnings.append("Diagnostic language was softened to educational guidance.")

    return GuardrailResult(passed=True, sanitized_text=cleaned, warnings=warnings)


def guardrail_no_prescription(response_text: str) -> GuardrailResult:
    """
    Guardrail 7 — Prescription Filter
    Removes specific dosage recommendations from AI output.
    """
    warnings = []
    cleaned = response_text

    for pattern in PRESCRIPTION_PHRASES:
        if re.search(pattern, cleaned, re.IGNORECASE):
            cleaned = re.sub(
                pattern,
                "[specific dosage omitted — consult a pharmacist or doctor]",
                cleaned,
                flags=re.IGNORECASE,
            )
            warnings.append("Specific drug dosage was removed. Always consult a healthcare professional for prescriptions.")

    return GuardrailResult(passed=True, sanitized_text=cleaned, warnings=warnings)


def guardrail_disclaimer_present(response_text: str, disclaimer: str) -> GuardrailResult:
    """
    Guardrail 8 — Disclaimer Enforcement
    Ensures the medical disclaimer is always appended to the response.
    """
    if DISCLAIMER_MARKER not in response_text.upper():
        response_text = response_text.rstrip() + f"\n\n{disclaimer}"
        return GuardrailResult(
            passed=True,
            sanitized_text=response_text,
            warnings=["Medical disclaimer was automatically appended."],
        )
    return GuardrailResult(passed=True, sanitized_text=response_text)


def guardrail_hedging_language(response_text: str) -> GuardrailResult:
    """
    Guardrail 9 — Hallucination Hedge Check
    Verifies the response contains appropriate uncertainty language.
    If completely absent, appends a standard hedge sentence.
    """
    text_lower = response_text.lower()
    has_hedge = any(phrase in text_lower for phrase in HEDGING_PHRASES)

    if not has_hedge:
        hedge = (
            "\n\n*Please note: The above information is for educational purposes only "
            "and is not a substitute for professional medical evaluation.*"
        )
        return GuardrailResult(
            passed=True,
            sanitized_text=response_text + hedge,
            warnings=["Hedging language was added to prevent overconfident interpretation."],
        )

    return GuardrailResult(passed=True, sanitized_text=response_text)


def guardrail_emergency_escalation(response_text: str, is_emergency: bool) -> GuardrailResult:
    """
    Guardrail 10 — Emergency Escalation Check
    If the agent detected an emergency but the response doesn't prominently
    mention calling emergency services, injects a mandatory emergency notice.
    """
    if not is_emergency:
        return GuardrailResult(passed=True, sanitized_text=response_text)

    emergency_markers = ["911", "112", "108", "emergency service", "call emergency", "ambulance"]
    text_lower = response_text.lower()

    if not any(m in text_lower for m in emergency_markers):
        notice = (
            "\n\n🚨 **IMPORTANT — EMERGENCY DETECTED:**\n"
            "Based on the symptoms described, please call emergency services immediately:\n"
            "• **India:** 108 (Ambulance) / 112 (National Emergency)\n"
            "• **USA/Canada:** 911\n"
            "• **Europe:** 112\n"
            "Do NOT wait — seek immediate medical attention."
        )
        return GuardrailResult(
            passed=True,
            sanitized_text=response_text + notice,
            warnings=["Emergency escalation notice was injected."],
        )

    return GuardrailResult(passed=True, sanitized_text=response_text)


# ─────────────────────────────────────────────────────────────────
# Composed Pipelines
# ─────────────────────────────────────────────────────────────────

def run_input_guardrails(text: str, session_id: str = "default") -> Tuple[bool, str, list]:
    """
    Runs all input guardrails in sequence.
    Returns: (passed: bool, cleaned_text: str, warnings: list)
    If any guardrail blocks, returns (False, block_reason, [])
    """
    warnings = []

    # 1. Rate limit (session-level)
    result = guardrail_rate_limit(session_id)
    if not result:
        return False, result.block_reason, []

    # 2. Sanitize input
    result = guardrail_sanitize_input(text)
    if not result:
        return False, result.block_reason, []
    text = result.sanitized_text

    # 3. Prompt injection check
    result = guardrail_prompt_injection(text)
    if not result:
        return False, result.block_reason, []

    # 4. Topic relevance check
    result = guardrail_topic_relevance(text)
    if not result:
        return False, result.block_reason, []

    # 5. PII detection & redaction (non-blocking)
    result = guardrail_pii_detection(text)
    text = result.sanitized_text
    warnings.extend(result.warnings)

    return True, text, warnings


def run_output_guardrails(
    response_text: str,
    is_emergency: bool,
    disclaimer: str,
) -> Tuple[str, list]:
    """
    Runs all output guardrails in sequence.
    Returns: (cleaned_response: str, warnings: list)
    Output guardrails are non-blocking — they always clean and return text.
    """
    warnings = []

    # 6. Remove diagnostic language
    result = guardrail_no_diagnosis(response_text)
    response_text = result.sanitized_text
    warnings.extend(result.warnings)

    # 7. Remove prescription language
    result = guardrail_no_prescription(response_text)
    response_text = result.sanitized_text
    warnings.extend(result.warnings)

    # 8. Ensure disclaimer is present
    result = guardrail_disclaimer_present(response_text, disclaimer)
    response_text = result.sanitized_text
    warnings.extend(result.warnings)

    # 9. Ensure hedging language
    result = guardrail_hedging_language(response_text)
    response_text = result.sanitized_text
    warnings.extend(result.warnings)

    # 10. Emergency escalation
    result = guardrail_emergency_escalation(response_text, is_emergency)
    response_text = result.sanitized_text
    warnings.extend(result.warnings)

    return response_text, warnings
