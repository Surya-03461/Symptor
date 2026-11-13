"""
Flask Application — Agentic AI Health Symptom Checker
Powered by IBM Granite via IBM watsonx.ai (IBM Cloud Lite)
"""

import os
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from medical_agent import HealthSymptomAgent
from symptom_db import DISCLAIMER, MULTILINGUAL_GREETINGS
from guardrails import run_input_guardrails

# Load environment variables from .env
load_dotenv()

# Use absolute paths so Flask finds templates/static on Vercel's filesystem
_BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_BASE, "templates"),
    static_folder=os.path.join(_BASE, "static"),
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "symptoms-checker-secret-2024")

# Initialize the IBM Granite-powered agent (singleton)
agent = HealthSymptomAgent()


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat UI."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body: { "message": "I have a sore throat and fever" }
    Returns: structured health guidance JSON
    """
    data = request.get_json(silent=True)
    if not data or not data.get("message"):
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"].strip()
    if len(user_message) > 2000:
        return jsonify({"error": "Message too long. Please keep it under 2000 characters."}), 400

    # ── INPUT GUARDRAILS ──────────────────────────────────────────
    # Uses session ID as the rate-limit key; falls back to IP address.
    session_id = session.get("_id", request.remote_addr or "anonymous")
    passed, cleaned_message, input_warnings = run_input_guardrails(user_message, session_id)

    if not passed:
        # Guardrail blocked the request — return 400 with friendly reason
        return jsonify({
            "error": cleaned_message,          # block_reason is returned in cleaned_message
            "guardrail_blocked": True,
            "urgency": "low",
            "urgency_info": {},
            "matched_symptoms": [],
            "is_emergency": False,
            "language": "en",
            "source": "guardrail",
        }), 400

    # Use guardrail-cleaned message for all downstream processing
    user_message = cleaned_message

    # Maintain conversation history in session (last 10 turns)
    if "history" not in session:
        session["history"] = []

    session["history"].append({"role": "user", "content": user_message})
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    # Run agentic analysis (output guardrails applied inside the agent)
    result = agent.analyze(user_message)

    # Merge input + output guardrail warnings for transparency
    all_warnings = input_warnings + result.get("guardrail_warnings", [])

    # Store assistant response in session
    session["history"].append({"role": "assistant", "content": result.get("response", "")})
    session.modified = True

    return jsonify({
        "response": result.get("response", ""),
        "urgency": result.get("urgency", "low"),
        "urgency_info": result.get("urgency_info", {}),
        "matched_symptoms": result.get("matched_symptoms", []),
        "is_emergency": result.get("is_emergency", False),
        "language": result.get("language", "en"),
        "source": result.get("source", "symptom_database"),
        "guardrail_warnings": all_warnings,
    })


@app.route("/api/disclaimer", methods=["GET"])
def get_disclaimer():
    """Returns the medical disclaimer text."""
    return jsonify({"disclaimer": DISCLAIMER})


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for deployment."""
    return jsonify({
        "status": "ok",
        "service": "AI Health Symptom Checker",
        "model": os.getenv("IBM_MODEL_ID", "ibm/granite-3-1-8b-base"),
        "ibm_granite_active": agent._ready,
    })


@app.route("/api/clear", methods=["POST"])
def clear_session():
    """Clears conversation history."""
    session.pop("history", None)
    return jsonify({"status": "cleared"})


@app.route("/api/symptoms", methods=["GET"])
def list_symptoms():
    """Returns list of symptoms the system can analyze."""
    from symptom_db import SYMPTOM_DATABASE
    return jsonify({
        "symptoms": list(SYMPTOM_DATABASE.keys()),
        "count": len(SYMPTOM_DATABASE),
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"\nAI Health Symptom Checker starting on http://localhost:{port}")
    print(f"   IBM Granite Active: {agent._ready}")
    print(f"   Debug Mode: {debug}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
