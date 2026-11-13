"""
Vercel Serverless Entry Point — AI Health Symptom Checker
Vercel Python runtime calls the WSGI `app` object from this file.
"""
import sys
import os

# ── Path setup ───────────────────────────────────────────────────
# Add project root so all sibling modules (app, guardrails, etc.) are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Load env before importing app ───────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

# ── Import Flask app ─────────────────────────────────────────────
# Vercel picks up the WSGI callable named `app` at module level
from app import app  # noqa: F401
