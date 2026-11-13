"""
Vercel Serverless Entry Point
Wraps the Flask app for Vercel Python runtime (WSGI handler).
Vercel calls `app` (the WSGI app object) from this file.
"""
import sys
import os

# Add the project root to path so imports work from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app — Vercel looks for `app` in this file
from app import app  # noqa: F401 — `app` must be importable at module level
