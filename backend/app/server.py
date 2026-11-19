"""ASGI entry point:  uvicorn app.server:app

Kept separate from main.py so importing the app factory (tests, scripts) does
not create the real database or load models as a side effect.
"""

from .main import create_app

app = create_app()
