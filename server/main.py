# Re-export app from env.py — Dockerfile entry: uvicorn server.main:app
from server.env import app  # noqa: F401
