"""FastAPI application for Paper Pal.

This module exposes the web UI and REST endpoints used by the frontend.
It also owns lightweight in-memory session management where each session maps
to a dedicated `Agent` instance.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import Agent, AgentTextResponse, Backend, PaperSearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load config.json
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "config.json"

def _load_config() -> dict:
    """Load application configuration from disk and hydrate env-backed secrets."""
    if not _CONFIG_PATH.exists():
        logger.warning("config.json not found at %s; using built-in defaults", _CONFIG_PATH)
        return {"default_backend": "ollama", "ollama": {}}
    with _CONFIG_PATH.open() as f:
        cfg = json.load(f)
    # Resolve Venice API key from env var if not set in file
    if "venice" in cfg and not cfg["venice"].get("api_key"):
        cfg["venice"]["api_key"] = os.environ.get("VENICE_API_KEY", "")
    logger.info("Loaded config from %s  default_backend=%s", _CONFIG_PATH, cfg.get("default_backend"))
    return cfg

_config: dict = _load_config()

# ---------------------------------------------------------------------------

app = FastAPI(title="Paper Pal")

# In-memory session store: session_id -> Agent
_sessions: dict[str, Agent] = {}


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    session_id: Optional[str] = None
    message: str
    backend: Optional[Backend] = None
    backend_config: Optional[dict] = None


class PaperOut(BaseModel):
    """Serialized paper returned by the API."""

    index: int
    title: str
    authors: Optional[str] = None
    year: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    relevance: str
    confidence: float = 0.5
    evidence: Optional[str] = None


class ChatResponse(BaseModel):
    """Unified API response for both conversational and ranked outputs."""

    session_id: str
    status: str
    message: Optional[str] = None
    reason: Optional[str] = None
    papers: Optional[list[PaperOut]] = None
    telemetry: Optional[dict] = None


def _is_sensitive_key(key: str) -> bool:
    """Return True for config keys that should be redacted in API responses."""
    lowered = key.lower()
    return "key" in lowered or "secret" in lowered or "password" in lowered


def _redact_config(config: dict) -> dict:
    """Return a shallow-redacted copy of the config for safe client exposure."""
    redacted: dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            redacted[key] = {inner_key: "***" if _is_sensitive_key(inner_key) else inner_value for inner_key, inner_value in value.items()}
        else:
            redacted[key] = value
    return redacted


def _merge_backend_config(backend: Backend, overrides: Optional[dict]) -> dict:
    """Merge backend defaults from config with request-level overrides."""
    file_defaults = _config.get(backend, {})
    return {**file_defaults, **(overrides or {})}


def _create_agent(session_id: str, backend: Backend, backend_config: Optional[dict]) -> Agent:
    """Construct a new per-session agent instance."""
    merged_config = _merge_backend_config(backend, backend_config)
    logger.info("Creating new session: %s  backend=%s", session_id, backend)
    return Agent(session=session_id, backend=backend, backend_config=merged_config)


def _get_agent(request: ChatRequest, session_id: str) -> Agent:
    """Get or create the session agent for an incoming request."""
    agent = _sessions.get(session_id)
    if agent is None:
        backend = request.backend or _config.get("default_backend", "ollama")
        agent = _create_agent(session_id, backend, request.backend_config)
        _sessions[session_id] = agent
    return agent


def _build_paper_outputs(result: PaperSearchResult) -> list[PaperOut]:
    """Convert internal paper models into API response models."""
    return [
        PaperOut(
            index=paper.index,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            journal=paper.journal,
            url=paper.url,
            relevance=paper.relevance,
            confidence=paper.confidence,
            evidence=paper.evidence,
        )
        for paper in result.papers
    ]


def _build_chat_response(session_id: str, result) -> ChatResponse:
    """Normalize agent outputs into a stable API response envelope."""
    if isinstance(result, PaperSearchResult):
        return ChatResponse(
            session_id=session_id,
            status=result.status,
            message=result.message,
            reason=result.reason,
            papers=_build_paper_outputs(result),
            telemetry=result.telemetry,
        )

    if isinstance(result, AgentTextResponse):
        return ChatResponse(
            session_id=session_id,
            status=result.status,
            message=result.message,
            reason=result.reason,
            papers=None,
            telemetry=result.telemetry,
        )

    return ChatResponse(
        session_id=session_id,
        status="conversational",
        message=str(result),
        reason=None,
        papers=None,
        telemetry=None,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle one chat turn and return either papers or conversational text."""
    session_id = request.session_id or str(uuid.uuid4())
    agent = _get_agent(request, session_id)

    try:
        result = agent.chat(request.message)
    except Exception as e:
        logger.exception("Agent error in session %s", session_id)
        raise HTTPException(status_code=500, detail=str(e))
    return _build_chat_response(session_id, result)


@app.get("/api/config")
async def get_config():
    """Return the active backend configuration. Secrets are redacted."""
    return JSONResponse(_redact_config(_config))


@app.get("/")
async def index():
    """Serve the single-page frontend."""
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
