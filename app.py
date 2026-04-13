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
    session_id: Optional[str] = None
    message: str
    # If omitted, the default_backend from config.json is used.
    backend: Optional[Backend] = None
    # If omitted, the backend's section from config.json is used.
    # Any keys supplied here override the config-file values.
    backend_config: Optional[dict] = None


class PaperOut(BaseModel):
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
    session_id: str
    status: str
    message: Optional[str] = None
    reason: Optional[str] = None
    papers: Optional[list[PaperOut]] = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in _sessions:
        backend = request.backend or _config.get("default_backend", "ollama")
        # Start from the file-level defaults for this backend, then apply any
        # per-request overrides on top so callers can tweak individual keys.
        file_defaults: dict = _config.get(backend, {})
        merged_config = {**file_defaults, **(request.backend_config or {})}
        logger.info("Creating new session: %s  backend=%s", session_id, backend)
        _sessions[session_id] = Agent(
            session=session_id,
            backend=backend,
            backend_config=merged_config,
        )

    agent = _sessions[session_id]

    try:
        result = agent.chatAgent(request.message)
    except Exception as e:
        logger.exception("Agent error in session %s", session_id)
        raise HTTPException(status_code=500, detail=str(e))

    if isinstance(result, PaperSearchResult):
        papers = [
            PaperOut(
                index=p.index,
                title=p.title,
                authors=p.authors,
                year=p.year,
                journal=p.journal,
                url=p.url,
                relevance=p.relevance,
                confidence=p.confidence,
                evidence=p.evidence,
            )
            for p in result.papers
        ]
        return ChatResponse(
            session_id=session_id,
            status=result.status,
            message=result.message,
            reason=result.reason,
            papers=papers,
        )

    if isinstance(result, AgentTextResponse):
        return ChatResponse(
            session_id=session_id,
            status=result.status,
            message=result.message,
            reason=result.reason,
            papers=None,
        )

    return ChatResponse(
        session_id=session_id,
        status="conversational",
        message=str(result),
        reason=None,
        papers=None,
    )


@app.get("/api/config")
async def get_config():
    """Return the active backend configuration. Secrets are redacted."""
    safe = {}
    for key, value in _config.items():
        if isinstance(value, dict):
            safe[key] = {k: "***" if "key" in k or "secret" in k or "password" in k else v
                         for k, v in value.items()}
        else:
            safe[key] = value
    return JSONResponse(safe)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
