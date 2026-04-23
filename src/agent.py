"""Core Paper Pal orchestration logic.

The `Agent` coordinates three stages:
1. Decide whether to call PubMed search via tool-enabled chat model.
2. Sanitize and validate untrusted PubMed metadata blocks.
3. Rank/filter results using a structured-output reasoning model.
"""

import json
import logging
import re
import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.model import Backend, ModelType, build_llm
from src.tools import searchPubMed
from src.prompts import chatSystemPrompt, rankerSystemPrompt, rankerRepairSystemPrompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns (security: validate / sanitize untrusted PubMed data)
# ---------------------------------------------------------------------------

_PAPER_BLOCK_RE = re.compile(
    r"^Title\s*:\s+[^\n]+\n"
    r"Authors\s*:\s+[^\n]+\n"
    r"Year\s*:\s+[^\n]+\|\s+Journal\s*:\s+[^\n]+\n"
    r"URL\s*:\s+https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/"
    r"(?:\n(?:Type|MeSH|Abstract)\s*:\s+[^\n]+)*$"
)
_FIELD_RE = re.compile(r"^(?P<key>Title|URL)\s+:\s+(?P<value>.+)$", re.MULTILINE)
_PUBMED_URL_RE = re.compile(r"https://pubmed\.ncbi\.nlm\.nih\.gov/(?P<pmid>\d+)/?")
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(?P<json>\{.*?\})\s*```", re.DOTALL)

_BLOCK_SEPARATOR = "\n\n---\n\n"
_DEFAULT_TOOL_LIMIT = 10
_NO_RESULTS_MESSAGE = "No relevant papers found for your query."
_RANKING_FAILED_MESSAGE = "I couldn't reliably rank the search results. Please try again."

# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------

def _build_title_url_map(valid_blocks: list[str]) -> dict[str, str]:
    """Map validated paper titles to canonical PubMed URLs."""
    mapping: dict[str, str] = {}
    for block in valid_blocks:
        fields = {m.group("key"): m.group("value").strip() for m in _FIELD_RE.finditer(block)}
        if "Title" in fields and "URL" in fields:
            mapping[fields["Title"]] = fields["URL"]
    return mapping


def _parse_block_fields(block: str) -> dict[str, str]:
    """Parse `Key: Value` lines from a metadata block into a lowercase dict."""
    parsed: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _normalize_year_journal(parsed: dict[str, str]) -> None:
    """Split combined `year | journal` lines into independent normalized fields."""
    year_line = parsed.get("year")
    if not (year_line and "|" in year_line and "journal" not in parsed):
        return

    left, right = year_line.split("|", 1)
    parsed["year"] = left.strip()
    if ":" not in right:
        return

    j_key, j_val = right.split(":", 1)
    if j_key.strip().lower().startswith("journal"):
        parsed["journal"] = j_val.strip()


def _format_candidate_block(parsed: dict[str, str], normalized_url: str) -> str:
    """Rebuild a canonical paper block from parsed fields."""
    lines = [
        f"Title   : {parsed['title']}",
        f"Authors : {parsed['authors']}",
        f"Year    : {parsed['year']}  |  Journal : {parsed['journal']}",
        f"URL     : {normalized_url}",
    ]
    if parsed.get("type"):
        lines.append(f"Type    : {parsed['type']}")
    if parsed.get("mesh"):
        lines.append(f"MeSH    : {parsed['mesh']}")
    if parsed.get("abstract"):
        lines.append(f"Abstract: {parsed['abstract']}")
    return "\n".join(lines)


def _normalize_candidate_block(block: str) -> Optional[str]:
    """Best-effort repair for malformed blocks before final regex validation."""
    parsed = _parse_block_fields(block)
    _normalize_year_journal(parsed)

    required = ("title", "authors", "year", "journal", "url")
    if not all(parsed.get(key) for key in required):
        return None
    m = _PUBMED_URL_RE.search(parsed["url"])
    if not m:
        return None
    candidate = _format_candidate_block(parsed, f"https://pubmed.ncbi.nlm.nih.gov/{m.group('pmid')}/")
    return candidate if _PAPER_BLOCK_RE.fullmatch(candidate) else None


def _sanitize_search_results(raw_results: list[str], session: str) -> tuple[str, dict[str, str]]:
    """Validate and filter raw paper blocks before they reach the LLM.

    Discards blocks that don't match the expected structure to prevent
    prompt-injection attacks embedded in paper metadata.
    """
    valid: list[str] = []
    for batch in raw_results:
        stripped = batch.strip()
        if (
            not stripped
            or stripped == "No results found."
            or stripped.startswith("PubMed search failed:")
            or stripped.startswith("PubMed fetch failed:")
        ):
            continue
        for block in batch.split(_BLOCK_SEPARATOR):
            block = block.strip()
            if _PAPER_BLOCK_RE.fullmatch(block):
                valid.append(block)
            else:
                recovered = _normalize_candidate_block(block)
                if recovered:
                    valid.append(recovered)
                else:
                    logger.warning(
                        "[session=%s] discarding malformed result block (possible prompt injection)",
                        session,
                    )
    combined = _BLOCK_SEPARATOR.join(valid) if valid else ""
    return combined, _build_title_url_map(valid)


# ---------------------------------------------------------------------------
# JSON parse helpers (for structured output fallbacks)
# ---------------------------------------------------------------------------

def _decode_first_json_value(content: str):
    """Extract the first JSON object/array from possibly mixed model output."""
    if not content:
        return None
    candidates: list[str] = []
    fenced = _FENCED_JSON_RE.search(content)
    if fenced:
        candidates.append(fenced.group("json"))
    candidates.append(content)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for i, ch in enumerate(candidate):
            if ch not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[i:])
                return value
            except Exception:
                continue
    return None


def _extract_result_from_raw(raw_msg) -> Optional["PaperSearchResult"]:
    """Decode structured ranking output from raw LLM text as a fallback path."""
    content = getattr(raw_msg, "content", None)
    if not content or not isinstance(content, str):
        return None
    value = _decode_first_json_value(content)
    if value is None:
        return None
    try:
        if isinstance(value, dict):
            if "papers" in value:
                return PaperSearchResult(**value)
            if "results" in value and isinstance(value["results"], list):
                return PaperSearchResult(papers=value["results"])
        if isinstance(value, list):
            return PaperSearchResult(papers=value)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    """Ranked paper candidate returned by the reasoning model."""

    index: int = Field(description="1-based rank position")
    title: str
    authors: Optional[str] = "N/A"
    year: Optional[str] = "N/A"
    journal: Optional[str] = "N/A"
    url: Optional[str] = "N/A"
    relevance: str = ""
    confidence: float = 0.5
    evidence: Optional[str] = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value):
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5


class PaperSearchResult(BaseModel):
    """Structured response for research-mode turns."""

    papers: list[Paper] = Field(default_factory=list)
    message: Optional[str] = None
    status: Literal["ranked", "no_results", "failed"] = "ranked"
    reason: Optional[str] = None
    telemetry: Optional[dict] = None

    @field_validator("papers", mode="before")
    @classmethod
    def _coerce_papers_string(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return []
        return v


class AgentTextResponse(BaseModel):
    """Conversational response for non-search turns."""

    message: str
    status: Literal["conversational", "failed"] = "conversational"
    reason: Optional[str] = None
    telemetry: Optional[dict] = None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_DEFAULT_MAX_HISTORY_TURNS = 10
_MAX_SUMMARY_CHARS = 1800


class Agent:
    """Conversational research assistant backed by PubMed search.

    Each turn runs up to two LLM calls:
    1. Chat agent (with searchPubMed tool bound) — decides whether to search.
    2. Ranker — filters and ranks raw results into a structured response.
    Conversational turns skip the ranker.
    """

    def __init__(self, session: str = None, backend: Backend = "ollama", backend_config: dict = None):
        """Create a per-session agent with chat + ranking model clients.

        Args:
            session: Stable session identifier used in logs and memory.
            backend: Backend provider name (ollama/aws_bedrock/openai).
            backend_config: Optional runtime backend overrides.
        """
        self.session = session or str(uuid.uuid4())
        self.conversation_history: list[tuple[str, str]] = []
        self._history_summary: str = ""
        config = backend_config or {}
        try:
            self._max_history_turns = max(2, int(config.get("max_history_turns", _DEFAULT_MAX_HISTORY_TURNS)))
        except (TypeError, ValueError):
            self._max_history_turns = _DEFAULT_MAX_HISTORY_TURNS

        chat_llm = build_llm(backend, config, ModelType.CHAT)
        reasoning_llm = build_llm(backend, config, ModelType.REASONING)
        self._search_agent = chat_llm.bind_tools([searchPubMed])
        self._ranker_agent = reasoning_llm.with_structured_output(PaperSearchResult, include_raw=True)
        self._llm = chat_llm

        logger.info(
            "[session=%s] Agent initialised | backend=%s model=%s",
            self.session, backend,
            config.get("model") or config.get("model_id", "(default)"),
        )

    def chat(
        self,
        user_input: str,
        search_mode: Optional[str] = None,
        force_research: bool = False,
    ) -> AgentTextResponse | PaperSearchResult:
        """Handle one user turn and return conversational or ranked output.

        The method preserves backward compatibility with older call signatures
        while using the current automatic tool-decision behavior.
        """
        # Compatibility note: search_mode/force_research are accepted for older callers.
        del search_mode, force_research
        self._log_info(">>> user: %s", user_input)
        start = time.perf_counter()
        messages = self._build_messages(user_input)
        response = self._search_agent.invoke(messages)

        if not response.tool_calls:
            content = str(getattr(response, "content", "")).strip()
            return self._build_conversational_response(user_input, content, start)

        raw_results = self._run_search_tools(response.tool_calls)

        combined, title_url_map = _sanitize_search_results(raw_results, self.session)
        if not combined:
            return PaperSearchResult(
                papers=[],
                message=_NO_RESULTS_MESSAGE,
                status="no_results",
                reason="no_trusted_results",
                telemetry={"duration_sec": self._duration_sec(start)},
            )

        result = self._rank_results(user_input, combined, title_url_map)
        if result.telemetry is None:
            result.telemetry = {}
        result.telemetry["duration_sec"] = self._duration_sec(start)
        self._remember_turn(user_input, result.message or f"Returned {len(result.papers)} paper(s).")
        return result

    def chatAgent(
        self,
        user_input: str,
        search_mode: Optional[str] = None,
        force_research: bool = False,
    ) -> AgentTextResponse | PaperSearchResult:
        """Backward-compatible alias for older callers."""
        return self.chat(user_input, search_mode=search_mode, force_research=force_research)

    def _rank_results(self, user_input: str, combined: str, title_url_map: dict[str, str]) -> PaperSearchResult:
        """Rank validated PubMed blocks and normalize parser fallbacks."""
        block_count = len(combined.split(_BLOCK_SEPARATOR))
        self._log_info("ranking %d paper(s)", block_count)

        raw = self._ranker_agent.invoke([
            ("system", rankerSystemPrompt),
            ("human", f"User query: {user_input}\n\nPubMed results:\n{combined}"),
        ])
        ranked: PaperSearchResult | None = raw.get("parsed")

        if ranked is None:
            self._log_warning("structured parse failed; trying JSON fallback")
            ranked = _extract_result_from_raw(raw.get("raw"))

        if ranked is None:
            ranked = self._repair_ranker_output(raw.get("raw"), user_input)

        if ranked is None:
            return PaperSearchResult(
                papers=[],
                message=_RANKING_FAILED_MESSAGE,
                status="failed",
                reason="ranking_parse_failed",
            )

        self._normalize_ranked_papers(ranked, title_url_map)

        if not ranked.papers:
            ranked.status = "no_results"
            ranked.reason = ranked.reason or "ranker_filtered_all"
            ranked.message = ranked.message or _NO_RESULTS_MESSAGE
        else:
            ranked.status = "ranked"

        self._log_info("ranker returned %d paper(s)", len(ranked.papers))
        return ranked

    def _repair_ranker_output(self, raw_msg, user_input: str) -> Optional[PaperSearchResult]:
        """Attempt to repair malformed ranker text into schema-valid JSON."""
        content = getattr(raw_msg, "content", None)
        if not content or not isinstance(content, str):
            return None
        try:
            repaired = self._llm.invoke([
                ("system", rankerRepairSystemPrompt),
                ("human", f"User query: {user_input}\n\nMalformed ranker output:\n{content}"),
            ]).content
            value = _decode_first_json_value(str(repaired))
            if isinstance(value, dict):
                return PaperSearchResult(**value)
            if isinstance(value, list):
                return PaperSearchResult(papers=value)
        except Exception as exc:
            self._log_warning("ranker repair failed: %s", exc)
        return None

    def _build_conversational_response(self, user_input: str, content: str, start: float) -> AgentTextResponse:
        """Build standardized conversational response with telemetry."""
        self._remember_turn(user_input, content)
        self._log_info("<<< conversational: %s", content)
        return AgentTextResponse(
            message=content,
            status="conversational",
            telemetry={"duration_sec": self._duration_sec(start)},
        )

    def _run_search_tools(self, tool_calls: list[dict]) -> list[str]:
        """Execute supported tool calls and collect raw tool outputs."""
        raw_results: list[str] = []
        for call in tool_calls:
            if call["name"] != "searchPubMed":
                continue
            args = dict(call["args"])
            if not args.get("query", "").strip():
                continue
            if not isinstance(args.get("limit"), int):
                args["limit"] = _DEFAULT_TOOL_LIMIT
            raw_results.append(searchPubMed.invoke(args))
        return raw_results

    def record_feedback(
        self,
        *,
        paper_url: str,
        paper_title: str,
        query: str,
        relevant: bool,
        note: Optional[str] = None,
        search_mode: Optional[str] = None,
        confidence: Optional[float] = None,
        score_weight: float = 1.0,
    ) -> None:
        """Compatibility hook for future feedback-driven ranking adjustments."""
        # Compatibility note: feedback hooks are currently accepted but do not affect ranking.
        del paper_url, paper_title, query, relevant, note, search_mode, confidence, score_weight

    def _normalize_ranked_papers(self, ranked: PaperSearchResult, title_url_map: dict[str, str]) -> None:
        """Patch missing URLs/confidence/evidence fields in ranker output."""
        for paper in ranked.papers:
            if (not paper.url or paper.url == "N/A") and paper.title in title_url_map:
                paper.url = title_url_map[paper.title]
            paper.confidence = max(0.0, min(1.0, float(paper.confidence)))
            if not paper.evidence:
                paper.evidence = paper.relevance

    def _log_info(self, message: str, *args) -> None:
        """Log an INFO message with the active session prefix."""
        logger.info("[session=%s] " + message, self.session, *args)

    def _log_warning(self, message: str, *args) -> None:
        """Log a WARNING message with the active session prefix."""
        logger.warning("[session=%s] " + message, self.session, *args)

    def _duration_sec(self, start: float) -> float:
        """Return elapsed wall-clock seconds rounded for telemetry payloads."""
        return round(time.perf_counter() - start, 4)

    def _build_messages(self, user_input: str) -> list[tuple[str, str]]:
        """Build full prompt context from system prompt, summary, and recent turns."""
        messages: list[tuple[str, str]] = [("system", chatSystemPrompt)]
        if self._history_summary:
            messages.append(("system", "Conversation context (older turns): " + self._history_summary))
        messages.extend(self.conversation_history)
        messages.append(("human", user_input))
        return messages

    def _remember_turn(self, user_input: str, assistant_output: str) -> None:
        """Persist the latest turn and trigger history compaction when needed."""
        self.conversation_history.append(("human", str(user_input)))
        self.conversation_history.append(("assistant", str(assistant_output)))
        self._compact_history_if_needed()

    def _compact_history_if_needed(self) -> None:
        """Condense old turns into a bounded text summary for token control."""
        max_items = self._max_history_turns * 2
        if len(self.conversation_history) <= max_items:
            return
        overflow = self.conversation_history[:-max_items]
        snippets = []
        for i in range(0, len(overflow), 2):
            user_text = overflow[i][1] if i < len(overflow) else ""
            assistant_text = overflow[i + 1][1] if i + 1 < len(overflow) else ""
            snippets.append(f"Q: {user_text[:140].strip()} | A: {assistant_text[:180].strip()}")
        merged = " || ".join(s for s in snippets if s)
        if merged:
            combined = f"{self._history_summary} || {merged}" if self._history_summary else merged
            self._history_summary = combined[-_MAX_SUMMARY_CHARS:]
        self.conversation_history = self.conversation_history[-max_items:]
