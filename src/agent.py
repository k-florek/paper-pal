
import json
import logging
import re
import time
import uuid

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

from src.model import Backend, build_llm, modelType
from src.tools import searchPubMed
from src.prompts import (
    chatSystemPrompt,
    clarificationSystemPrompt,
    intentSystemPrompt,
    rankerSystemPrompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Validates a paper block produced by searchPubMed.
# Required core lines are title/authors/year+journal/url, followed by optional
# metadata lines (publication type, MeSH terms, abstract).
_PAPER_BLOCK_RE = re.compile(
    r"^Title\s+:\s+[^\n]+\n"
    r"Authors\s+:\s+[^\n]+\n"
    r"Year\s+:\s+[^\n]+\|\s+Journal\s+:\s+[^\n]+\n"
    r"URL\s+:\s+https://pubmed\.ncbi\.nlm\.nih\.gov/\d+/"
    r"(?:\n(?:Type|MeSH|Abstract)\s+:\s+[^\n]+)*$"
)

# Extracts Title and URL fields from a validated paper block
_FIELD_RE = re.compile(r"^(?P<key>Title|URL)\s+:\s+(?P<value>.+)$", re.MULTILINE)

# Detects a text-formatted tool call (used by models that skip structured tool_calls)
_TEXT_TOOL_CALL_RE = re.compile(
    r'searchPubMed\s*\(\s*query\s*=\s*["\'](?P<query>[^"\']+)["\']'
    r'(?:\s*,\s*limit\s*=\s*(?P<limit>\d+))?\s*\)',
    re.IGNORECASE,
)

# Finds a JSON object in LLM output — fenced block preferred over bare match
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(?P<json>\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(?P<json>\{.*\})", re.DOTALL)

# ---------------------------------------------------------------------------
# Tool-call utilities
# ---------------------------------------------------------------------------

def _parse_text_tool_calls(content: str, session: str) -> list[dict]:
    """Parse a searchPubMed call that a model emitted as plain text rather than
    a structured tool_calls entry (common with some Bedrock models)."""
    m = _TEXT_TOOL_CALL_RE.search(content)
    if not m:
        return []
    query = m.group("query").strip()
    limit = int(m.group("limit")) if m.group("limit") else 10
    logger.info(
        "[session=%s] text-formatted tool call detected; query=%r limit=%d",
        session, query, limit,
    )
    return [{"name": "searchPubMed", "args": {"query": query, "limit": limit}}]


def _normalize_tool_call_args(args: dict) -> dict:
    """Coerce tool call arguments to the expected types.

    Some models pass the JSON-schema descriptor object for ``limit`` instead of
    a plain integer.  This normalises it to a safe default when that happens.
    """
    args = dict(args)
    if "limit" in args and not isinstance(args["limit"], int):
        try:
            args["limit"] = int(args["limit"].get("value", 10))
        except (AttributeError, TypeError, ValueError):
            args["limit"] = 10
    return args

# ---------------------------------------------------------------------------
# Result sanitization
# ---------------------------------------------------------------------------

def _build_title_url_map(valid_blocks: list[str]) -> dict[str, str]:
    """Return a title → URL mapping extracted from pre-validated paper blocks."""
    mapping: dict[str, str] = {}
    for block in valid_blocks:
        fields = {m.group("key"): m.group("value").strip() for m in _FIELD_RE.finditer(block)}
        if "Title" in fields and "URL" in fields:
            mapping[fields["Title"]] = fields["URL"]
    return mapping


def _sanitize_search_results(
    raw_results: list[str], session: str
) -> tuple[str, dict[str, str]]:
    """Validate and filter raw search result strings before they reach the LLM.

    Each result string may contain multiple paper blocks separated by
    ``\\n\\n---\\n\\n``.  Only blocks that match the expected 4-line structure
    and a genuine PubMed URL are kept; malformed blocks are discarded to prevent
    prompt-injection attacks embedded in paper metadata.

    Returns:
        A tuple of (combined_text, title_url_map) where combined_text is the
        safe, re-joined blocks and title_url_map maps titles to their URLs.
    """
    valid: list[str] = []
    for batch in raw_results:
        for block in batch.split("\n\n---\n\n"):
            block = block.strip()
            if _PAPER_BLOCK_RE.fullmatch(block):
                valid.append(block)
            else:
                logger.warning(
                    "[session=%s] discarding malformed result block (possible prompt injection)",
                    session,
                )
    combined = "\n\n---\n\n".join(valid) if valid else ""
    return combined, _build_title_url_map(valid)

# ---------------------------------------------------------------------------
# Structured output utilities
# ---------------------------------------------------------------------------

def _extract_result_from_raw(raw_msg, session: str) -> Optional["PaperSearchResult"]:
    """Attempt to recover a PaperSearchResult from a raw AIMessage.

    Used as a fallback when ``with_structured_output`` fails to parse the
    model's response.  Searches for a JSON object in the message content —
    first inside a fenced code block, then as a bare object — and attempts
    to coerce it into a :class:`PaperSearchResult`.
    """
    content = getattr(raw_msg, "content", None)
    if not content or not isinstance(content, str):
        return None

    match = _FENCED_JSON_RE.search(content) or _BARE_JSON_RE.search(content)
    if not match:
        logger.warning("[session=%s] no JSON object found in raw ranker response", session)
        return None

    try:
        data = json.loads(match.group("json"))
        return PaperSearchResult(**data)
    except Exception as exc:
        logger.warning("[session=%s] fallback JSON parse failed: %s", session, exc)
        return None


def _extract_json_payload(content: str) -> Optional[dict]:
    """Extract a JSON object from content that may include fences or prose."""
    if not content:
        return None
    match = _FENCED_JSON_RE.search(content) or _BARE_JSON_RE.search(content)
    if not match:
        return None
    try:
        payload = json.loads(match.group("json"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Paper(BaseModel):
    """A single PubMed paper as returned by the ranker."""

    index: int = Field(description="1-based rank position (1 = most relevant)")
    title: str = Field(description="Full paper title")
    authors: Optional[str] = Field(default="N/A", description="Comma-separated author names")
    year: Optional[str] = Field(default="N/A", description="Publication year")
    journal: Optional[str] = Field(default="N/A", description="Journal or venue name")
    url: Optional[str] = Field(default="N/A", description="PubMed URL")
    relevance: str = Field(default="", description="One sentence explaining relevance to the query")


class PaperSearchResult(BaseModel):
    """Ranked list of papers returned by the ranker agent.

    ``papers`` is non-empty when relevant papers are found.
    ``message`` is set when no papers could be returned.
    """

    papers: list[Paper] = Field(default_factory=list)
    message: Optional[str] = Field(default=None)
    status: Literal["ranked", "no_results", "failed"] = "ranked"
    reason: Optional[str] = Field(default=None)

    @field_validator("papers", mode="before")
    @classmethod
    def _coerce_papers_string(cls, v):
        """Handle models that return ``papers`` as a JSON string instead of a list."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return []
        return v


class IntentDecision(BaseModel):
    """Routing decision used before triggering search."""

    intent: Literal["RESEARCH", "CONVERSATIONAL"] = "CONVERSATIONAL"
    confidence: float = 0.5
    ambiguous: bool = False
    missing_constraints: list[str] = Field(default_factory=list)
    needs_clarification: bool = False

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value):
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, conf))


class AgentTextResponse(BaseModel):
    """Non-paper response payload with explicit state."""

    message: str
    status: Literal["conversational", "clarifying", "failed"] = "conversational"
    reason: Optional[str] = None

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_NO_RESULTS = PaperSearchResult(
    papers=[],
    message="No relevant papers found for your query.",
    status="no_results",
    reason="no_relevant_papers",
)
_DEFAULT_MAX_HISTORY_TURNS = 10
_MAX_SUMMARY_CHARS = 1800


class Agent:
    """Conversational research assistant backed by PubMed search.

    Each turn runs up to three LLM calls:

    1. **Intent classification** — decides if the message warrants a search.
    2. **Search dispatch** — selects a PubMed query and calls ``searchPubMed``.
    3. **Ranking** — filters and ranks raw results into a structured response.

    Conversational turns (greetings, etc.) skip steps 2 and 3.
    """

    def __init__(
        self,
        session: str = None,
        backend: Backend = "ollama",
        backend_config: dict = None,
    ):
        if not session:
            session = str(uuid.uuid4())
        self.session = session
        self.conversation_history: list[tuple[str, str]] = []
        self._history_summary: str = ""
        self._max_history_turns: int = _DEFAULT_MAX_HISTORY_TURNS

        config = backend_config or {}
        try:
            self._max_history_turns = max(2, int(config.get("max_history_turns", _DEFAULT_MAX_HISTORY_TURNS)))
        except (TypeError, ValueError):
            self._max_history_turns = _DEFAULT_MAX_HISTORY_TURNS
        chat_llm = build_llm(backend, config, modelType.CHAT)
        reasoning_llm = build_llm(backend, config, modelType.REASONING)

        self._llm = chat_llm
        self._search_agent = chat_llm.bind_tools([searchPubMed])
        # include_raw=True keeps the raw AIMessage so we can fall back to manual
        # JSON extraction when the structured-output parser fails.
        self._ranker_agent = reasoning_llm.with_structured_output(
            PaperSearchResult, include_raw=True
        )

        logger.info(
            "[session=%s] Agent initialised | backend=%s model=%s temp=%.2f top_p=%.2f",
            session,
            backend,
            config.get("model") or config.get("model_id", "(default)"),
            config.get("temperature", 0.4),
            config.get("top_p", 0.9),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chatAgent(self, user_input: str) -> AgentTextResponse | PaperSearchResult:
        """Process *user_input* and return either a plain reply or ranked papers.

        Returns:
            A ``str`` for conversational messages, or a :class:`PaperSearchResult`
            for research queries (``papers`` may be empty if nothing was found).
        """
        logger.info("[session=%s] >>> user: %s", self.session, user_input)

        start = time.perf_counter()
        messages = self._build_context_messages(chatSystemPrompt, user_input)

        decision = self._classify_intent(user_input)
        logger.info(
            "[session=%s] route | intent=%s confidence=%.2f ambiguous=%s needs_clarification=%s",
            self.session,
            decision.intent,
            decision.confidence,
            decision.ambiguous,
            decision.needs_clarification,
        )

        if decision.intent == "CONVERSATIONAL" and not decision.needs_clarification:
            reply = self._conversational_reply(messages, user_input)
            logger.info("[session=%s] turn completed in %.3fs", self.session, time.perf_counter() - start)
            return AgentTextResponse(message=reply, status="conversational")

        if decision.needs_clarification:
            question = self._clarify_query(user_input, decision)
            self._remember_turn(user_input, question)
            logger.info("[session=%s] turn completed in %.3fs", self.session, time.perf_counter() - start)
            return AgentTextResponse(
                message=question,
                status="clarifying",
                reason="missing_search_constraints",
            )

        result = self._research_pipeline(messages, user_input)
        if isinstance(result, PaperSearchResult):
            summary = result.message or f"Returned {len(result.papers)} paper(s)."
            self._remember_turn(user_input, summary)
        logger.info("[session=%s] turn completed in %.3fs", self.session, time.perf_counter() - start)
        return result

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _classify_intent(self, user_input: str) -> IntentDecision:
        """Return a structured routing decision for the current message."""
        messages = self._build_context_messages(intentSystemPrompt, user_input)
        result = self._llm.invoke(messages)
        payload = _extract_json_payload(getattr(result, "content", ""))
        if payload is not None:
            try:
                return IntentDecision(**payload)
            except Exception as exc:
                logger.warning("[session=%s] intent parse failed: %s", self.session, exc)

        # Fallback for models that ignore JSON instruction.
        classification = str(getattr(result, "content", "")).strip().upper()
        is_research = classification.startswith("RESEARCH")
        return IntentDecision(
            intent="RESEARCH" if is_research else "CONVERSATIONAL",
            confidence=0.6,
            ambiguous=is_research,
            missing_constraints=[],
            needs_clarification=False,
        )

    def _clarify_query(self, user_input: str, decision: IntentDecision) -> str:
        """Ask one focused question to reduce retrieval ambiguity."""
        clarifier_input = {
            "user_query": user_input,
            "missing_constraints": decision.missing_constraints,
        }
        content = self._llm.invoke(
            [
                ("system", clarificationSystemPrompt),
                ("human", json.dumps(clarifier_input)),
            ]
        ).content
        text = str(content).strip()
        return text or "Could you narrow this down by population, study type, or timeframe?"

    def _research_pipeline(self, messages: list, user_input: str) -> AgentTextResponse | PaperSearchResult:
        """Run the search → sanitize → rank pipeline for a research query."""
        response = self._search_agent.invoke(messages)
        logger.info(
            "[session=%s] search agent | content=%r | tool_calls=%s",
            self.session, response.content, response.tool_calls,
        )

        tool_calls = response.tool_calls or _parse_text_tool_calls(response.content, self.session)

        if not tool_calls:
            # Model chose not to call a tool — treat as conversational
            return AgentTextResponse(
                message=self._conversational_reply(messages, user_input, prefetched=response.content),
                status="conversational",
                reason="no_tool_call",
            )

        raw_results = self._execute_tool_calls(tool_calls)

        if not raw_results:
            # All tool calls were skipped or returned nothing — fall back to plain reply
            return AgentTextResponse(
                message="I could not complete the PubMed search. Please try again or refine your query.",
                status="failed",
                reason="search_failed",
            )

        combined, title_url_map = _sanitize_search_results(raw_results, self.session)
        if not combined:
            logger.warning("[session=%s] all result blocks discarded; returning empty result", self.session)
            return PaperSearchResult(
                papers=[],
                message="No trusted results could be extracted from the PubMed response.",
                status="failed",
                reason="sanitization_rejected_all",
            )

        return self._rank_results(user_input, combined, title_url_map)

    def _execute_tool_calls(self, tool_calls: list[dict]) -> list[str]:
        """Run each searchPubMed tool call and return the raw result strings."""
        results: list[str] = []
        for call in tool_calls:
            if call["name"] != "searchPubMed":
                continue
            args = _normalize_tool_call_args(call["args"])
            if not args.get("query", "").strip():
                logger.info("[session=%s] skipping searchPubMed — empty query", self.session)
                continue
            logger.info("[session=%s] searchPubMed(%s)", self.session, args)
            result = searchPubMed.invoke(args)
            logger.info("[session=%s] searchPubMed returned %d chars", self.session, len(result))
            logger.info(result)
            results.append(result)
        return results

    def _rank_results(
        self, user_input: str, combined: str, title_url_map: dict[str, str]
    ) -> PaperSearchResult:
        """Invoke the ranker and return a validated PaperSearchResult.

        Tries the structured-output chain first; if the parser returns ``None``
        the raw AIMessage content is searched for a JSON object as a fallback.
        """
        block_count = len(combined.split("\n\n---\n\n"))
        logger.info("[session=%s] handing %d paper(s) to ranker", self.session, block_count)

        raw = self._ranker_agent.invoke([
            ("system", rankerSystemPrompt),
            ("human", f"User query: {user_input}\n\nPubMed results:\n{combined}"),
        ])

        ranked: PaperSearchResult | None = raw.get("parsed")

        if ranked is None:
            logger.warning(
                "[session=%s] structured parse failed (%s); trying JSON fallback",
                self.session, raw.get("parsing_error"),
            )
            ranked = _extract_result_from_raw(raw.get("raw"), self.session)

        if ranked is None:
            logger.warning("[session=%s] JSON fallback failed; returning empty result", self.session)
            return PaperSearchResult(
                papers=[],
                message="I couldn't reliably rank the search results. Please try again.",
                status="failed",
                reason="ranking_parse_failed",
            )

        # Patch URLs the LLM failed to reproduce in its structured output
        for paper in ranked.papers:
            if (not paper.url or paper.url == "N/A") and paper.title in title_url_map:
                paper.url = title_url_map[paper.title]

        logger.info(
            "[session=%s] ranker returned %d paper(s) | message=%r",
            self.session, len(ranked.papers), ranked.message,
        )
        if not ranked.papers:
            ranked.status = "no_results"
            ranked.reason = ranked.reason or "ranker_filtered_all"
            if not ranked.message:
                ranked.message = "No relevant papers found for your query."
        else:
            ranked.status = "ranked"
        return ranked

    def _conversational_reply(
        self, messages: list, user_input: str, *, prefetched: str = None
    ) -> str:
        """Return a plain LLM reply and update conversation history.

        Pass *prefetched* when the model content is already available (e.g. the
        search agent returned a response without calling any tools) to avoid a
        redundant LLM call.
        """
        if prefetched is not None:
            content = prefetched
        else:
            content = self._llm.invoke(messages).content

        self._remember_turn(user_input, content)
        logger.info("[session=%s] <<< assistant (conversational): %s", self.session, content)
        return content

    def _remember_turn(self, user_input: str, assistant_output: str) -> None:
        """Persist conversation history for follow-up turns."""
        self.conversation_history.append(("human", str(user_input)))
        self.conversation_history.append(("assistant", str(assistant_output)))
        self._compact_history_if_needed()

    def _build_context_messages(self, system_prompt: str, user_input: str) -> list[tuple[str, str]]:
        """Build model input with summarized older context + recent turn window."""
        messages: list[tuple[str, str]] = [("system", system_prompt)]
        if self._history_summary:
            messages.append((
                "system",
                "Conversation context summary (older turns): " + self._history_summary,
            ))
        messages.extend(self.conversation_history)
        messages.append(("human", user_input))
        return messages

    def _compact_history_if_needed(self) -> None:
        """Keep only recent turns in-memory and summarize older context."""
        max_items = self._max_history_turns * 2
        if len(self.conversation_history) <= max_items:
            return

        overflow = self.conversation_history[:-max_items]
        snippets: list[str] = []
        for i in range(0, len(overflow), 2):
            user_text = overflow[i][1] if i < len(overflow) else ""
            assistant_text = overflow[i + 1][1] if i + 1 < len(overflow) else ""
            snippet = (
                f"Q: {user_text[:140].strip()} | "
                f"A: {assistant_text[:180].strip()}"
            )
            snippets.append(snippet)

        merged = " || ".join(s for s in snippets if s)
        if merged:
            if self._history_summary:
                self._history_summary = f"{self._history_summary} || {merged}"
            else:
                self._history_summary = merged
            if len(self._history_summary) > _MAX_SUMMARY_CHARS:
                self._history_summary = self._history_summary[-_MAX_SUMMARY_CHARS:]

        self.conversation_history = self.conversation_history[-max_items:]