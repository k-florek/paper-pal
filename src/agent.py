
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

SearchMode = Literal["balanced", "clinical", "mechanism", "latest", "reviews"]

_SEARCH_MODE_GUIDANCE: dict[SearchMode, str] = {
    "balanced": "Search mode is BALANCED: optimize for overall relevance and coverage.",
    "clinical": (
        "Search mode is CLINICAL: prioritize human studies, trials, treatment outcomes, and evidence quality. "
        "Prefer clinical terminology and intervention/comparator/outcome framing where possible."
    ),
    "mechanism": (
        "Search mode is MECHANISM: prioritize molecular, cellular, pathway, and mechanistic evidence. "
        "Bias toward studies explaining biological causality and mechanisms."
    ),
    "latest": (
        "Search mode is LATEST: prioritize recency and rapidly evolving evidence while maintaining topical relevance."
    ),
    "reviews": (
        "Search mode is REVIEWS: prioritize systematic reviews, meta-analyses, and high-level synthesis papers first."
    ),
}

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
_TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "what", "with",
}
_GENERIC_BROAD_TERMS = {
    "cancer", "therapy", "treatment", "disease", "infection", "biology", "mechanism", "immune", "genetics",
}
_TERM_EXPANSIONS: dict[str, list[str]] = {
    "covid": ["COVID-19", "SARS-CoV-2"],
    "sars-cov-2": ["COVID-19", "SARS-CoV-2"],
    "crispr": ["CRISPR", "gene editing"],
    "ngs": ["NGS", "next-generation sequencing", "high-throughput sequencing"],
    "ai": ["artificial intelligence", "machine learning"],
    "ml": ["machine learning", "deep learning"],
}
_RESEARCH_HINT_TOKENS = {
    "paper", "papers", "study", "studies", "trial", "trials", "meta-analysis", "mechanism",
    "treatment", "therapy", "evidence", "review", "pubmed", "disease", "outcome", "model",
}

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
        stripped_batch = batch.strip()
        if (
            not stripped_batch
            or stripped_batch == "No results found."
            or stripped_batch.startswith("PubMed search failed:")
            or stripped_batch.startswith("PubMed fetch failed:")
        ):
            logger.info("[session=%s] skipping non-paper tool output: %r", session, stripped_batch[:120])
            continue

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


def _tokenize(text: str) -> set[str]:
    """Tokenize text into normalized keyword tokens."""
    if not text:
        return set()
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


def _parse_paper_block(block: str) -> dict[str, str]:
    """Parse key/value lines from a validated paper block."""
    parsed: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _score_block(query_tokens: set[str], parsed: dict[str, str]) -> float:
    """Score a paper block deterministically before LLM reranking."""
    title_tokens = _tokenize(parsed.get("Title", ""))
    mesh_tokens = _tokenize(parsed.get("MeSH", ""))
    abstract_tokens = _tokenize(parsed.get("Abstract", ""))

    title_overlap = len(query_tokens & title_tokens)
    mesh_overlap = len(query_tokens & mesh_tokens)
    abstract_overlap = len(query_tokens & abstract_tokens)

    score = (title_overlap * 3.0) + (mesh_overlap * 2.0) + (abstract_overlap * 1.0)

    publication_type = parsed.get("Type", "").lower()
    if "meta-analysis" in publication_type or "systematic review" in publication_type:
        score += 1.0
    if "randomized" in publication_type or "clinical trial" in publication_type:
        score += 0.8

    year_value = parsed.get("Year", "")
    try:
        year = int(year_value[:4])
        # Mild recency preference, without suppressing classic foundational work.
        score += max(0.0, min(1.0, (year - 2015) * 0.08))
    except (TypeError, ValueError):
        pass

    return score


def _pre_rank_blocks(
    user_input: str,
    combined: str,
    title_url_map: dict[str, str],
    *,
    top_n: int,
    token_preferences: dict[str, float],
    url_preferences: dict[str, float],
    session: str,
) -> tuple[str, dict[str, str]]:
    """Deterministically pre-rank blocks and keep top candidates for LLM ranking."""
    blocks = [block for block in combined.split("\n\n---\n\n") if block.strip()]
    if len(blocks) <= top_n:
        return combined, title_url_map

    query_tokens = _tokenize(user_input)
    scored: list[tuple[float, str, str]] = []
    for block in blocks:
        parsed = _parse_paper_block(block)
        title = parsed.get("Title", "")
        score = _score_block(query_tokens, parsed)

        url = parsed.get("URL", "")
        if url in url_preferences:
            score += url_preferences[url]

        feedback_tokens = _tokenize(
            " ".join([parsed.get("Title", ""), parsed.get("MeSH", ""), parsed.get("Abstract", "")])
        )
        for token in feedback_tokens:
            score += token_preferences.get(token, 0.0)

        scored.append((score, block, title))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept = scored[:top_n]
    kept_blocks = [item[1] for item in kept]
    kept_titles = {item[2] for item in kept if item[2]}
    kept_map = {title: url for title, url in title_url_map.items() if title in kept_titles}

    logger.info(
        "[session=%s] pre-ranker kept %d/%d papers (top_n=%d)",
        session,
        len(kept_blocks),
        len(blocks),
        top_n,
    )

    return "\n\n---\n\n".join(kept_blocks), kept_map


def _count_result_blocks(result: str) -> int:
    """Count formatted paper blocks in a tool response string."""
    if not result or result.startswith("PubMed ") or result == "No results found.":
        return 0
    return len([part for part in result.split("\n\n---\n\n") if part.strip()])


def _adaptive_limit(query: str, mode: SearchMode, requested: int) -> int:
    """Estimate retrieval breadth and choose a bounded search limit."""
    query_tokens = _tokenize(query)
    broad_hits = len(query_tokens & _GENERIC_BROAD_TERMS)
    has_boolean = " AND " in query.upper() or " OR " in query.upper()
    is_broad = broad_hits >= 2 or len(query_tokens) < 4 or not has_boolean

    if mode == "latest":
        target = 25 if is_broad else 20
    elif mode == "reviews":
        target = 18 if is_broad else 14
    elif mode == "clinical":
        target = 20 if is_broad else 15
    else:
        target = 22 if is_broad else 12

    return max(5, min(25, max(requested, target)))


def _rewrite_pubmed_query(raw_query: str, mode: SearchMode) -> str:
    """Rewrite plain-language queries into a more retrieval-friendly PubMed query."""
    query = (raw_query or "").strip()
    if not query:
        return query

    upper = query.upper()
    if " AND " in upper or " OR " in upper:
        rewritten = query
    else:
        tokens = list(_tokenize(query))[:6]
        concepts: list[str] = []
        for token in tokens:
            expansions = _TERM_EXPANSIONS.get(token, [])
            if expansions:
                opts = " OR ".join([f'"{token}"'] + [f'"{term}"' for term in expansions])
                concepts.append(f"({opts})")
            else:
                concepts.append(f'"{token}"')
        rewritten = " AND ".join(concepts[:4]) if concepts else query

    if mode == "clinical":
        rewritten = f"({rewritten}) AND (clinical OR trial OR patient)"
    elif mode == "mechanism":
        rewritten = f"({rewritten}) AND (mechanism OR pathway OR molecular OR cellular)"
    elif mode == "latest":
        rewritten = f"({rewritten}) AND (2020:3000[pdat])"
    elif mode == "reviews":
        rewritten = f"({rewritten}) AND (\"systematic review\" OR \"meta-analysis\")"

    return rewritten


def _broaden_query(query: str) -> str:
    """Generate one broader fallback query when initial search is too sparse."""
    q = query.replace(" AND ", " OR ")
    # Keep fallback bounded and simple.
    return q


def _precision_query(query: str, mode: SearchMode) -> str:
    """Generate one narrower fallback query when results are too broad."""
    if mode == "reviews":
        return f"({query}) AND (\"systematic review\" OR \"meta-analysis\")"
    if mode == "latest":
        return f"({query}) AND (2022:3000[pdat])"
    if mode == "clinical":
        return f"({query}) AND (randomized OR placebo OR patient)"
    if mode == "mechanism":
        return f"({query}) AND (pathway OR signaling OR mechanism)"
    return f"({query}) AND (human OR mechanism)"


def _title_signature(title: str) -> str:
    """Create a compact signature for near-duplicate title suppression."""
    toks = sorted(_tokenize(title))
    return " ".join(toks[:6])


def _apply_diversity_controls(papers: list["Paper"]) -> list["Paper"]:
    """Reduce near-duplicates and over-concentration from a single journal."""
    seen_signatures: set[str] = set()
    per_journal: dict[str, int] = {}
    filtered: list["Paper"] = []

    for paper in papers:
        signature = _title_signature(paper.title or "")
        if signature and signature in seen_signatures:
            continue

        journal_key = (paper.journal or "unknown").strip().lower()
        count = per_journal.get(journal_key, 0)
        if count >= 2:
            continue

        if signature:
            seen_signatures.add(signature)
        per_journal[journal_key] = count + 1
        filtered.append(paper)

    for idx, paper in enumerate(filtered, start=1):
        paper.index = idx
    return filtered


def _looks_research_query(user_input: str) -> bool:
    """Heuristic guardrail for cases where intent LLM under-classifies research asks."""
    tokens = _tokenize(user_input)
    if not tokens:
        return False
    if len(tokens & _RESEARCH_HINT_TOKENS) >= 1:
        return True
    # Longer domain-like prompts are likely research asks even without explicit keywords.
    return len(tokens) >= 5

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
    confidence: float = Field(default=0.5, description="Model confidence from 0.0 to 1.0")
    evidence: Optional[str] = Field(default="", description="Short evidence snippet supporting relevance")

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value):
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, conf))


class PaperSearchResult(BaseModel):
    """Ranked list of papers returned by the ranker agent.

    ``papers`` is non-empty when relevant papers are found.
    ``message`` is set when no papers could be returned.
    """

    papers: list[Paper] = Field(default_factory=list)
    message: Optional[str] = Field(default=None)
    status: Literal["ranked", "no_results", "failed"] = "ranked"
    reason: Optional[str] = Field(default=None)
    telemetry: Optional[dict] = Field(default=None)

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
    telemetry: Optional[dict] = None

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
        self._pre_rank_top_n: int = 20
        self._search_mode: SearchMode = "balanced"
        self._token_preferences: dict[str, float] = {}
        self._url_preferences: dict[str, float] = {}
        self._feedback_events: list[dict[str, str | bool]] = []

        config = backend_config or {}
        try:
            self._max_history_turns = max(2, int(config.get("max_history_turns", _DEFAULT_MAX_HISTORY_TURNS)))
        except (TypeError, ValueError):
            self._max_history_turns = _DEFAULT_MAX_HISTORY_TURNS
        try:
            self._pre_rank_top_n = max(5, int(config.get("pre_rank_top_n", 20)))
        except (TypeError, ValueError):
            self._pre_rank_top_n = 20
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

    def chatAgent(
        self,
        user_input: str,
        search_mode: Optional[SearchMode] = None,
        force_research: bool = False,
    ) -> AgentTextResponse | PaperSearchResult:
        """Process *user_input* and return either a plain reply or ranked papers.

        Returns:
            A ``str`` for conversational messages, or a :class:`PaperSearchResult`
            for research queries (``papers`` may be empty if nothing was found).
        """
        logger.info("[session=%s] >>> user: %s", self.session, user_input)

        start = time.perf_counter()
        if search_mode in _SEARCH_MODE_GUIDANCE:
            self._search_mode = search_mode
        messages = self._build_context_messages(chatSystemPrompt, user_input)

        decision = IntentDecision(intent="RESEARCH", confidence=1.0) if force_research else self._classify_intent(user_input)
        logger.info(
            "[session=%s] route | intent=%s confidence=%.2f ambiguous=%s needs_clarification=%s",
            self.session,
            decision.intent,
            decision.confidence,
            decision.ambiguous,
            decision.needs_clarification,
        )

        if decision.intent == "CONVERSATIONAL" and _looks_research_query(user_input):
            logger.info("[session=%s] overriding CONVERSATIONAL -> RESEARCH via heuristic", self.session)
            decision.intent = "RESEARCH"
            decision.needs_clarification = False
            decision.ambiguous = False

        if decision.intent == "CONVERSATIONAL" and not decision.needs_clarification:
            reply = self._conversational_reply(messages, user_input)
            duration = time.perf_counter() - start
            logger.info("[session=%s] turn completed in %.3fs", self.session, time.perf_counter() - start)
            return AgentTextResponse(
                message=reply,
                status="conversational",
                telemetry={
                    "search_mode": self._search_mode,
                    "intent": decision.intent,
                    "intent_confidence": decision.confidence,
                    "duration_sec": round(duration, 4),
                },
            )

        if decision.needs_clarification:
            question = self._clarify_query(user_input, decision)
            self._remember_turn(user_input, question)
            duration = time.perf_counter() - start
            logger.info("[session=%s] turn completed in %.3fs", self.session, time.perf_counter() - start)
            return AgentTextResponse(
                message=question,
                status="clarifying",
                reason="missing_search_constraints",
                telemetry={
                    "search_mode": self._search_mode,
                    "intent": decision.intent,
                    "intent_confidence": decision.confidence,
                    "needs_clarification": True,
                    "duration_sec": round(duration, 4),
                },
            )

        result = self._research_pipeline(messages, user_input)
        if isinstance(result, PaperSearchResult):
            summary = result.message or f"Returned {len(result.papers)} paper(s)."
            self._remember_turn(user_input, summary)
            if result.telemetry is None:
                result.telemetry = {}
            result.telemetry.update(
                {
                    "search_mode": self._search_mode,
                    "intent": decision.intent,
                    "intent_confidence": decision.confidence,
                    "duration_sec": round(time.perf_counter() - start, 4),
                }
            )
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
        pipeline_start = time.perf_counter()
        response = self._search_agent.invoke(messages)
        logger.info(
            "[session=%s] search agent | content=%r | tool_calls=%s",
            self.session, response.content, response.tool_calls,
        )

        tool_calls = response.tool_calls or _parse_text_tool_calls(response.content, self.session)

        if not tool_calls:
            # Bedrock/tool-call fallback: run a direct retrieval pass from user input.
            logger.info("[session=%s] no tool call detected; using direct-search fallback", self.session)
            tool_calls = [{"name": "searchPubMed", "args": {"query": user_input, "limit": 12}}]

        raw_results, search_attempts = self._execute_tool_calls(tool_calls)

        if not raw_results:
            # All tool calls were skipped or returned nothing — fall back to plain reply
            return AgentTextResponse(
                message="I could not complete the PubMed search. Please try again or refine your query.",
                status="failed",
                reason="search_failed",
                telemetry={
                    "pipeline": "research",
                    "tool_calls_detected": len(tool_calls),
                    "search_attempts": search_attempts,
                    "duration_sec": round(time.perf_counter() - pipeline_start, 4),
                },
            )

        combined, title_url_map = _sanitize_search_results(raw_results, self.session)
        if not combined:
            logger.warning("[session=%s] all result blocks discarded; returning empty result", self.session)
            return PaperSearchResult(
                papers=[],
                message="No trusted results could be extracted from the PubMed response.",
                status="failed",
                reason="sanitization_rejected_all",
                telemetry={
                    "pipeline": "research",
                    "tool_calls_detected": len(tool_calls),
                    "search_attempts": search_attempts,
                    "valid_blocks": 0,
                    "duration_sec": round(time.perf_counter() - pipeline_start, 4),
                },
            )

        combined, title_url_map = _pre_rank_blocks(
            user_input,
            combined,
            title_url_map,
            top_n=self._pre_rank_top_n,
            token_preferences=self._token_preferences,
            url_preferences=self._url_preferences,
            session=self.session,
        )

        result = self._rank_results(user_input, combined, title_url_map)
        if result.telemetry is None:
            result.telemetry = {}
        result.telemetry.update(
            {
                "pipeline": "research",
                "tool_calls_detected": len(tool_calls),
                "search_attempts": search_attempts,
                "valid_blocks": len([part for part in combined.split("\n\n---\n\n") if part.strip()]),
                "duration_sec": round(time.perf_counter() - pipeline_start, 4),
            }
        )
        return result

    def _execute_tool_calls(self, tool_calls: list[dict]) -> tuple[list[str], list[dict]]:
        """Run each searchPubMed tool call with bounded multi-pass retrieval."""
        results: list[str] = []
        attempts: list[dict] = []
        for call in tool_calls:
            if call["name"] != "searchPubMed":
                continue
            args = _normalize_tool_call_args(call["args"])
            if not args.get("query", "").strip():
                logger.info("[session=%s] skipping searchPubMed — empty query", self.session)
                continue

            raw_query = args.get("query", "")
            requested_limit = int(args.get("limit", 10)) if isinstance(args.get("limit"), int) else 10
            rewritten = _rewrite_pubmed_query(raw_query, self._search_mode)
            adaptive_limit = _adaptive_limit(rewritten, self._search_mode, requested_limit)

            logger.info(
                "[session=%s] search primary | mode=%s limit=%d query=%r",
                self.session,
                self._search_mode,
                adaptive_limit,
                rewritten,
            )
            primary = searchPubMed.invoke({"query": rewritten, "limit": adaptive_limit})
            primary_count = _count_result_blocks(primary)
            results.append(primary)
            attempts.append(
                {
                    "stage": "primary",
                    "query": rewritten,
                    "limit": adaptive_limit,
                    "result_blocks": primary_count,
                }
            )

            # Fallback 1: broaden once if sparse/empty.
            if primary_count < 3:
                broadened = _broaden_query(rewritten)
                logger.info("[session=%s] search broaden fallback query=%r", self.session, broadened)
                fallback = searchPubMed.invoke({"query": broadened, "limit": adaptive_limit})
                results.append(fallback)
                attempts.append(
                    {
                        "stage": "broaden",
                        "query": broadened,
                        "limit": adaptive_limit,
                        "result_blocks": _count_result_blocks(fallback),
                    }
                )

            # Fallback 2: precision once if result set appears broad/full.
            if primary_count >= max(8, adaptive_limit - 2):
                precise = _precision_query(rewritten, self._search_mode)
                precise_limit = max(8, adaptive_limit // 2)
                logger.info("[session=%s] search precision fallback query=%r", self.session, precise)
                fallback = searchPubMed.invoke({"query": precise, "limit": precise_limit})
                results.append(fallback)
                attempts.append(
                    {
                        "stage": "precision",
                        "query": precise,
                        "limit": precise_limit,
                        "result_blocks": _count_result_blocks(fallback),
                    }
                )

        return results, attempts

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
            paper.confidence = max(0.0, min(1.0, float(paper.confidence)))
            if not paper.evidence:
                paper.evidence = paper.relevance

        ranked.papers = _apply_diversity_controls(ranked.papers)

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

    def record_feedback(
        self,
        *,
        paper_url: str,
        paper_title: str,
        query: str,
        relevant: bool,
        note: Optional[str] = None,
    ) -> None:
        """Store session feedback and update lightweight preference weights."""
        delta = 1.5 if relevant else -1.5
        if paper_url:
            self._url_preferences[paper_url] = self._url_preferences.get(paper_url, 0.0) + delta

        source_text = " ".join([paper_title or "", query or "", note or ""])
        for token in _tokenize(source_text):
            self._token_preferences[token] = self._token_preferences.get(token, 0.0) + (0.2 if relevant else -0.2)

        self._feedback_events.append(
            {
                "paper_url": paper_url,
                "paper_title": paper_title,
                "query": query,
                "relevant": relevant,
                "note": note or "",
            }
        )
        # Keep recent events only to avoid unbounded in-memory growth.
        if len(self._feedback_events) > 200:
            self._feedback_events = self._feedback_events[-200:]

    def _build_context_messages(self, system_prompt: str, user_input: str) -> list[tuple[str, str]]:
        """Build model input with summarized older context + recent turn window."""
        messages: list[tuple[str, str]] = [("system", system_prompt)]
        messages.append(("system", _SEARCH_MODE_GUIDANCE[self._search_mode]))
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