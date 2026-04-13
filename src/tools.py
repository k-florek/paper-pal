import requests
import time
import xml.etree.ElementTree as ET
from langchain.tools import tool

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_FIELD_MAX_LEN = 300
_MAX_LIMIT = 25
_MIN_LIMIT = 1
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _request_json(url: str, params: dict, timeout: int = 10, attempts: int = 3) -> dict:
    """GET JSON with small exponential backoff for transient failures."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in _RETRYABLE_STATUS and attempt < attempts:
                time.sleep(0.4 * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(0.4 * (2 ** (attempt - 1)))
    raise RuntimeError(str(last_error))

def _sanitize_field(value: str, max_length: int = _FIELD_MAX_LEN) -> str:
    """Strip newlines and truncate a paper metadata field to prevent prompt injection."""
    if not isinstance(value, str):
        return "N/A"
    return value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")[:max_length].strip()


def _request_text(url: str, params: dict, timeout: int = 10, attempts: int = 3) -> str:
    """GET plain text with retry and exponential backoff."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in _RETRYABLE_STATUS and attempt < attempts:
                time.sleep(0.4 * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(0.4 * (2 ** (attempt - 1)))
    raise RuntimeError(str(last_error))


def _extract_efetch_metadata(pmids: list[str]) -> dict[str, dict[str, str]]:
    """Return abstract/MeSH/publication type fields keyed by PMID."""
    if not pmids:
        return {}

    xml_text = _request_text(
        EFETCH_URL,
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
    )

    root = ET.fromstring(xml_text)
    metadata: dict[str, dict[str, str]] = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_node = article.find(".//MedlineCitation/PMID")
        if pmid_node is None or not pmid_node.text:
            continue
        pmid = pmid_node.text.strip()

        abstract_parts = []
        for node in article.findall(".//MedlineCitation/Article/Abstract/AbstractText"):
            label = (node.attrib.get("Label") or "").strip()
            text = "".join(node.itertext()).strip()
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)

        mesh_terms = []
        for node in article.findall(".//MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName"):
            term = "".join(node.itertext()).strip()
            if term:
                mesh_terms.append(term)

        pub_types = []
        for node in article.findall(".//MedlineCitation/Article/PublicationTypeList/PublicationType"):
            value = "".join(node.itertext()).strip()
            if value:
                pub_types.append(value)

        metadata[pmid] = {
            "abstract": _sanitize_field("; ".join(abstract_parts) or "N/A", max_length=700),
            "mesh": _sanitize_field(", ".join(mesh_terms[:8]) or "N/A", max_length=300),
            "publication_type": _sanitize_field(", ".join(pub_types[:5]) or "N/A", max_length=220),
        }

    return metadata

@tool
def searchPubMed(query: str, limit: int = 10) -> str:
    """Search PubMed and return compact metadata blocks for ranking."""
    try:
        safe_limit = int(limit)
    except (TypeError, ValueError):
        safe_limit = 10
    safe_limit = max(_MIN_LIMIT, min(_MAX_LIMIT, safe_limit))

    try:
        search_payload = _request_json(
            ESEARCH_URL,
            params={"db": "pubmed", "term": query, "retmax": safe_limit, "retmode": "json"},
        )
        pmids = search_payload.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        return f"PubMed search failed: {e}"

    if not pmids:
        return "No results found."

    try:
        summary_payload = _request_json(
            ESUMMARY_URL,
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        )
        summaries = summary_payload.get("result", {})
    except Exception as e:
        return f"PubMed fetch failed: {e}"

    try:
        efetch_metadata = _extract_efetch_metadata(pmids)
    except Exception:
        efetch_metadata = {}

    results = []
    for pmid in pmids:
        paper = summaries.get(pmid)
        if not paper or not isinstance(paper, dict):
            continue

        title = _sanitize_field(paper.get("title", "N/A"))
        year = _sanitize_field(paper.get("pubdate", "N/A")[:4] if paper.get("pubdate") else "N/A")
        journal = _sanitize_field(paper.get("fulljournalname") or paper.get("source", "N/A"))
        authors = _sanitize_field(", ".join(a.get("name", "") for a in paper.get("authors", [])) or "N/A")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        extra = efetch_metadata.get(pmid, {})
        publication_type = extra.get("publication_type", "N/A")
        mesh = extra.get("mesh", "N/A")
        abstract = extra.get("abstract", "N/A")

        results.append(
            f"Title   : {title}\n"
            f"Authors : {authors}\n"
            f"Year    : {year}  |  Journal : {journal}\n"
            f"URL     : {url}\n"
            f"Type    : {publication_type}\n"
            f"MeSH    : {mesh}\n"
            f"Abstract: {abstract}"
        )

    return "\n\n---\n\n".join(results) if results else "No results found."
