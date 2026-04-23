"""Minimal CLI entry point for running a single Paper Pal query.

Useful for headless testing and backend validation without starting FastAPI.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.agent import Agent, AgentTextResponse, PaperSearchResult

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
DIM    = "\033[2m"


def format_papers(result: PaperSearchResult) -> str:
    """Render ranked papers in a readable terminal format."""
    if not result.papers:
        return f"{YELLOW}{result.message or 'No relevant papers found.'}{RESET}"

    seen_urls = set()
    lines = []
    display_index = 1
    for paper in result.papers:
        if paper.url in seen_urls:
            continue
        seen_urls.add(paper.url)
        lines.append(
            f"{BOLD}{CYAN}[{display_index}] {paper.title}{RESET}\n"
            f"    {GREEN}{paper.url}{RESET}\n"
            f"    {DIM}{paper.relevance}{RESET}"
        )
        display_index += 1

    return "\n\n".join(lines)


def print_response(response: PaperSearchResult | AgentTextResponse | object) -> None:
    """Print any supported agent response shape to stdout."""
    if isinstance(response, PaperSearchResult):
        print(format_papers(response))
        return
    if isinstance(response, AgentTextResponse):
        print(response.message)
        return
    print(response)


if __name__ == "__main__":
    # Default probe query for quick local smoke-testing.
    agent = Agent(backend="ollama")
    response = agent.chat("SARS-CoV-2 Evolution")
    print_response(response)