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
    if not result.papers:
        msg = result.message or "No relevant papers found."
        return f"{YELLOW}{msg}{RESET}"
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


ai = Agent(backend="ollama")

r = ai.chatAgent('SARS-CoV-2 Evolution')

if isinstance(r, PaperSearchResult):
    print(format_papers(r))
elif isinstance(r, AgentTextResponse):
    print(r.message)
else:
    print(r)