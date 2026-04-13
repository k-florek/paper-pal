# Paper Pal

An AI-powered academic research assistant that searches PubMed for scientific papers based on natural language queries. Paper Pal uses a multi-step LLM pipeline to understand your research question, search the literature, and return a ranked, relevance-filtered list of papers.

## Features

- **Natural language queries** — ask a research question in plain English
- **PubMed integration** — live search via the NCBI Entrez API
- **LLM-powered ranking** — a second LLM pass filters and ranks results by relevance to your query
- **Conversational** — maintains session history; handles greetings and non-research messages without triggering a search
- **Prompt-injection protection** — paper metadata is validated against a strict regex before being passed to the LLM
- **Pluggable backends** — run against a local Ollama server, AWS Bedrock, or any OpenAI-compatible API (e.g. Venice.ai)
- **Web UI + REST API** — chat interface served by FastAPI; also usable headlessly via `main.py`

## Project Structure

```
Paper-Pal/
├── app.py           # FastAPI server + session management
├── main.py          # Headless CLI entry point
├── config.json      # Backend configuration (models, parameters, API keys)
├── src/
│   ├── agent.py     # Core agent logic (intent detection, tool calls, ranking)
│   ├── model.py     # LLM factory — builds the right client per backend
│   ├── tools.py     # PubMed search tool (LangChain @tool)
│   └── prompts.py   # System prompts for chat, intent, and ranker agents
└── static/
    └── index.html   # Web UI
```

## Requirements

- Python 3.11+
- Core dependencies:
  ```
  langchain-core
  langchain-ollama
  fastapi
  uvicorn
  pydantic
  requests
  ```
- Optional, install only the backend(s) you use:
  | Backend | Package |
  |---------|---------|
  | Ollama | `pip install langchain-ollama` |
  | AWS Bedrock | `pip install langchain-aws` |
  | OpenAI-compatible | `pip install langchain-openai` |

## Setup

### 1. Install dependencies

```bash
pip install langchain-core langchain-ollama fastapi uvicorn pydantic requests
```

### 2. Configure `config.json`

Open `config.json` and set the backend you want to use:

```json
{
  "default_backend": "ollama",

  "ollama": {
    "chat_model": "llama3.2:3b",
    "reasoning_model": "llama3.2:3b",
    "base_url": null,
    "temperature": 0.4,
    "top_k": 40,
    "top_p": 0.9,
    "mirostat": 0,
    "mirostat_eta": 1.0,
    "mirostat_tau": 5.0
  },

  "aws_bedrock": {
    "chat_model": "anthropic.claude-3-haiku-20240307-v1:0",
    "reasoning_model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "region": "us-east-1",
    "temperature": 0.4,
    "top_p": 0.9
  },

  "openai": {
    "chat_model": "llama-3.3-70b",
    "reasoning_model": "llama-3.3-70b",
    "base_url": "https://api.venice.ai/api/v1",
    "api_key": "",
    "temperature": 0.4,
    "top_p": 0.9
  }
}
```

Set `"default_backend"` to `"ollama"`, `"aws_bedrock"`, or `"openai"`.

> **API keys** — leave `api_key` blank and set the `OPENAI_API_KEY` environment variable instead to keep secrets out of the file.

### 3. Start the server

```bash
uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### 4. Headless / CLI usage

```bash
python main.py
```

Runs a single query (`SARS-CoV-2 Evolution` by default) and prints formatted results to the terminal.

## API

### `POST /api/chat`

Send a message and receive papers or a conversational reply.

**Request body:**
```json
{
  "session_id": "optional-uuid-to-continue-a-conversation",
  "message": "CRISPR off-target effects in human cells",
  "search_mode": "balanced",
  "backend": "ollama",
  "backend_config": {}
}
```

- `backend` — overrides `default_backend` from `config.json` for this session
- `backend_config` — key/value overrides merged on top of the `config.json` section for this backend
- `search_mode` — optional retrieval preset: `balanced`, `clinical`, `mechanism`, `latest`, `reviews`

**Response:**
```json
{
  "session_id": "abc-123",
  "message": null,
  "papers": [
    {
      "index": 1,
      "title": "Genome-wide CRISPR screen...",
      "authors": "Smith J, ...",
      "year": "2023",
      "journal": "Nature Biotechnology",
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
      "relevance": "Directly measures off-target editing rates..."
    }
  ]
}
```

When the query is conversational, `papers` is `null` and `message` contains the reply. When no relevant papers are found, `papers` is an empty array and `message` is set.

### `GET /api/config`

Returns the active backend configuration with secrets redacted. Useful for confirming which model and parameters are in use.

## Backend Configuration

### Ollama (local)

Requires [Ollama](https://ollama.com) running locally. Pull the model you want first:

```bash
ollama pull llama3.2:3b
```

Set `base_url` to `null` to use the default `http://localhost:11434`, or point it at a remote Ollama instance.

### AWS Bedrock

Requires `pip install langchain-aws`. Credentials are resolved through the standard AWS credential chain — environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), `~/.aws/credentials`, or an IAM role. No `api_key` needed in `config.json`.

### OpenAI-compatible (Venice.ai, OpenAI, etc.)

Requires `pip install langchain-openai`. Works with any OpenAI-compatible endpoint:

- **Venice.ai**: `"base_url": "https://api.venice.ai/api/v1"`
- **OpenAI**: remove `base_url` or set it to `null`

Set your key via the environment variable to avoid storing it in the file:
```bash
export OPENAI_API_KEY="your-key-here"
```

## How It Works

1. **Intent detection** — a lightweight LLM call classifies the message as `RESEARCH` or `CONVERSATIONAL`. Non-research messages skip the search pipeline entirely.
2. **Search** — the chat agent decides the optimal PubMed query and calls `searchPubMed` (max one call per turn).
3. **Validation** — every result block is matched against a strict regex. Malformed blocks (potential prompt injections) are discarded and logged.
4. **Ranking** — a second structured-output LLM call filters irrelevant papers, deduplicates, and adds a per-paper relevance sentence.
5. **Response** — the ranked list is returned to the client as structured JSON.


## Tuning the Samplers
More details here: [https://rentry.org/samplers](https://rentry.org/samplers)  
1. **Temperature**:  
    Think of this as the "creativity knob" on your LLM. At low temperatures (close to 0), the model becomes very cautious and predictable - it almost always picks the most likely next word. It's like ordering the same dish at your favourite restaurant every time because you know you'll like it (or maybe you don't know any better). At higher temperatures (like 0.7-1.0), the model gets very creative and willing to take chances. It may choose the 3rd or 4th most likely word instead of always the top choice. This makes text more varied and interesting, but also increases the chance of errors. Very high temperatures (above 1.0) make the model wild and unpredictable, unless you use it in conjunction with other sampling methods (e.g. min-p) to reign it in.

2. **Top-K**:  
    Instead of considering all possible next words (which could be tens of thousands), the model narrows down to only the K most likely candidates. If K is 40, the model will only choose from the top 40 most likely next words. This approach prevents the model from selecting extremely unlikely words while still maintaining some randomness.

3. **Top-P**:  
    Instead of picking a fixed number of options like Top-K, Top-P selects the smallest set of words whose combined probability exceeds threshold P. It's like saying "I'll only consider dishes that make up 90% of all orders at this restaurant." If P is 0.9, the model includes just enough of the highest-probability words to reach 90% cumulative probability, whether that's 5 words or 500. In situations where the model is very confident, it might only need a few options, but when uncertainty is high, it can consider more possibilities.

4. **Mirostat**:  
    It is like an adaptive thermostat for text generation that automatically adjusts to maintain a consistent level of "surprise" or unpredictability. Just as a thermostat keeps your room temperature stable by turning heating on and off, Mirostat keeps text generation at a consistent level of unpredictability by dynamically adjusting how conservative or creative the sampling is. It works by measuring the "surprisal" (how unexpected each token is) and comparing it to a target value. If recent text has been to predictable, Mirostat allows more surprising tokens to be selected. If it's been too chaotic, Mirostat tightens the constraints to focus on more predictable ones. This creates a feedback loop that maintains a consistent perplexity throughput the generated text.  

    Mirostat's main selling point is that it's self-regulating, adapting to different contexts without requiring manual parameter adjustment.  

    **mirostat**: Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)  
    
    **mirostat_eta**: Influences how quickly the algorithm responds to feedback from the generated text. A lower learning rate will result in slower adjustments, while a higher learning rate will make the algorithm more responsive. (Default: 0.1)  
    
    **mirostat_tau**: Controls the balance between coherence and diversity of the output. A lower value will result in more focused and coherent text. (Default: 5.0)  