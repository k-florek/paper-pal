"""LLM backend factory utilities.

This module centralizes construction of LangChain chat models so all entry
points share the same backend selection and configuration behavior.
"""

from typing import Literal
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel

Backend = Literal["ollama", "aws_bedrock", "openai"]

class ModelType(Enum):
    """Logical model role used by the application pipeline."""

    CHAT = 1
    REASONING = 2


# Backward-compatible alias for older imports.
modelType = ModelType


def _get_model_name(config: dict, model_type: ModelType) -> str:
    """Return model name for the requested role from backend config."""
    key = "reasoning_model" if model_type == ModelType.REASONING else "chat_model"
    return config.get(key, "")


def build_llm(backend: Backend, config: dict, model_type: ModelType) -> BaseChatModel:
    """Return a LangChain chat model for the requested backend.

    Packages are imported lazily so that only the dependency for the chosen
    backend needs to be installed.

    Supported backends
    ------------------
    ollama  - local Ollama server (langchain-ollama)
                config keys: model, base_url, temperature, top_k, top_p,
                mirostat, mirostat_eta, mirostat_tau

    aws_bedrock - AWS Bedrock (langchain-aws)
                    config keys: model_id, region, temperature, top_p
                    Credentials come from the standard AWS credential chain
                    (env vars, ~/.aws/credentials, IAM role).

    openai  - OpenAI-compatible API (langchain-openai)
                config keys: model, api_key, base_url, temperature, top_p

    Args:
        backend: Backend identifier.
        config: Backend configuration map.
        model_type: Whether to construct the chat or reasoning model.

    Returns:
        A configured LangChain chat model instance.

    Raises:
        ImportError: If backend-specific dependency is not installed.
        ValueError: If the backend identifier is unsupported.
    """
    temperature: float = config.get("temperature", 0.4)
    top_p: float = config.get("top_p", 0.9)
    model_name = _get_model_name(config, model_type)

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Install langchain-ollama to use the Ollama backend: "
                "pip install langchain-ollama"
            ) from exc


        proxy: str | None = config.get("proxy") or None
        client_kwargs: dict = {"proxy": proxy} if proxy else {}

        return ChatOllama(
            model=model_name,
            base_url=config.get("base_url") or None,
            temperature=temperature,
            top_k=config.get("top_k", 40),
            top_p=top_p,
            mirostat=config.get("mirostat", 0),
            mirostat_eta=config.get("mirostat_eta", 1.0),
            mirostat_tau=config.get("mirostat_tau", 5.0),
            client_kwargs=client_kwargs,
        )

    if backend == "aws_bedrock":
        try:
            from langchain_aws import ChatBedrockConverse  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Install langchain-aws to use the AWS Bedrock backend: "
                "pip install langchain-aws"
            ) from exc


        return ChatBedrockConverse(
            model_id=model_name,
            region_name=config.get("region", "us-east-1"),
            temperature=temperature,
            top_p=top_p,
        )

    if backend == "openai":
        try:
            from langchain_openai import ChatOpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Install langchain-openai to use the OpenAI backend: "
                "pip install langchain-openai"
            ) from exc

        return ChatOpenAI(
            model=model_name,
            base_url=config.get("base_url") or None,
            api_key=config.get("api_key"),
            temperature=temperature,
            top_p=top_p,
        )

    raise ValueError(f"Unknown backend {backend!r}. Choose 'ollama', 'aws_bedrock', or 'openai'.")