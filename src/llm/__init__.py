import os
from typing import Optional
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain.output_parsers.json import SimpleJsonOutputParser

json_parser = SimpleJsonOutputParser()


@lru_cache(maxsize=None)
def get_llm(provider: str, model: str, base_url: Optional[str] = None):
    """
    Return a cached LLM instance based on provider and model.
    Supported providers: openai, groq, openrouter
    """
    timeout = 120  # for this in ollama can be very slow, so set it to a bigger value.
    max_retries = 3
    if provider == "openai":
        base_url = base_url or "https://api.openai.com/v1"
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "groq":
        base_url = base_url or "https://api.groq.com/v1"
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "openrouter":
        base_url = base_url or "https://openrouter.ai/api/v1"
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "gemini":
        base_url = base_url or "https://generativelanguage.googleapis.com/v1beta"
        return ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "anthropic":
        base_url = base_url or "https://api.anthropic.com"
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif provider == "minimax":
        base_url = base_url or "https://api.minimax.io/v1"
        # MiniMax requires temperature in (0.0, 1.0]
        return ChatOpenAI(
            api_key=os.getenv("MINIMAX_API_KEY"),
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            temperature=0.7,
        )
    elif provider == "ollama":
        base_url = base_url or "http://localhost:11434"
        return ChatOllama(
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


__all__ = ["json_parser", "get_llm"]

