import os
from typing import Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


openai_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",  # or "gpt-4" if you have access
    timeout=30,           # timeout in seconds (set to your desired limit)
    max_retries=3
)

# Groq LLM
groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    timeout=30,
    max_retries=3,
)

# OpenRouter LLM
openrouter_llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="deepseek/deepseek-r1-0528:free",
    timeout=30,
    max_retries=3,
)

default_llm = openrouter_llm

json_parser = JsonOutputParser()

__all__ = ["openai_llm", "json_parser", "groq_llm", "openrouter_llm", "default_llm"]
