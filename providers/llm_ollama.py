"""Ollama LLM provider implementation.

Wraps the Ollama HTTP API (localhost:11434 by default) to conform to the
BaseLLMProvider interface. Ollama runs open-source models locally with
no API key required.

Usage:
    from providers.llm_ollama import OllamaLLMProvider

    provider = OllamaLLMProvider(base_url="http://localhost:11434")
    async for token in provider.chat_stream(messages, model="llama3"):
        print(token, end="")

Environment:
    OLLAMA_BASE_URL - Optional. Defaults to http://localhost:11434.

Dependencies:
    pip install httpx
"""

import json
from typing import List, Dict, AsyncIterator

from providers.base import BaseLLMProvider


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama LLM provider using the HTTP REST API.

    Communicates with a locally running Ollama server via httpx.
    Supports all Ollama-compatible models (llama3, mistral, codellama,
    phi3, etc.). No API key is required.

    The streaming implementation uses Ollama's /api/chat endpoint with
    stream=true, which returns newline-delimited JSON objects.
    """

    def __init__(self, base_url: str = None):
        """Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL. If not provided, reads from the
                      OLLAMA_BASE_URL environment variable, defaulting to
                      http://localhost:11434.
        """
        if not base_url:
            import os
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._base_url = base_url.rstrip("/")

        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError(
                "The httpx package is required for the Ollama provider. "
                "Install it with: pip install httpx"
            )

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens from Ollama.

        Uses the /api/chat endpoint with stream=true. Ollama returns
        newline-delimited JSON, where each line has a "message.content"
        field containing the token delta.
        """
        import httpx

        url = f"{self._base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        # Stop when Ollama signals done
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Single-shot generation via Ollama generate API.

        Uses the /api/generate endpoint with stream=false for a complete
        response in a single request.
        """
        import httpx

        url = f"{self._base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
