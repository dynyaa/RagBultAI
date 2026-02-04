"""OpenAI LLM provider implementation.

Wraps the OpenAI AsyncOpenAI client to conform to the BaseLLMProvider
interface. This is the default provider and mirrors the streaming logic
previously embedded directly in app.py's /api/chat endpoint.

Usage:
    from providers.llm_openai import OpenAILLMProvider

    provider = OpenAILLMProvider(api_key="sk-...")
    async for token in provider.chat_stream(messages, model="gpt-4o"):
        print(token, end="")

Environment:
    OPENAI_API_KEY - Required. Set via .env or shell.
"""

from typing import List, Dict, AsyncIterator

from openai import AsyncOpenAI

from providers.base import BaseLLMProvider


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI LLM provider using the official AsyncOpenAI client.

    Supports all OpenAI chat models (gpt-4o, gpt-4o-mini, o1, o3-mini, etc.).
    The streaming implementation uses the chat.completions API with stream=True.
    """

    def __init__(self, api_key: str = None):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, reads from the
                     OPENAI_API_KEY environment variable.

        Raises:
            ValueError: If no API key is available.
        """
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY in your environment or pass api_key= to the constructor."
            )
        self._client = AsyncOpenAI(api_key=api_key)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens from OpenAI.

        Uses the chat.completions.create endpoint with stream=True.
        Each yielded string is a single token delta.
        """
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Single-shot generation via OpenAI chat completions.

        Sends the prompt as a single user message and returns the
        complete response text.
        """
        response = await self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
