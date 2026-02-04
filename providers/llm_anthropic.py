"""Anthropic LLM provider implementation.

Wraps the Anthropic AsyncAnthropic client to conform to the BaseLLMProvider
interface. Supports Claude models (claude-sonnet-4-20250514, claude-opus-4-20250514, etc.).

Usage:
    from providers.llm_anthropic import AnthropicLLMProvider

    provider = AnthropicLLMProvider(api_key="sk-ant-...")
    async for token in provider.chat_stream(messages, model="claude-sonnet-4-20250514"):
        print(token, end="")

Environment:
    ANTHROPIC_API_KEY - Required. Set via .env or shell.

Dependencies:
    pip install anthropic
"""

from typing import List, Dict, AsyncIterator

from providers.base import BaseLLMProvider


class AnthropicLLMProvider(BaseLLMProvider):
    """Anthropic LLM provider using the official AsyncAnthropic client.

    The Anthropic API uses a different message format than OpenAI:
    - System messages are passed as a separate 'system' parameter
    - The messages list contains only user/assistant turns
    This provider handles the conversion automatically.
    """

    def __init__(self, api_key: str = None):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If not provided, reads from the
                     ANTHROPIC_API_KEY environment variable.

        Raises:
            ValueError: If no API key is available.
            ImportError: If the anthropic package is not installed.
        """
        if not api_key:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY in your environment or pass api_key= to the constructor."
            )

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required for the Anthropic provider. "
                "Install it with: pip install anthropic"
            )

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _extract_system_message(
        messages: List[Dict[str, str]],
    ) -> tuple:
        """Separate system message from conversation messages.

        Anthropic's API expects the system prompt as a top-level parameter,
        not as a message in the list. This helper splits them apart.

        Returns:
            (system_text, filtered_messages) where system_text may be empty
            and filtered_messages contains only user/assistant messages.
        """
        system_text = ""
        filtered = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                filtered.append(msg)
        return system_text, filtered

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens from Anthropic.

        Uses the messages.stream() context manager for server-sent events.
        Automatically extracts system messages into the separate parameter.
        """
        system_text, filtered_messages = self._extract_system_message(messages)

        kwargs = {
            "model": model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if system_text:
            kwargs["system"] = system_text

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Single-shot generation via Anthropic messages API.

        Sends the prompt as a single user message and returns the
        complete response text.
        """
        response = await self._client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=4096,
        )
        # Anthropic returns a list of content blocks; concatenate text blocks
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)
