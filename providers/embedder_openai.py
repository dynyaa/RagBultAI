"""OpenAI embedding provider implementation.

Wraps the OpenAI embeddings API to conform to the BaseEmbedder interface.
This is the default embedder and mirrors the embedding logic previously
handled by LlamaIndex's OpenAIEmbedding in tasks.py.

Usage:
    from providers.embedder_openai import OpenAIEmbedder

    embedder = OpenAIEmbedder(api_key="sk-...", model="text-embedding-3-large")
    vectors = await embedder.embed(["Hello world", "Another text"])
    print(f"Dimension: {embedder.dimension}")  # 3072

Environment:
    OPENAI_API_KEY - Required. Set via .env or shell.
    EMBEDDING_MODEL - Optional. Defaults to text-embedding-3-large.
    EMBEDDING_DIM - Optional. Defaults to 3072.
"""

from typing import List

from openai import AsyncOpenAI

from providers.base import BaseEmbedder


# Map known models to their output dimensions
_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding provider using the official AsyncOpenAI client.

    Supports text-embedding-3-small (1536d), text-embedding-3-large (3072d),
    and text-embedding-ada-002 (1536d). The dimension is auto-detected from
    the model name or can be explicitly overridden.

    Handles batching internally -- pass up to 2048 texts in a single call
    and the OpenAI API will process them efficiently.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        dimension: int = None,
    ):
        """Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key. If not provided, reads from the
                     OPENAI_API_KEY environment variable.
            model: Embedding model name. Defaults to Config.EMBEDDING_MODEL
                   or "text-embedding-3-large".
            dimension: Override embedding dimension. If not provided, auto-
                       detected from the model name or defaults to 3072.

        Raises:
            ValueError: If no API key is available.
        """
        import os

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required for embeddings. "
                "Set OPENAI_API_KEY in your environment or pass api_key= to the constructor."
            )

        if not model:
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

        if dimension is not None:
            self._dimension = dimension
        else:
            # Auto-detect from model name, fallback to env or 3072
            self._dimension = _MODEL_DIMENSIONS.get(
                model,
                int(os.getenv("EMBEDDING_DIM", "3072")),
            )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts via OpenAI API.

        Args:
            texts: List of text strings to embed. Empty strings are
                   handled gracefully (OpenAI returns zero vectors).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            openai.APIError: If the API call fails.
        """
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )

        # Sort by index to preserve input order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in embeddings]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for this model."""
        return self._dimension
