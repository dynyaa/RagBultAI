"""Abstract base classes for the RAG provider plugin system.

Defines the contracts that all provider implementations must follow.
Each base class represents a swappable component in the RAG pipeline:
  - LLM (chat/generation)
  - Retriever (document search)
  - Chunker (text splitting)
  - Reranker (result reordering)
  - Embedder (vector generation)

To add a new provider, subclass the appropriate base class, implement
all abstract methods, and register it in providers/registry.py.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator, Optional


class BaseLLMProvider(ABC):
    """Abstract base for LLM backends.

    Implementations must support both streaming chat completion and
    single-shot generation. The streaming interface is used for the
    main chat endpoint; single-shot is used for internal tasks like
    query decomposition, HyDE generation, and conversation titling.

    Example usage:
        provider = OpenAILLMProvider(api_key="sk-...")
        async for token in provider.chat_stream(messages, model="gpt-4o"):
            print(token, end="")
    """

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}.
            model: Model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514").
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

        Yields:
            Individual text tokens as they are generated.
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.3,
    ) -> str:
        """Single-shot text generation.

        Used for internal tasks: query decomposition, HyDE, conversation
        titling, and context generation. Not streamed to the end user.

        Args:
            prompt: The full prompt string.
            model: Model identifier.
            temperature: Sampling temperature.

        Returns:
            The complete generated text.
        """
        pass


class BaseRetriever(ABC):
    """Abstract base for retrieval strategies.

    Implementations encapsulate the full retrieval pipeline: query
    transformation, vector/keyword search, and optional reranking.
    The returned dicts must include at minimum: content, chunk_id,
    document_id, filename, and similarity.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        project_id: int,
        user_id: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The user's search query.
            project_id: Scope retrieval to this project.
            user_id: The authenticated user (for data isolation).
            top_k: Maximum number of chunks to return.

        Returns:
            List of dicts, each containing at least:
              - content (str): The chunk text
              - chunk_id (int): Database ID of the chunk
              - document_id (int): Database ID of the source document
              - filename (str): Original filename
              - similarity (float): Relevance score (0-1)
        """
        pass


class BaseChunker(ABC):
    """Abstract base for text chunking strategies.

    Implementations split raw document text into smaller chunks suitable
    for embedding and retrieval. Each chunk dict should contain 'content'
    and optionally 'metadata' with page numbers, section info, etc.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata.

        Args:
            text: The full document text to split.
            metadata: Optional metadata to attach to each chunk
                      (e.g. filename, document_id).

        Returns:
            List of dicts, each containing:
              - content (str): The chunk text
              - metadata (dict): Merged metadata for this chunk
        """
        pass


class BaseReranker(ABC):
    """Abstract base for reranking retrieved chunks.

    Rerankers take an initial set of candidate chunks and reorder them
    by relevance to the query, typically using a cross-encoder model
    for higher accuracy than the initial vector similarity scores.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by relevance to query.

        Args:
            query: The user's search query.
            chunks: List of chunk dicts from initial retrieval.
            top_k: Maximum number of chunks to return after reranking.

        Returns:
            Reranked list of chunk dicts, trimmed to top_k.
        """
        pass


class BaseEmbedder(ABC):
    """Abstract base for embedding providers.

    Implementations generate vector embeddings for text, used for both
    document indexing (at ingestion time) and query embedding (at
    retrieval time). The dimension property must match the database
    column width (vector(N) in pgvector).

    Example usage:
        embedder = OpenAIEmbedder(api_key="sk-...")
        vectors = await embedder.embed(["Hello world", "Another text"])
        assert len(vectors[0]) == embedder.dimension
    """

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
            Each vector has length == self.dimension.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension.

        Must match the pgvector column width configured in
        Config.EMBEDDING_DIM.
        """
        pass
