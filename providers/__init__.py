"""Provider plugin system for the RAG application.

This package contains abstract base classes defining the provider contracts
and concrete implementations for each supported backend. The registry module
provides factory functions for instantiating providers by name.

Quick start:
    from providers import get_llm_provider, get_embedder

    # Get the configured LLM provider (reads Config.LLM_PROVIDER)
    llm = get_llm_provider()

    # Or request a specific one
    llm = get_llm_provider("anthropic")

    # Get the configured embedder (reads Config.EMBEDDING_PROVIDER)
    embedder = get_embedder()

Base classes (for implementing new providers):
    from providers.base import (
        BaseLLMProvider,
        BaseRetriever,
        BaseChunker,
        BaseReranker,
        BaseEmbedder,
    )
"""

# Base classes
from providers.base import (
    BaseLLMProvider,
    BaseRetriever,
    BaseChunker,
    BaseReranker,
    BaseEmbedder,
)

# Factory functions
from providers.registry import (
    get_llm_provider,
    get_embedder,
    list_llm_providers,
    list_embedder_providers,
)

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "BaseRetriever",
    "BaseChunker",
    "BaseReranker",
    "BaseEmbedder",
    # Factory functions
    "get_llm_provider",
    "get_embedder",
    "list_llm_providers",
    "list_embedder_providers",
]
