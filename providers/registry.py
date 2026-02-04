"""Provider registry and factory functions.

Central lookup for all provider implementations. The factory functions
read the configured provider name from Config (or accept an override)
and return an initialized instance of the appropriate provider class.

Usage:
    from providers.registry import get_llm_provider, get_embedder

    llm = get_llm_provider()           # uses Config.LLM_PROVIDER
    llm = get_llm_provider("anthropic") # explicit override

    embedder = get_embedder()           # uses Config.EMBEDDING_PROVIDER
"""

from typing import Optional

from providers.base import BaseLLMProvider, BaseEmbedder


# Registry of known LLM providers: name -> (module_path, class_name)
# Lazy-imported to avoid pulling in optional dependencies at startup.
_LLM_REGISTRY = {
    "openai": ("providers.llm_openai", "OpenAILLMProvider"),
    "anthropic": ("providers.llm_anthropic", "AnthropicLLMProvider"),
    "ollama": ("providers.llm_ollama", "OllamaLLMProvider"),
}

# Registry of known embedder providers
_EMBEDDER_REGISTRY = {
    "openai": ("providers.embedder_openai", "OpenAIEmbedder"),
}


def _import_class(module_path: str, class_name: str):
    """Dynamically import a class from a module path.

    Used for lazy loading so that optional dependencies (anthropic, httpx)
    are only imported when the corresponding provider is actually requested.
    """
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_llm_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    """Factory that returns the configured LLM provider.

    Reads the provider name from Config.LLM_PROVIDER if not explicitly
    provided. Initializes the provider with the appropriate API key or
    connection settings from Config.

    Args:
        provider_name: Override provider name. If None, uses Config.LLM_PROVIDER.
                       Supported values: "openai", "anthropic", "ollama".

    Returns:
        An initialized BaseLLMProvider instance.

    Raises:
        ValueError: If the provider name is unknown or required configuration
                    (API keys, etc.) is missing.
    """
    from config import Config

    if provider_name is None:
        provider_name = Config.LLM_PROVIDER
    provider_name = provider_name.lower()

    if provider_name not in _LLM_REGISTRY:
        available = ", ".join(sorted(_LLM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Supported providers: {available}"
        )

    module_path, class_name = _LLM_REGISTRY[provider_name]
    provider_class = _import_class(module_path, class_name)

    # Pass provider-specific configuration
    if provider_name == "openai":
        return provider_class(api_key=Config.OPENAI_API_KEY)
    elif provider_name == "anthropic":
        return provider_class(api_key=Config.ANTHROPIC_API_KEY)
    elif provider_name == "ollama":
        return provider_class(base_url=Config.OLLAMA_BASE_URL)
    else:
        # Fallback: try to instantiate with no args
        return provider_class()


def get_embedder(provider_name: Optional[str] = None) -> BaseEmbedder:
    """Factory that returns the configured embedding provider.

    Reads the provider name from Config.EMBEDDING_PROVIDER if not
    explicitly provided.

    Args:
        provider_name: Override provider name. If None, uses
                       Config.EMBEDDING_PROVIDER. Currently supported: "openai".

    Returns:
        An initialized BaseEmbedder instance.

    Raises:
        ValueError: If the provider name is unknown or required configuration
                    is missing.
    """
    from config import Config

    if provider_name is None:
        provider_name = Config.EMBEDDING_PROVIDER
    provider_name = provider_name.lower()

    if provider_name not in _EMBEDDER_REGISTRY:
        available = ", ".join(sorted(_EMBEDDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown embedding provider: '{provider_name}'. "
            f"Supported providers: {available}"
        )

    module_path, class_name = _EMBEDDER_REGISTRY[provider_name]
    embedder_class = _import_class(module_path, class_name)

    # Pass provider-specific configuration
    if provider_name == "openai":
        return embedder_class(
            api_key=Config.OPENAI_API_KEY,
            model=Config.EMBEDDING_MODEL,
            dimension=Config.EMBEDDING_DIM,
        )
    else:
        return embedder_class()


def list_llm_providers() -> dict:
    """Return a dict of all registered LLM providers and their availability.

    Useful for the /api/models endpoint to show which providers are
    configured and ready to use.

    Returns:
        Dict mapping provider name to {"available": bool, "module": str}.
    """
    from config import Config

    available_set = set(Config.get_available_providers())
    result = {}
    for name in _LLM_REGISTRY:
        result[name] = {
            "available": name in available_set,
            "module": _LLM_REGISTRY[name][0],
        }
    return result


def list_embedder_providers() -> dict:
    """Return a dict of all registered embedding providers.

    Returns:
        Dict mapping provider name to {"available": bool, "module": str}.
    """
    from config import Config

    result = {}
    for name in _EMBEDDER_REGISTRY:
        # Currently only OpenAI embeddings require a key
        available = bool(Config.OPENAI_API_KEY) if name == "openai" else True
        result[name] = {
            "available": available,
            "module": _EMBEDDER_REGISTRY[name][0],
        }
    return result
