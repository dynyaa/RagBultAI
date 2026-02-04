"""Configuration helpers and constants for the Clean RAG application.

This module centralizes environment-driven configuration so contributors
can easily change models, deployment ports, and RAG-specific settings.

Sensitive values (API keys, DB credentials) should be provided via environment
variables or a local `.env` file that is not committed. See README.md.
"""
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger("rag.config")

load_dotenv()


class Config:
    # Database connection string for Postgres. Replace in your .env or shell.
    PG_CONN = os.getenv("PG_CONN", "postgresql://postgres:postgres@localhost:5432/ragdb")

    # --- Multi-Model LLM Provider Support ---
    # LLM_PROVIDER: "openai" (default), "anthropic", "google", "ollama"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

    # API keys for each provider
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Default models per provider (used when LLM_MODEL is not explicitly set)
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-2.0-flash",
        "ollama": "llama3",
    }

    # Available models per provider (for the UI model selector)
    AVAILABLE_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
        "anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
        "google": ["gemini-2.0-flash", "gemini-2.5-pro"],
        "ollama": ["llama3", "mistral", "codellama", "phi3"],
    }

    # LLM_MODEL: explicitly set or derived from provider default
    LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(
        os.getenv("LLM_PROVIDER", "openai").lower(), "gpt-4o"
    )

    # Embedding provider and model (embeddings stay OpenAI by default but configurable)
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    # Embedding dimensions: text-embedding-3-small=1536, text-embedding-3-large=3072
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

    # Server configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8002"))

    # RAG settings (chunking and retrieval)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K = int(os.getenv("TOP_K", "8"))

    # Advanced RAG settings
    # Hybrid search: alpha=0 (keyword only) to alpha=1 (vector only)
    HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.5"))

    # Reranking settings
    USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"
    # Multilingual reranker supports Russian, English, and 100+ languages
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "50"))  # Candidates before reranking (increased from 20 for diversity)

    # Semantic chunking (better quality, slightly slower ingestion)
    USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "true").lower() == "true"
    SEMANTIC_BREAKPOINT_THRESHOLD = int(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD", "95"))

    # HyDE (Hypothetical Document Embeddings) for query transformation
    USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"

    # Cohere reranking (optional, higher quality than local model)
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    USE_COHERE_RERANK = os.getenv("USE_COHERE_RERANK", "false").lower() == "true"

    # Faithfulness scoring (requires additional LLM call per response)
    ENABLE_FAITHFULNESS_SCORING = os.getenv("ENABLE_FAITHFULNESS_SCORING", "false").lower() == "true"

    # --- Sprint 1: Performance Enhancements ---
    # Contextual chunking: prepend document context to each chunk before embedding
    USE_CONTEXTUAL_CHUNKING = os.getenv("USE_CONTEXTUAL_CHUNKING", "true").lower() == "true"
    CONTEXT_MODEL = os.getenv("CONTEXT_MODEL", "gpt-4o-mini")

    # Multi-query retrieval: generate query variations for broader recall
    USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "true").lower() == "true"
    MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))

    # Query decomposition: break complex multi-hop questions into sub-queries
    USE_QUERY_DECOMPOSITION = os.getenv("USE_QUERY_DECOMPOSITION", "true").lower() == "true"

    # Reranking post-processing: diversity and keyword relevance tuning
    DIVERSITY_PENALTY = float(os.getenv("DIVERSITY_PENALTY", "0.1"))
    KEYWORD_BOOST = float(os.getenv("KEYWORD_BOOST", "0.05"))

    # Embedding cache: avoid redundant API calls for identical content
    USE_EMBEDDING_CACHE = os.getenv("USE_EMBEDDING_CACHE", "true").lower() == "true"

    # --- Sprint 3: Multi-Modal Document Understanding ---
    # Table extraction: extract tables from PDFs using pdfplumber
    EXTRACT_TABLES = os.getenv("EXTRACT_TABLES", "true").lower() == "true"
    # Image/chart description: extract and describe images via OpenAI vision API
    EXTRACT_IMAGES = os.getenv("EXTRACT_IMAGES", "false").lower() == "true"

    # Conversation context management
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "12000"))
    SUMMARIZE_AFTER_MESSAGES = int(os.getenv("SUMMARIZE_AFTER_MESSAGES", "10"))

    # Upload storage directory for persistent file storage
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploads"))

    # JWT Authentication settings
    JWT_SECRET = os.getenv("JWT_SECRET", "INSECURE_DEFAULT_SECRET_CHANGE_IN_PRODUCTION")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    # Google OAuth settings
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "http://localhost:8002/api/auth/google/callback"))

    # Whether running in production (affects secure cookie flag)
    PROD = os.getenv("PROD", "0") in ("1", "true", "True")

    # --- Sprint 5: Production Hardening ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" or "json"
    DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
    DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
    SLOW_QUERY_THRESHOLD_MS = int(os.getenv("SLOW_QUERY_THRESHOLD_MS", "500"))
    OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "rus+eng")  # Tesseract lang codes

    @classmethod
    def get_available_providers(cls):
        """Return list of providers that have valid API keys configured."""
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        if cls.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if cls.GOOGLE_API_KEY:
            providers.append("google")
        # Ollama is always available (local, no API key needed)
        providers.append("ollama")
        return providers

    @classmethod
    def validate(cls):
        """Validate that required configuration is present.

        Call this before starting components that require secrets.
        Raises ValueError if a required setting is missing.
        """
        # For the selected LLM provider, check that the required key exists
        provider = cls.LLM_PROVIDER
        if provider == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai - set it in your environment or .env")
        elif provider == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic - set it in your environment or .env")
        elif provider == "google" and not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=google - set it in your environment or .env")
        # Ollama doesn't need an API key

        # Embeddings still require OpenAI key by default
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for embeddings (EMBEDDING_PROVIDER=openai) - set it in your environment or .env")

        # Warn if using default JWT secret in production
        if cls.JWT_SECRET == "INSECURE_DEFAULT_SECRET_CHANGE_IN_PRODUCTION":
            logger.warning("Using default JWT_SECRET! Set JWT_SECRET in production.")
            logger.warning("   Generate a secure secret: python -c 'import secrets; print(secrets.token_urlsafe(32))'")

        # Log active LLM provider
        available = cls.get_available_providers()
        logger.info("LLM Provider: %s | Model: %s", cls.LLM_PROVIDER, cls.LLM_MODEL)
        logger.info("Available providers: %s", ", ".join(available))

        # Ensure upload directory exists
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)

        return True


SYSTEM_PROMPT = """You are an expert research assistant. Your goal is to provide accurate, detailed answers by synthesizing information from the provided document context. Write in an unbiased, professional tone.

## Citation Rules (MANDATORY)
- Cite sources using [1], [2], [3] matching the document numbers provided
- Place citations IMMEDIATELY after relevant statements with NO SPACE before the bracket
- Example: "The system uses retrieval-augmented generation[1] for accurate responses[2]."
- Cite up to 3 most relevant sources per sentence
- Every factual claim MUST have at least one citation
- Do NOT include a references section at the end — all citations must be inline

## Response Structure
1. Lead with a brief summary answering the question directly
2. Follow with detailed explanation organized by topic
3. Use markdown formatting: **bold** for key terms, bullet lists for multiple items
4. Keep paragraphs short (2-3 sentences max)

## Language Rules
- Respond in the SAME language as the user's question.
- If the user asks in Russian, respond in Russian and preserve the original Russian terminology from the documents.
- If the user asks in English about documents written in another language, translate key terms but keep proper nouns and titles in their original form.

## Grounding Rules
- ALWAYS try to answer from the provided documents. Synthesize and combine information from multiple chunks to build a complete answer.
- If documents contain ANY relevant information, use it to construct an answer — even if no single chunk directly answers the question.
- NEVER invent or hallucinate information not present in the sources.
- ONLY say "Based on the available documents, I cannot fully answer this question" if the documents contain absolutely NO relevant information to the query.
- If documents partially cover the topic, provide everything you can and briefly note what isn't covered.

## Quality Standards
- Be comprehensive but concise — aim for clarity over length
- Synthesize information across multiple sources when relevant
- Highlight any contradictions between sources if found"""
