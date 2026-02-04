"""RAG (Retrieval-Augmented Generation) FastAPI app.

This service provides endpoints to upload documents, chunk and embed
them for retrieval, and run conversational RAG queries using LlamaIndex
with OpenAI as the LLM and embedding provider.

The code intentionally keeps the pipeline simple and easy to customize
for contributors. See README.md for developer setup and deployment notes.
"""
import os
import json
import logging
import uuid
import threading
import time
from typing import List, Optional
from pathlib import Path

from core.logging_config import setup_logging

# Configure logging before anything else
setup_logging()

logger = logging.getLogger("rag.app")

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
import importlib.metadata as importlib_metadata
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import timedelta, datetime, timezone

from core.db import init_pool, close_pool, get_db
from core.auth import create_user, authenticate_user, get_current_user, create_access_token, hash_password, verify_password

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import AsyncOpenAI

from core.config import Config, SYSTEM_PROMPT

# Optional LLM provider imports - handle gracefully if not installed
try:
    from llama_index.llms.anthropic import Anthropic as AnthropicLLM
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from llama_index.llms.gemini import Gemini as GeminiLLM
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from llama_index.llms.ollama import Ollama as OllamaLLM
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


def get_llm_client(provider: str = None, model: str = None):
    """Factory function that returns the right LlamaIndex LLM based on provider.

    Args:
        provider: LLM provider name ("openai", "anthropic", "google", "ollama").
                  Defaults to Config.LLM_PROVIDER.
        model: Model name to use. Defaults to Config.LLM_MODEL or provider default.

    Returns:
        A LlamaIndex LLM instance configured for the specified provider.

    Raises:
        ValueError: If the provider is unknown or required dependencies are missing.
    """
    provider = (provider or Config.LLM_PROVIDER).lower()
    model = model or Config.LLM_MODEL

    # If provider changed from config default, use provider-specific default model
    if provider != Config.LLM_PROVIDER and model == Config.LLM_MODEL:
        model = Config.DEFAULT_MODELS.get(provider, model)

    if provider == "openai":
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        return OpenAI(model=model, api_key=Config.OPENAI_API_KEY)

    elif provider == "anthropic":
        if not HAS_ANTHROPIC:
            raise ValueError(
                "llama-index-llms-anthropic package not installed. "
                "Run: pip install llama-index-llms-anthropic"
            )
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
        return AnthropicLLM(model=model, api_key=Config.ANTHROPIC_API_KEY)

    elif provider == "google":
        if not HAS_GEMINI:
            raise ValueError(
                "llama-index-llms-gemini package not installed. "
                "Run: pip install llama-index-llms-gemini"
            )
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for Google provider")
        return GeminiLLM(model=model, api_key=Config.GOOGLE_API_KEY)

    elif provider == "ollama":
        if not HAS_OLLAMA:
            raise ValueError(
                "llama-index-llms-ollama package not installed. "
                "Run: pip install llama-index-llms-ollama"
            )
        return OllamaLLM(model=model, base_url=Config.OLLAMA_BASE_URL)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Supported: openai, anthropic, google, ollama"
        )


# Runtime model override (per-session, mutable at runtime via API)
_runtime_provider = Config.LLM_PROVIDER
_runtime_model = Config.LLM_MODEL


def get_runtime_provider_model():
    """Get the currently active provider and model (may be changed at runtime)."""
    return _runtime_provider, _runtime_model


# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.pptx', '.ppt',
    '.txt', '.md', '.csv', '.json', '.html', '.htm',
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
    '.xml', '.yaml', '.yml', '.ini', '.cfg', '.conf'
}

# Validate configuration and configure LlamaIndex
# Contributors: ensure environment variables are set (see README)
Config.validate()

Settings.llm = get_llm_client()
Settings.embed_model = OpenAIEmbedding(
    model=Config.EMBEDDING_MODEL,
    api_key=Config.OPENAI_API_KEY,
)
Settings.chunk_size = Config.CHUNK_SIZE
Settings.chunk_overlap = Config.CHUNK_OVERLAP

openapi_tags = [
    {
        "name": "Health",
        "description": "Application health and readiness checks.",
    },
    {
        "name": "Auth",
        "description": "User registration, login, OAuth, token management, and password operations.",
    },
    {
        "name": "Projects",
        "description": "CRUD operations for projects that organize documents and conversations.",
    },
    {
        "name": "Documents",
        "description": "Upload, list, delete, reprocess documents and view document statistics.",
    },
    {
        "name": "Conversations",
        "description": "Create, list, rename, and delete conversations within projects.",
    },
    {
        "name": "Chat",
        "description": "RAG-powered streaming chat with citation support and conversation history.",
    },
    {
        "name": "Models",
        "description": "List available LLM providers/models and switch the active model at runtime.",
    },
    {
        "name": "Analytics",
        "description": "Usage analytics: query volume, document stats, citation rankings, and latency metrics.",
    },
]

app = FastAPI(
    title="RAG Chat API",
    description=(
        "Enterprise RAG system with multi-model support, hybrid search, "
        "and advanced document processing. Features include multi-user "
        "authentication, document chunking with OCR, vector + BM25 hybrid "
        "retrieval, cross-encoder reranking, and streaming chat responses "
        "with inline citations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=openapi_tags,
)


# --- Request ID Middleware ---------------------------------------------------
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects a unique X-Request-ID header into every request/response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler that returns JSON instead of HTML.

    This helps the frontend avoid parsing HTML error pages when API
    endpoints fail during deployment or runtime (e.g., import errors,
    DB connectivity issues). The full traceback is printed to logs.
    """
    logger.error("Unhandled exception (global handler):", exc_info=exc)
    return JSONResponse(status_code=500, content={
        "detail": "Internal server error",
        "error": str(exc)
    })

# Background worker thread for processing document jobs
def background_worker():
    """Background thread that processes queued document jobs.

    This runs as a daemon thread alongside the API server, polling the jobs
    table for queued jobs and processing them asynchronously.
    """
    from core.tasks import process_document

    logger.info("Background worker started")
    poll_interval = 2  # seconds

    while True:
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    # Lock and fetch next queued job (priority DESC so retries run first)
                    cur.execute("""
                        SELECT id, document_id
                        FROM jobs
                        WHERE status = 'queued'
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    """)
                    row = cur.fetchone()

                    if not row:
                        time.sleep(poll_interval)
                        continue

                    job_id, document_id = row

                    # Mark job as processing
                    cur.execute("""
                        UPDATE jobs
                        SET status = 'processing', started_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (job_id,))

                    # Also update document status
                    cur.execute(
                        "UPDATE documents SET status = 'processing' WHERE id = %s",
                        (document_id,)
                    )
                conn.commit()

            # Process the document
            logger.info("Processing job %d (document %d)...", job_id, document_id)

            try:
                process_document(document_id, job_id)
                logger.info("Job %d completed successfully", job_id)

            except Exception as e:
                logger.error("Job %d failed: %s", job_id, e)

                # Dead letter queue: check retry_count vs max_retries
                with get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT retry_count, max_retries FROM jobs WHERE id = %s",
                            (job_id,)
                        )
                        retry_row = cur.fetchone()
                        retry_count = (retry_row[0] or 0) if retry_row else 0
                        max_retries = (retry_row[1] or 3) if retry_row else 3

                        if retry_count + 1 >= max_retries:
                            # Exhausted retries -> dead letter
                            cur.execute("""
                                UPDATE jobs
                                SET status = 'dead_letter', error_message = %s,
                                    retry_count = retry_count + 1,
                                    completed_at = CURRENT_TIMESTAMP
                                WHERE id = %s
                            """, (str(e), job_id))
                            cur.execute("""
                                UPDATE documents
                                SET status = 'error', error_message = %s
                                WHERE id = %s
                            """, (f"Processing failed after {max_retries} attempts: {e}", document_id))
                            logger.warning("Job %d moved to dead_letter after %d retries", job_id, retry_count + 1)
                        else:
                            # Re-queue for retry
                            cur.execute("""
                                UPDATE jobs
                                SET status = 'queued', error_message = %s,
                                    retry_count = retry_count + 1,
                                    started_at = NULL, progress = 0
                                WHERE id = %s
                            """, (str(e), job_id))
                            cur.execute("""
                                UPDATE documents
                                SET status = 'queued'
                                WHERE id = %s
                            """, (document_id,))
                            logger.info("Job %d re-queued (retry %d/%d)", job_id, retry_count + 1, max_retries)
                    conn.commit()

        except Exception as e:
            logger.error("Worker error: %s", e)
            time.sleep(poll_interval)

# ============================================================
# Database Bootstrap Functions (Ensure Pattern)
# ============================================================

def ensure_pgvector():
    """Ensure pgvector extension is enabled."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    logger.info("pgvector extension enabled")

def ensure_users_table():
    """Ensure users table exists."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255),
                    google_id VARCHAR(255) UNIQUE,
                    oauth_provider VARCHAR(50) DEFAULT 'google',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login_at TIMESTAMP,
                    CONSTRAINT email_or_google CHECK (
                        (password_hash IS NOT NULL) OR (google_id IS NOT NULL)
                    )
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS users_email_idx ON users(email);")
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS users_google_id_idx
                ON users(google_id) WHERE google_id IS NOT NULL;
            """)
        conn.commit()
    logger.info("users table ready")

def ensure_projects_table():
    """Ensure projects table exists."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    logger.info("projects table ready")

def ensure_documents_table():
    """Ensure documents table exists."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    filename VARCHAR(500) NOT NULL,
                    file_type VARCHAR(50),
                    file_path TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    status VARCHAR(50) DEFAULT 'queued',
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT
                );
            """)
        conn.commit()
    logger.info("documents table ready")

def ensure_conversations_table():
    """Ensure conversations table exists."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    title VARCHAR(200) DEFAULT 'New Conversation',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    logger.info("conversations table ready")

def ensure_messages_table():
    """Ensure messages table exists."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Add feedback columns if not present (user thumbs up/down)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'messages' AND column_name = 'feedback'
                    ) THEN
                        ALTER TABLE messages ADD COLUMN feedback TEXT;
                    END IF;
                END $$;
            """)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'messages' AND column_name = 'feedback_comment'
                    ) THEN
                        ALTER TABLE messages ADD COLUMN feedback_comment TEXT;
                    END IF;
                END $$;
            """)
        conn.commit()
    logger.info("messages table ready")

def ensure_chunks_table():
    """Ensure chunks table with vector embeddings and full-text search exists."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding vector({Config.EMBEDDING_DIM}),
                    chunk_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_project_idx ON chunks(project_id);")
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS chunks_doc_idx_unique
                ON chunks(document_id, chunk_index);
            """)
            # Vector index - may fail if no data exists or dims > 2000
            # Use savepoint so failure doesn't abort the entire transaction
            cur.execute("SAVEPOINT vector_idx")
            try:
                if Config.EMBEDDING_DIM <= 2000:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
                        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                    """)
                else:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
                        USING hnsw (embedding vector_cosine_ops);
                    """)
            except Exception as e:
                logger.warning("Vector index creation skipped: %s", e)
                cur.execute("ROLLBACK TO SAVEPOINT vector_idx")

            # Add page_number column if not exists
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'chunks' AND column_name = 'page_number'
                    ) THEN
                        ALTER TABLE chunks ADD COLUMN page_number INTEGER;
                    END IF;
                END $$;
            """)

            # Add content_type column for multi-modal chunks (Sprint 3)
            # Values: 'text' (default), 'table', 'image'
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'chunks' AND column_name = 'content_type'
                    ) THEN
                        ALTER TABLE chunks ADD COLUMN content_type TEXT DEFAULT 'text';
                    END IF;
                END $$;
            """)

            # Add tsvector column for hybrid search (BM25 + vector)
            # Use 'simple' config for multilingual support (Russian, English, etc.)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'chunks' AND column_name = 'content_tsv'
                    ) THEN
                        ALTER TABLE chunks ADD COLUMN content_tsv tsvector
                            GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;
                    END IF;
                END $$;
            """)
            # Migrate existing 'english' tsvector to 'simple' for multilingual support
            cur.execute("""
                DO $$
                DECLARE
                    col_default text;
                BEGIN
                    SELECT pg_get_expr(d.adbin, d.adrelid) INTO col_default
                    FROM pg_attribute a
                    JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
                    WHERE a.attrelid = 'chunks'::regclass AND a.attname = 'content_tsv';

                    IF col_default IS NOT NULL AND col_default LIKE '%english%' THEN
                        ALTER TABLE chunks DROP COLUMN content_tsv;
                        ALTER TABLE chunks ADD COLUMN content_tsv tsvector
                            GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;
                    END IF;
                END $$;
            """)
            # GIN index for full-text search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
                ON chunks USING GIN(content_tsv);
            """)
        conn.commit()
    logger.info("chunks table ready (with hybrid search support)")

def ensure_jobs_table():
    """Ensure jobs table for async processing exists."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    status VARCHAR(50) DEFAULT 'queued',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS jobs_status_idx
                ON jobs(status, created_at);
            """)
            # Sprint 5 Phase B: Add priority, progress, retry_count, max_retries columns
            for col, col_def in [
                ("priority", "INTEGER DEFAULT 0"),
                ("progress", "INTEGER DEFAULT 0"),
                ("retry_count", "INTEGER DEFAULT 0"),
                ("max_retries", "INTEGER DEFAULT 3"),
            ]:
                cur.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'jobs' AND column_name = '{col}'
                        ) THEN
                            ALTER TABLE jobs ADD COLUMN {col} {col_def};
                        END IF;
                    END $$;
                """)
        conn.commit()
    logger.info("jobs table ready")

def ensure_citations_table():
    """Ensure citations table exists."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    id SERIAL PRIMARY KEY,
                    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
                    chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
                    relevance_score FLOAT,
                    position INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS citations_message_idx ON citations(message_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS citations_chunk_idx ON citations(chunk_id);")
        conn.commit()
    logger.info("citations table ready")

def ensure_embedding_cache_table():
    """Ensure embedding cache table exists for Sprint 1 performance."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash VARCHAR(64) PRIMARY KEY,
                    embedding vector({Config.EMBEDDING_DIM}),
                    model VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    logger.info("embedding_cache table ready")

def ensure_analytics_columns():
    """Ensure analytics columns exist on the messages table.

    Adds response_latency_ms and token_count columns using
    ALTER TABLE ... ADD COLUMN IF NOT EXISTS so the migration
    is idempotent and safe to run on every startup.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'messages' AND column_name = 'response_latency_ms'
                    ) THEN
                        ALTER TABLE messages ADD COLUMN response_latency_ms INTEGER;
                    END IF;
                END $$;
            """)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'messages' AND column_name = 'token_count'
                    ) THEN
                        ALTER TABLE messages ADD COLUMN token_count INTEGER;
                    END IF;
                END $$;
            """)
        conn.commit()
    logger.info("analytics columns ready (messages.response_latency_ms, messages.token_count)")

def ensure_optimized_indexes():
    """Create optimized indexes for common query patterns (Sprint 5)."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS conv_project_updated
                ON conversations(project_id, updated_at DESC);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS msg_conv_created
                ON messages(conversation_id, created_at);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS jobs_queue_idx
                ON jobs(status, created_at) WHERE status = 'queued';
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS cit_msg_position
                ON citations(message_id, position);
            """)
        conn.commit()
    logger.info("optimized indexes ready (conv_project_updated, msg_conv_created, jobs_queue_idx, cit_msg_position)")


def bootstrap_schema():
    """Bootstrap database schema on startup."""
    logger.info("=" * 60)
    logger.info("BOOTSTRAPPING DATABASE SCHEMA")
    logger.info("=" * 60)

    try:
        # Order matters due to foreign keys
        ensure_pgvector()
        ensure_users_table()
        ensure_projects_table()
        ensure_documents_table()
        ensure_conversations_table()
        ensure_messages_table()
        ensure_chunks_table()
        ensure_jobs_table()
        ensure_citations_table()
        ensure_embedding_cache_table()
        ensure_analytics_columns()
        ensure_optimized_indexes()

        logger.info("=" * 60)
        logger.info("DATABASE READY")
        logger.info("=" * 60)
        return True
    except Exception as e:
        logger.error("=" * 60)
        logger.error("BOOTSTRAP FAILED: %s", e, exc_info=True)
        logger.error("=" * 60)
        return False

# ============================================================
# Bootstrap database and start background worker
# ============================================================

# Initialise connection pool and bootstrap database schema on startup
init_pool()
if not bootstrap_schema():
    logger.warning("Application may not function correctly without database!")

# Start background worker thread after DB is ready
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()

# Pydantic Models
class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserResponse(BaseModel):
    id: int
    email: str
    created_at: str

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ConversationCreate(BaseModel):
    project_id: int
    title: Optional[str] = "New Conversation"

class ChatRequest(BaseModel):
    conversation_id: int
    message: str

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str

class RenameConversation(BaseModel):
    title: str

class FeedbackRequest(BaseModel):
    feedback: str  # 'positive' or 'negative'
    comment: Optional[str] = None

# Health Check Endpoint
@app.get("/api/health", tags=["Health"], summary="Health check")
def health_check():
    """Health check endpoint for load balancers.

    Returns 200 if database is accessible, 503 otherwise.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )

# Authentication Endpoints
@app.post("/api/auth/register", response_model=AuthResponse, tags=["Auth"], summary="Register new user")
def register(request: RegisterRequest):
    """Register a new user account with email and password. Returns a JWT access token on success."""
    try:
        user = create_user(request.email, request.password)

        # Generate access token
        access_token = create_access_token(
            data={"user_id": user["id"], "email": user["email"]}
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "email": user["email"],
                "created_at": user["created_at"].isoformat() if user["created_at"] else None
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Registration failed: {str(e)}")

@app.post("/api/auth/login", response_model=AuthResponse, tags=["Auth"], summary="Login")
def login(request: LoginRequest):
    """Authenticate with email and password. Returns a JWT access token on success."""
    try:
        user = authenticate_user(request.email, request.password)

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        # Generate access token
        access_token = create_access_token(
            data={"user_id": user["id"], "email": user["email"]}
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "email": user["email"],
                "created_at": user["created_at"].isoformat() if user["created_at"] else None
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, f"Login failed: {str(e)}")

@app.get("/api/auth/me", response_model=UserResponse, tags=["Auth"], summary="Get current user")
def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Return profile information for the currently authenticated user."""
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "created_at": current_user["created_at"].isoformat() if current_user["created_at"] else None
    }

@app.post("/api/auth/refresh", response_model=AuthResponse, tags=["Auth"], summary="Refresh token")
def refresh_token(current_user: dict = Depends(get_current_user)):
    """Issue a new JWT access token for the authenticated user."""
    try:
        # Generate new access token
        access_token = create_access_token(
            data={"user_id": current_user["id"], "email": current_user["email"]}
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": current_user["id"],
                "email": current_user["email"],
                "created_at": current_user["created_at"].isoformat() if current_user["created_at"] else None
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Token refresh failed: {str(e)}")


@app.post("/api/auth/logout", tags=["Auth"], summary="Logout")
def logout():
    """Logout the user by clearing the auth cookie."""
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie(key="auth_token")
    return response


@app.post("/api/auth/change-password", tags=["Auth"], summary="Change password")
async def change_password(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Change the password for a password-based account. Requires the current password for verification. OAuth-only accounts cannot use this endpoint."""
    current_password = request.get("current_password")
    new_password = request.get("new_password")

    if not current_password or not new_password:
        raise HTTPException(status_code=400, detail="Current and new password required")

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    # Verify this is a password-based account
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash FROM users WHERE id = %s",
                (current_user["id"],)
            )
            row = cur.fetchone()

            if not row or not row[0]:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot change password for OAuth accounts"
                )

            # Verify current password
            if not verify_password(current_password, row[0]):
                raise HTTPException(status_code=401, detail="Current password is incorrect")

            # Hash and update new password
            new_hash = hash_password(new_password)
            cur.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (new_hash, current_user["id"])
            )
        conn.commit()

    return {"message": "Password changed successfully"}


# Google OAuth endpoints
@app.get("/api/auth/google/login", tags=["Auth"], summary="Google OAuth login")
def google_login():
    """Redirect the user to Google's OAuth consent screen to begin the login flow."""
    try:
        from core.oauth import build_google_auth_url
        url = build_google_auth_url()
        return RedirectResponse(url)
    except Exception as e:
        raise HTTPException(500, f"Failed to build Google auth URL: {e}")


@app.get("/api/auth/google/callback", tags=["Auth"], summary="Google OAuth callback")
def google_callback(code: Optional[str] = None):
    """OAuth callback: exchange the authorization code for tokens, verify the ID token, create or link the user account, and set a JWT cookie."""
    try:
        if not code:
            raise HTTPException(400, "Missing authorization code")

        from core.oauth import exchange_code_for_tokens, verify_id_token, create_or_get_user_from_google

        token_resp = exchange_code_for_tokens(code)
        id_token = token_resp.get("id_token")
        if not id_token:
            raise HTTPException(400, "ID token not returned from Google")

        id_info = verify_id_token(id_token)

        # Create or get user
        user = create_or_get_user_from_google(id_info)

        # Issue JWT
        access_token = create_access_token(data={"user_id": user["id"], "email": user["email"]})

        # Set cookie and redirect to frontend root
        redirect_url = "/"
        resp = RedirectResponse(redirect_url)
        # Cookie lifetime in seconds
        max_age = int(Config.JWT_EXPIRY_HOURS * 3600)
        resp.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            secure=Config.PROD,  # secure cookie only in production
            max_age=max_age,
            samesite="lax",
        )
        return resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Google OAuth callback failed: {e}")

# Project Endpoints
@app.get("/api/projects", tags=["Projects"], summary="List projects")
def list_projects(current_user: dict = Depends(get_current_user)):
    """Return all projects owned by the authenticated user, including document and conversation counts."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.id, p.name, p.description, p.created_at,
                       COUNT(DISTINCT d.id) as doc_count,
                       COUNT(DISTINCT c.id) as conv_count
                FROM projects p
                LEFT JOIN documents d ON d.project_id = p.id
                LEFT JOIN conversations c ON c.project_id = p.id
                WHERE p.user_id = %s
                GROUP BY p.id
                ORDER BY p.created_at DESC
            """, (current_user["id"],))
            rows = cur.fetchall()
    return [
        {
            "id": r[0], "name": r[1], "description": r[2],
            "created_at": r[3].isoformat() if r[3] else None,
            "document_count": r[4], "conversation_count": r[5]
        }
        for r in rows
    ]

@app.post("/api/projects", tags=["Projects"], summary="Create project")
def create_project(project: ProjectCreate, current_user: dict = Depends(get_current_user)):
    """Create a new project to organize documents and conversations."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO projects (name, description, user_id) VALUES (%s, %s, %s) RETURNING id",
                (project.name, project.description, current_user["id"])
            )
            project_id = cur.fetchone()[0]
        conn.commit()
    return {"id": project_id, "name": project.name}

@app.delete("/api/projects/{project_id}", tags=["Projects"], summary="Delete project")
def delete_project(project_id: int, current_user: dict = Depends(get_current_user)):
    """Delete a project and cascade-delete all its documents, conversations, and chunks."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify user owns the project
            cur.execute("SELECT user_id FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Project not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            cur.execute("DELETE FROM projects WHERE id = %s", (project_id,))
        conn.commit()
    return {"status": "deleted"}

# Document Endpoints
@app.get("/api/projects/{project_id}/documents", tags=["Documents"], summary="List documents")
def list_documents(
    project_id: int,
    search: Optional[str] = Query(None, description="Filter by filename (case-insensitive)"),
    status: Optional[str] = Query(None, description="Filter by document status"),
    sort: Optional[str] = Query("created_at_desc", description="Sort order: created_at_desc, created_at_asc, name_asc, name_desc"),
    current_user: dict = Depends(get_current_user)
):
    """List all documents in a project with optional search, status filter, and sorting."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify user owns the project
            cur.execute("SELECT user_id FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Project not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            # Build query with optional filters
            query = "SELECT id, filename, file_type, chunk_count, status, uploaded_at, error_message FROM documents WHERE project_id = %s"
            params = [project_id]

            if search:
                query += " AND filename ILIKE %s"
                params.append(f"%{search}%")

            if status:
                query += " AND status = %s"
                params.append(status)

            # Sort order
            sort_map = {
                "created_at_desc": "uploaded_at DESC",
                "created_at_asc": "uploaded_at ASC",
                "name_asc": "filename ASC",
                "name_desc": "filename DESC",
            }
            order_clause = sort_map.get(sort, "uploaded_at DESC")
            query += f" ORDER BY {order_clause}"

            cur.execute(query, params)
            rows = cur.fetchall()
    return [
        {
            "id": r[0], "filename": r[1], "file_type": r[2],
            "chunk_count": r[3], "status": r[4],
            "uploaded_at": r[5].isoformat() if r[5] else None,
            "error_message": r[6]
        }
        for r in rows
    ]

@app.post("/api/projects/{project_id}/upload", tags=["Documents"], summary="Upload documents")
async def upload_document(project_id: int, files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user)):
    """Upload one or more files and enqueue background jobs.

    Accepts multiple files via the 'files' field. Each file is saved to
    persistent storage and a document + job record is created for async
    processing by the background worker.

    Returns a list of created documents (or a single object for backward
    compatibility when only one file is uploaded).

    Supported formats: PDF, DOCX, DOC, PPTX, PPT, TXT, MD, CSV, JSON, HTML,
    and common code files (PY, JS, TS, etc.)
    """
    # Verify user owns the project
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Project not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

    results = []
    errors = []

    for file in files:
        content = await file.read()
        filename = file.filename

        # Get file extension
        file_ext = Path(filename).suffix.lower()
        file_type = file_ext.lstrip('.')

        # Validate file type
        if file_ext not in SUPPORTED_EXTENSIONS:
            errors.append({
                "filename": filename,
                "error": f"Unsupported file type: {file_ext}"
            })
            continue

        # Save file to persistent storage
        file_id = str(uuid.uuid4())
        file_path = os.path.join(Config.UPLOAD_DIR, f"{file_id}_{filename}")
        Path(file_path).write_bytes(content)

        # Create document and job records
        try:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO documents (project_id, filename, file_type, file_path, status)
                           VALUES (%s, %s, %s, %s, 'queued') RETURNING id""",
                        (project_id, filename, file_type, file_path)
                    )
                    doc_id = cur.fetchone()[0]

                    cur.execute(
                        "INSERT INTO jobs (document_id, status) VALUES (%s, 'queued') RETURNING id",
                        (doc_id,)
                    )
                    job_id = cur.fetchone()[0]
                conn.commit()

            results.append({
                "id": doc_id,
                "job_id": job_id,
                "filename": filename,
                "status": "queued"
            })

        except Exception as e:
            # Clean up file if DB insert fails
            if os.path.exists(file_path):
                os.remove(file_path)
            errors.append({
                "filename": filename,
                "error": str(e)
            })

    if not results and errors:
        raise HTTPException(400, f"All uploads failed: {errors}")

    # Backward compatibility: return single object if only one file uploaded
    if len(files) == 1 and len(results) == 1:
        return results[0]

    return {"documents": results, "errors": errors}

@app.delete("/api/documents/batch", tags=["Documents"], summary="Batch delete documents")
def batch_delete_documents(ids: str = Query(..., description="Comma-separated document IDs"), current_user: dict = Depends(get_current_user)):
    """Delete multiple documents at once, including their chunks, jobs, and uploaded files.

    All specified documents must belong to projects owned by the current user.
    Returns the count of successfully deleted documents.
    """
    try:
        doc_ids = [int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "Invalid document IDs")
    if not doc_ids:
        raise HTTPException(400, "No document IDs provided")

    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify all documents belong to the current user and collect file paths
            placeholders = ",".join(["%s"] * len(doc_ids))
            cur.execute(f"""
                SELECT d.id, d.file_path, p.user_id
                FROM documents d
                JOIN projects p ON p.id = d.project_id
                WHERE d.id IN ({placeholders})
            """, doc_ids)
            rows = cur.fetchall()

            if not rows:
                raise HTTPException(404, "No documents found")

            # Check ownership for all documents
            unauthorized = [r[0] for r in rows if r[2] != current_user["id"]]
            if unauthorized:
                raise HTTPException(403, f"Access denied for document(s): {unauthorized}")

            found_ids = [r[0] for r in rows]
            file_paths = [r[1] for r in rows if r[1]]

            # Delete all documents (CASCADE handles chunks, jobs, citations)
            cur.execute(f"""
                DELETE FROM documents WHERE id IN ({placeholders})
            """, found_ids)
            deleted_count = cur.rowcount
        conn.commit()

    # Clean up files (best-effort)
    for fp in file_paths:
        if fp and os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    logger.info("Batch deleted %d documents for user %d", deleted_count, current_user["id"])
    return {"deleted": deleted_count}

@app.delete("/api/documents/{doc_id}", tags=["Documents"], summary="Delete document")
def delete_document(doc_id: int, current_user: dict = Depends(get_current_user)):
    """Delete a document, its chunks, and the uploaded file from storage."""
    # Get file path and verify ownership
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.file_path, p.user_id
                FROM documents d
                JOIN projects p ON p.id = d.project_id
                WHERE d.id = %s
            """, (doc_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Document not found")
            file_path, owner_id = row
            if owner_id != current_user["id"]:
                raise HTTPException(403, "Access denied")

            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()

    # Clean up file
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass  # File cleanup is best-effort

    return {"status": "deleted"}

@app.get("/api/jobs/{job_id}", tags=["Documents"], summary="Get job status")
def get_job_status(job_id: int, current_user: dict = Depends(get_current_user)):
    """Get the current status of a document processing job, including timing and document details."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT j.id, j.document_id, j.status, j.error_message,
                       j.created_at, j.started_at, j.completed_at,
                       d.filename, d.status as doc_status, d.chunk_count,
                       p.user_id
                FROM jobs j
                JOIN documents d ON d.id = j.document_id
                JOIN projects p ON p.id = d.project_id
                WHERE j.id = %s
            """, (job_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(404, "Job not found")

    # Verify ownership
    if row[10] != current_user["id"]:
        raise HTTPException(403, "Access denied")

    return {
        "id": row[0],
        "document_id": row[1],
        "status": row[2],
        "error_message": row[3],
        "created_at": row[4].isoformat() if row[4] else None,
        "started_at": row[5].isoformat() if row[5] else None,
        "completed_at": row[6].isoformat() if row[6] else None,
        "document": {
            "filename": row[7],
            "status": row[8],
            "chunk_count": row[9]
        }
    }

@app.get("/api/jobs/{job_id}/progress", tags=["Documents"], summary="Get job progress")
def get_job_progress(job_id: int, current_user: dict = Depends(get_current_user)):
    """Get the processing progress for a job.

    Returns lightweight progress information suitable for polling:
    job_id, status, progress (0-100), retry_count, and max_retries.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT j.id, j.status, j.progress, j.retry_count, j.max_retries,
                       p.user_id
                FROM jobs j
                JOIN documents d ON d.id = j.document_id
                JOIN projects p ON p.id = d.project_id
                WHERE j.id = %s
            """, (job_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(404, "Job not found")

    # Verify ownership
    if row[5] != current_user["id"]:
        raise HTTPException(403, "Access denied")

    return {
        "job_id": row[0],
        "status": row[1],
        "progress": row[2] or 0,
        "retry_count": row[3] or 0,
        "max_retries": row[4] or 3,
    }

@app.post("/api/documents/{doc_id}/retry", tags=["Documents"], summary="Retry document processing")
def retry_document_processing(doc_id: int, current_user: dict = Depends(get_current_user)):
    """Retry processing for a document by resetting its status and creating a new background job."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Check document exists, has file_path, and verify ownership
            cur.execute("""
                SELECT d.file_path, p.user_id
                FROM documents d
                JOIN projects p ON p.id = d.project_id
                WHERE d.id = %s
            """, (doc_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Document not found")

            file_path, owner_id = row
            if owner_id != current_user["id"]:
                raise HTTPException(403, "Access denied")
            if not file_path:
                raise HTTPException(400, "Document has no file to process")

            # Reset document status
            cur.execute(
                "UPDATE documents SET status = 'queued', error_message = NULL WHERE id = %s",
                (doc_id,)
            )

            # Create new job with elevated priority (retries run before new uploads)
            cur.execute(
                "INSERT INTO jobs (document_id, status, priority) VALUES (%s, 'queued', 10) RETURNING id",
                (doc_id,)
            )
            job_id = cur.fetchone()[0]
        conn.commit()

    return {"status": "queued", "job_id": job_id}


@app.post("/api/documents/{doc_id}/reprocess", tags=["Documents"], summary="Reprocess failed document")
def reprocess_document(doc_id: int, current_user: dict = Depends(get_current_user)):
    """Re-process a failed document by deleting existing chunks, resetting its status, and creating a new job.

    Only documents with status 'error' or 'failed' can be reprocessed.
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            # Check document exists, verify ownership, and get status
            cur.execute("""
                SELECT d.file_path, d.status, p.user_id
                FROM documents d
                JOIN projects p ON p.id = d.project_id
                WHERE d.id = %s
            """, (doc_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Document not found")

            file_path, doc_status, owner_id = row
            if owner_id != current_user["id"]:
                raise HTTPException(403, "Access denied")
            if not file_path:
                raise HTTPException(400, "Document has no file to process")
            if doc_status not in ('error', 'failed'):
                raise HTTPException(400, f"Only failed documents can be reprocessed (current status: {doc_status})")

            # Delete existing chunks for a clean reprocess
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))

            # Reset document status
            cur.execute(
                "UPDATE documents SET status = 'processing', error_message = NULL, chunk_count = 0 WHERE id = %s",
                (doc_id,)
            )

            # Create new job with elevated priority (reprocesses run before new uploads)
            cur.execute(
                "INSERT INTO jobs (document_id, status, priority) VALUES (%s, 'queued', 10) RETURNING id",
                (doc_id,)
            )
            job_id = cur.fetchone()[0]
        conn.commit()

    return {"status": "queued", "job_id": job_id, "document_id": doc_id}


@app.get("/api/documents/{doc_id}/stats", tags=["Documents"], summary="Get document stats")
def get_document_stats(doc_id: int, current_user: dict = Depends(get_current_user)):
    """Get detailed statistics for a specific document including chunk count, file size, processing time, status, and page count."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify ownership and get document info
            cur.execute("""
                SELECT d.id, d.filename, d.file_type, d.file_path, d.chunk_count,
                       d.status, d.uploaded_at, d.error_message,
                       p.user_id
                FROM documents d
                JOIN projects p ON p.id = d.project_id
                WHERE d.id = %s
            """, (doc_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Document not found")
            if row[8] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            doc_id_val = row[0]
            filename = row[1]
            file_type = row[2]
            file_path = row[3]
            chunk_count = row[4] or 0
            status = row[5]
            uploaded_at = row[6]
            error_message = row[7]

            # Get file size
            file_size = 0
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)

            # Get page count from chunks (distinct page numbers)
            cur.execute("""
                SELECT COUNT(DISTINCT page_number)
                FROM chunks
                WHERE document_id = %s AND page_number IS NOT NULL
            """, (doc_id_val,))
            page_count = cur.fetchone()[0] or 0

            # Get processing time from the most recent completed job
            cur.execute("""
                SELECT started_at, completed_at
                FROM jobs
                WHERE document_id = %s AND status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
            """, (doc_id_val,))
            job_row = cur.fetchone()
            processing_time_seconds = None
            if job_row and job_row[0] and job_row[1]:
                delta = job_row[1] - job_row[0]
                processing_time_seconds = round(delta.total_seconds(), 2)

    return {
        "id": doc_id_val,
        "filename": filename,
        "file_type": file_type,
        "status": status,
        "chunk_count": chunk_count,
        "page_count": page_count,
        "file_size": file_size,
        "file_size_display": _format_file_size(file_size),
        "processing_time_seconds": processing_time_seconds,
        "uploaded_at": uploaded_at.isoformat() if uploaded_at else None,
        "error_message": error_message
    }


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.1f} {units[i]}"


# Conversation Endpoints
@app.get("/api/projects/{project_id}/conversations", tags=["Conversations"], summary="List conversations")
def list_conversations(project_id: int, current_user: dict = Depends(get_current_user)):
    """List all conversations in a project, ordered by most recently updated."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify user owns the project
            cur.execute("SELECT user_id FROM projects WHERE id = %s", (project_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Project not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            cur.execute("""
                SELECT id, title, created_at, updated_at
                FROM conversations WHERE project_id = %s
                ORDER BY updated_at DESC
            """, (project_id,))
            rows = cur.fetchall()
    return [
        {
            "id": r[0], "title": r[1],
            "created_at": r[2].isoformat() if r[2] else None,
            "updated_at": r[3].isoformat() if r[3] else None
        }
        for r in rows
    ]

@app.post("/api/conversations", tags=["Conversations"], summary="Create conversation")
def create_conversation(conv: ConversationCreate, current_user: dict = Depends(get_current_user)):
    """Create a new conversation within a project."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify user owns the project
            cur.execute("SELECT user_id FROM projects WHERE id = %s", (conv.project_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Project not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            cur.execute(
                "INSERT INTO conversations (project_id, title) VALUES (%s, %s) RETURNING id",
                (conv.project_id, conv.title)
            )
            conv_id = cur.fetchone()[0]
        conn.commit()
    return {"id": conv_id, "title": conv.title}

@app.delete("/api/conversations/{conv_id}", tags=["Conversations"], summary="Delete conversation")
def delete_conversation(conv_id: int, current_user: dict = Depends(get_current_user)):
    """Delete a conversation and all its messages."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify ownership
            cur.execute("""
                SELECT p.user_id
                FROM conversations c
                JOIN projects p ON p.id = c.project_id
                WHERE c.id = %s
            """, (conv_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Conversation not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            cur.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        conn.commit()
    return {"status": "deleted"}

@app.put("/api/conversations/{conv_id}", tags=["Conversations"], summary="Rename conversation")
def rename_conversation(conv_id: int, data: RenameConversation, current_user: dict = Depends(get_current_user)):
    """Update the title of an existing conversation."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify ownership
            cur.execute("""
                SELECT p.user_id
                FROM conversations c
                JOIN projects p ON p.id = c.project_id
                WHERE c.id = %s
            """, (conv_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Conversation not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            cur.execute(
                "UPDATE conversations SET title = %s WHERE id = %s",
                (data.title, conv_id)
            )
        conn.commit()
    return {"status": "updated", "title": data.title}

async def generate_conversation_title(message: str) -> str:
    """Use LLM to generate a 3-word title for the conversation"""
    provider, model = get_runtime_provider_model()
    llm = get_llm_client(provider, model)
    prompt = f"""Generate a short title (maximum 3 words) for a conversation that starts with this message.
Return ONLY the title, nothing else. No quotes, no punctuation at the end.

Message: {message}

Title:"""
    response = await llm.acomplete(prompt)
    title = response.text.strip().strip('"\'')
    # Ensure max 3 words
    words = title.split()[:3]
    return " ".join(words)

@app.get("/api/conversations/{conv_id}/messages", tags=["Conversations"], summary="Get messages")
def get_messages(conv_id: int, current_user: dict = Depends(get_current_user)) -> List[MessageResponse]:
    """Retrieve all messages in a conversation in chronological order, including feedback status."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify ownership
            cur.execute("""
                SELECT p.user_id
                FROM conversations c
                JOIN projects p ON p.id = c.project_id
                WHERE c.id = %s
            """, (conv_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Conversation not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            cur.execute(
                "SELECT id, role, content, feedback FROM messages WHERE conversation_id = %s ORDER BY created_at",
                (conv_id,)
            )
            rows = cur.fetchall()
    return [{"id": r[0], "role": r[1], "content": r[2], "feedback": r[3]} for r in rows]

@app.post("/api/messages/{message_id}/feedback", tags=["Conversations"], summary="Submit feedback")
def submit_feedback(message_id: int, request: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    """Submit thumbs-up or thumbs-down feedback for an assistant message, with an optional comment."""
    if request.feedback not in ('positive', 'negative'):
        raise HTTPException(400, "Feedback must be 'positive' or 'negative'")

    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify message belongs to user via conversation -> project -> user
            cur.execute("""
                SELECT p.user_id
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                JOIN projects p ON p.id = c.project_id
                WHERE m.id = %s
            """, (message_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Message not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            cur.execute(
                "UPDATE messages SET feedback = %s, feedback_comment = %s WHERE id = %s",
                (request.feedback, request.comment, message_id)
            )
        conn.commit()

    return {"status": "ok"}

@app.get("/api/messages/{message_id}/citations", tags=["Conversations"], summary="Get message citations")
def get_message_citations(message_id: int, current_user: dict = Depends(get_current_user)):
    """Get the source citations (retrieved chunks) that were used to generate a specific assistant message."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify message ownership through conversation -> project -> user
            cur.execute("""
                SELECT p.user_id
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                JOIN projects p ON p.id = c.project_id
                WHERE m.id = %s
            """, (message_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Message not found")
            if row[0] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            # Get citations with chunk and document details
            cur.execute("""
                SELECT
                    cit.id,
                    cit.chunk_id,
                    cit.relevance_score,
                    cit.position,
                    ch.content,
                    ch.chunk_index,
                    ch.document_id,
                    ch.page_number,
                    d.filename
                FROM citations cit
                JOIN chunks ch ON ch.id = cit.chunk_id
                JOIN documents d ON d.id = ch.document_id
                WHERE cit.message_id = %s
                ORDER BY cit.position
            """, (message_id,))
            rows = cur.fetchall()

    return [
        {
            "citation_id": r[0],
            "chunk_id": r[1],
            "relevance_score": float(r[2]) if r[2] else None,
            "position": r[3],
            "content": r[4],
            "chunk_index": r[5],
            "document_id": r[6],
            "page_number": r[7],
            "document_name": r[8]
        }
        for r in rows
    ]

@app.get("/api/chunks/{chunk_id}", tags=["Documents"], summary="Get chunk details")
def get_chunk_details(chunk_id: int, current_user: dict = Depends(get_current_user)):
    """Get the full content and metadata for a specific text chunk, including its source document."""
    with get_db() as conn:
        with conn.cursor() as cur:
            # Verify chunk ownership through project -> user
            cur.execute("""
                SELECT
                    ch.id,
                    ch.content,
                    ch.chunk_index,
                    ch.document_id,
                    ch.project_id,
                    ch.created_at,
                    ch.page_number,
                    d.filename,
                    d.file_type,
                    p.user_id
                FROM chunks ch
                JOIN documents d ON d.id = ch.document_id
                JOIN projects p ON p.id = ch.project_id
                WHERE ch.id = %s
            """, (chunk_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(404, "Chunk not found")
    if row[9] != current_user["id"]:
        raise HTTPException(403, "Access denied")

    # Format content for display
    from core.tasks import format_citation_text
    formatted_content = format_citation_text(row[1], max_length=2000)

    return {
        "id": row[0],
        "content": formatted_content,
        "chunk_index": row[2],
        "document_id": row[3],
        "project_id": row[4],
        "created_at": row[5].isoformat() if row[5] else None,
        "page_number": row[6],
        "document": {
            "filename": row[7],
            "file_type": row[8]
        }
    }

# RAG Chat - Import advanced retrieval
from core.retrieval import retrieve_context, advanced_retrieve, RetrievalConfig, preload_models, calculate_faithfulness_score


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses a fast heuristic (~4 chars per token for English).
    Accurate enough for context window management without requiring tiktoken.
    """
    if not text:
        return 0
    return len(text) // 4


async def summarize_conversation(messages: List[dict]) -> str:
    """Summarize older conversation messages to save context window space.

    Called when conversation history exceeds SUMMARIZE_AFTER_MESSAGES threshold.
    Keeps recent messages verbatim and summarizes older ones.
    """
    provider, model = get_runtime_provider_model()
    llm = get_llm_client(provider, model)

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in messages
    )

    prompt = f"""Summarize this conversation concisely, preserving key facts, questions asked, and answers given.
Focus on information that would be useful context for continuing the conversation.
Keep the summary under 300 words.

Conversation:
{history_text}

Summary:"""

    try:
        response = await llm.acomplete(prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning("Conversation summarization failed: %s", e)
        # Fallback: just keep last few messages
        return ""


def build_rag_prompt(query: str, context: List[dict], history: List[dict]) -> str:
    """Build prompt with context and history, respecting token limits."""
    context_text = ""
    if context:
        context_text = "\n\n---\nReference Documents (cite as [1], [2], etc.):\n"
        for i, c in enumerate(context, 1):
            context_text += f"\n[{i}] {c['filename']}:\n\"\"\"\n{c['content']}\n\"\"\"\n"

    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n"
        # Calculate available tokens for history
        base_tokens = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(context_text) + estimate_tokens(query) + 200
        available_for_history = Config.MAX_CONTEXT_TOKENS - base_tokens

        if available_for_history > 0:
            # Build history from most recent, stop when budget exhausted
            history_messages = []
            token_count = 0
            for msg in reversed(history[-20:]):  # Cap at last 20 messages
                msg_text = f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}\n"
                msg_tokens = estimate_tokens(msg_text)
                if token_count + msg_tokens > available_for_history:
                    break
                history_messages.insert(0, msg_text)
                token_count += msg_tokens

            history_text += "".join(history_messages)

    return f"""{SYSTEM_PROMPT}
{context_text}
{history_text}
User question: {query}"""

@app.post("/api/chat", tags=["Chat"], summary="Send chat message")
async def chat(request: ChatRequest = None, current_user: dict = Depends(get_current_user)):
    """Send a message and receive a streaming RAG response with citations.

    The response is a Server-Sent Events (SSE) stream that emits:
    - `data: {"content": "..."}` -- incremental LLM tokens
    - `event: citations` -- retrieved source chunks used for the answer
    - `event: faithfulness` -- optional faithfulness score
    - `event: message_id` -- saved message ID for feedback
    - `data: [DONE]` -- end of stream

    Context is retrieved via hybrid search (vector + BM25) with optional
    cross-encoder reranking. Conversation history is included automatically.
    """
    # Get conversation info and verify ownership
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.project_id, c.title, p.user_id
                FROM conversations c
                JOIN projects p ON p.id = c.project_id
                WHERE c.id = %s
            """, (request.conversation_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Conversation not found")
            project_id = row[0]
            current_title = row[1]
            owner_id = row[2]

            if owner_id != current_user["id"]:
                raise HTTPException(403, "Access denied")

            # Get message history
            cur.execute(
                "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at",
                (request.conversation_id,)
            )
            history = [{"role": r[0], "content": r[1]} for r in cur.fetchall()]

    # Summarize older messages if conversation is long
    if len(history) > Config.SUMMARIZE_AFTER_MESSAGES:
        # Keep last 6 messages verbatim, summarize the rest
        older_messages = history[:-6]
        recent_messages = history[-6:]
        summary = await summarize_conversation(older_messages)
        if summary:
            # Replace history with summary + recent messages
            history = [{"role": "assistant", "content": f"[Conversation summary: {summary}]"}] + recent_messages
        else:
            history = recent_messages

    # Check if this is the first message (auto-name the conversation)
    is_first_message = len(history) == 0
    new_title = None
    if is_first_message and current_title == "New Conversation":
        try:
            new_title = await generate_conversation_title(request.message)
        except Exception:
            new_title = None  # Fallback: keep default title

    # Save user message and optionally update title
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
                (request.conversation_id, request.message)
            )
            if new_title:
                cur.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP, title = %s WHERE id = %s",
                    (new_title, request.conversation_id)
                )
            else:
                cur.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (request.conversation_id,)
                )
        conn.commit()

    # Determine if we need new retrieval or if this is a follow-up query
    # Follow-up queries like "repeat that", "explain more", "go on" don't need new context
    follow_up_patterns = [
        "repeat", "again", "explain more", "go on", "continue",
        "what do you mean", "clarify", "elaborate", "tell me more",
        "say that again", "one more time", "rephrase"
    ]
    query_lower = request.message.lower().strip()
    is_follow_up = (
        len(request.message.split()) <= 6 and
        any(pattern in query_lower for pattern in follow_up_patterns)
    )

    # Retrieve context (skip for follow-up queries to avoid wrong citations)
    if is_follow_up and history:
        context = []  # Use conversation history only, no new retrieval
    else:
        context = retrieve_context(project_id, request.message)

    # Build prompt
    prompt = build_rag_prompt(request.message, context, history)

    # Collect full response for saving
    full_response = []

    async def generate():
        # Send new title if conversation was auto-named
        if new_title:
            yield f"data: {json.dumps({'type': 'title', 'title': new_title})}\n\n"

        provider, model = get_runtime_provider_model()
        llm = get_llm_client(provider, model)

        # Track response latency for analytics
        llm_start_time = time.time()

        response = await llm.astream_complete(prompt)
        async for chunk in response:
            if chunk.delta:
                full_response.append(chunk.delta)
                yield f"data: {json.dumps({'content': chunk.delta})}\n\n"

        # Calculate latency and estimated token count
        llm_end_time = time.time()
        response_latency_ms = int((llm_end_time - llm_start_time) * 1000)

        # Save assistant message
        complete_response = "".join(full_response)
        estimated_tokens = len(complete_response) // 4  # rough estimate: ~4 chars per token
        message_id = None
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO messages (conversation_id, role, content, response_latency_ms, token_count)
                       VALUES (%s, 'assistant', %s, %s, %s) RETURNING id""",
                    (request.conversation_id, complete_response, response_latency_ms, estimated_tokens)
                )
                message_id = cur.fetchone()[0]

                # Store citations for retrieved chunks
                for idx, ctx in enumerate(context, 1):
                    cur.execute(
                        """INSERT INTO citations (message_id, chunk_id, relevance_score, position)
                           VALUES (%s, %s, %s, %s)""",
                        (message_id, ctx["chunk_id"], ctx["similarity"], idx)
                    )
            conn.commit()

        # Send citations before [DONE]
        if context:
            from core.tasks import format_citation_text

            citations_data = []
            for idx, ctx in enumerate(context, 1):
                # Format preview with proper text cleaning
                preview = format_citation_text(ctx["content"], max_length=250)

                citations_data.append({
                    "citation_number": idx,
                    "chunk_id": ctx["chunk_id"],
                    "document_id": ctx["document_id"],
                    "document_name": ctx["filename"],
                    "relevance_score": round(ctx["similarity"], 3),
                    "page_number": ctx.get("page_number"),
                    "content_preview": preview
                })

            yield f"event: citations\ndata: {json.dumps({'citations': citations_data})}\n\n"

        # Calculate and send faithfulness score if enabled
        if Config.ENABLE_FAITHFULNESS_SCORING and context:
            try:
                openai_client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                faithfulness = await calculate_faithfulness_score(
                    complete_response,
                    context,
                    openai_client
                )
                yield f"event: faithfulness\ndata: {json.dumps(faithfulness)}\n\n"
            except Exception as e:
                logger.warning("Faithfulness scoring failed: %s", e)

        # Send message_id so frontend can attach feedback buttons
        if message_id:
            yield f"event: message_id\ndata: {json.dumps({'message_id': message_id})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================
# Conversation Export Endpoint
# ============================================================

@app.get(
    "/api/conversations/{conversation_id}/export",
    tags=["Conversations"],
    summary="Export conversation",
)
def export_conversation(
    conversation_id: int,
    format: str = Query("md", description="Export format: md, pdf, or json"),
    current_user: dict = Depends(get_current_user),
):
    """Export a conversation in Markdown, PDF, or JSON format.

    Returns a file download with the appropriate Content-Type and
    Content-Disposition headers.
    """
    from core.export import export_markdown, export_json, export_pdf

    if format not in ("md", "pdf", "json"):
        raise HTTPException(400, "Format must be md, pdf, or json")

    # Fetch conversation and verify ownership
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.title, c.created_at, p.user_id
                FROM conversations c
                JOIN projects p ON p.id = c.project_id
                WHERE c.id = %s
            """, (conversation_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Conversation not found")
            if row[3] != current_user["id"]:
                raise HTTPException(403, "Access denied")

            conv_id = row[0]
            conv_title = row[1] or "Untitled"
            conv_created = row[2].isoformat() if row[2] else ""

            # Fetch messages
            cur.execute(
                "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY created_at",
                (conv_id,)
            )
            messages = [{"role": r[0], "content": r[1]} for r in cur.fetchall()]

    conversation = {
        "id": conv_id,
        "title": conv_title,
        "created_at": conv_created,
        "messages": messages,
    }

    # Sanitize title for filename
    safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in conv_title).strip()[:50]
    if not safe_title:
        safe_title = f"conversation_{conv_id}"

    if format == "md":
        content = export_markdown(conversation)
        return StreamingResponse(
            iter([content]),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.md"'},
        )
    elif format == "json":
        content = export_json(conversation)
        return StreamingResponse(
            iter([content]),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.json"'},
        )
    elif format == "pdf":
        pdf_bytes = export_pdf(conversation)
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.pdf"'},
        )


# ============================================================
# Multi-Model Selection Endpoints
# ============================================================

class ModelSelectRequest(BaseModel):
    provider: str
    model: str

@app.get("/api/models", tags=["Models"], summary="List available models")
def get_models(current_user: dict = Depends(get_current_user)):
    """Return available LLM providers and their models, along with the currently active provider and model."""
    provider, model = get_runtime_provider_model()
    available_providers = Config.get_available_providers()

    providers_info = {}
    for p in available_providers:
        providers_info[p] = {
            "available": True,
            "models": Config.AVAILABLE_MODELS.get(p, []),
            "default_model": Config.DEFAULT_MODELS.get(p, ""),
        }

    # Also include providers that are known but not configured
    for p in ["openai", "anthropic", "google", "ollama"]:
        if p not in providers_info:
            providers_info[p] = {
                "available": False,
                "models": Config.AVAILABLE_MODELS.get(p, []),
                "default_model": Config.DEFAULT_MODELS.get(p, ""),
            }

    return {
        "current_provider": provider,
        "current_model": model,
        "providers": providers_info,
    }

@app.post("/api/models/select", tags=["Models"], summary="Switch active model")
def select_model(request: ModelSelectRequest, current_user: dict = Depends(get_current_user)):
    """Switch the active LLM provider and model at runtime. Validates that the provider is configured and the model is in the allowed list."""
    global _runtime_provider, _runtime_model

    provider = request.provider.lower()
    model = request.model

    # Validate provider
    if provider not in ["openai", "anthropic", "google", "ollama"]:
        raise HTTPException(400, f"Unknown provider: {provider}")

    # Validate the provider is available
    available = Config.get_available_providers()
    if provider not in available:
        raise HTTPException(400, f"Provider '{provider}' is not configured (missing API key)")

    # Validate model is in the allowed list
    allowed_models = Config.AVAILABLE_MODELS.get(provider, [])
    if model not in allowed_models:
        raise HTTPException(400, f"Model '{model}' not available for provider '{provider}'. Available: {allowed_models}")

    # Test that we can create the client
    try:
        get_llm_client(provider, model)
    except ValueError as e:
        raise HTTPException(400, str(e))

    _runtime_provider = provider
    _runtime_model = model

    return {"status": "ok", "provider": provider, "model": model}

# ============================================================
# Analytics Dashboard Endpoint
# ============================================================

@app.get("/api/analytics", tags=["Analytics"], summary="Get usage analytics")
def get_analytics(current_user: dict = Depends(get_current_user)):
    """Return analytics dashboard data for the current user.

    Includes summary counts (documents, chunks, queries, latency, conversations),
    queries-per-day over the last 30 days, document type distribution,
    top-cited documents, and processing job statistics. All data is scoped to the
    authenticated user's projects.
    """
    user_id = current_user["id"]

    with get_db() as conn:
        with conn.cursor() as cur:
            # --- Summary counts ---
            # Total documents across user's projects
            cur.execute("""
                SELECT COUNT(*)
                FROM documents d
                JOIN projects p ON p.id = d.project_id
                WHERE p.user_id = %s
            """, (user_id,))
            total_documents = cur.fetchone()[0] or 0

            # Total chunks
            cur.execute("""
                SELECT COUNT(*)
                FROM chunks ch
                JOIN projects p ON p.id = ch.project_id
                WHERE p.user_id = %s
            """, (user_id,))
            total_chunks = cur.fetchone()[0] or 0

            # Total queries (user messages)
            cur.execute("""
                SELECT COUNT(*)
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                JOIN projects p ON p.id = c.project_id
                WHERE p.user_id = %s AND m.role = 'user'
            """, (user_id,))
            total_queries = cur.fetchone()[0] or 0

            # Average response latency (only for messages that have it)
            cur.execute("""
                SELECT AVG(m.response_latency_ms)
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                JOIN projects p ON p.id = c.project_id
                WHERE p.user_id = %s
                  AND m.role = 'assistant'
                  AND m.response_latency_ms IS NOT NULL
            """, (user_id,))
            avg_latency_row = cur.fetchone()[0]
            avg_latency_ms = int(avg_latency_row) if avg_latency_row else 0

            # Total conversations
            cur.execute("""
                SELECT COUNT(*)
                FROM conversations c
                JOIN projects p ON p.id = c.project_id
                WHERE p.user_id = %s
            """, (user_id,))
            total_conversations = cur.fetchone()[0] or 0

            # --- Queries per day (last 30 days) ---
            cur.execute("""
                SELECT DATE(m.created_at) as day, COUNT(*) as cnt
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                JOIN projects p ON p.id = c.project_id
                WHERE p.user_id = %s
                  AND m.role = 'user'
                  AND m.created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(m.created_at)
                ORDER BY day
            """, (user_id,))
            queries_per_day = [
                {"date": row[0].isoformat(), "count": row[1]}
                for row in cur.fetchall()
            ]

            # --- Document types distribution ---
            cur.execute("""
                SELECT COALESCE(d.file_type, 'unknown') as ftype, COUNT(*) as cnt
                FROM documents d
                JOIN projects p ON p.id = d.project_id
                WHERE p.user_id = %s
                GROUP BY ftype
                ORDER BY cnt DESC
            """, (user_id,))
            document_types = [
                {"type": row[0], "count": row[1]}
                for row in cur.fetchall()
            ]

            # --- Top cited documents ---
            cur.execute("""
                SELECT d.id, d.filename, COUNT(cit.id) as citation_count
                FROM citations cit
                JOIN chunks ch ON ch.id = cit.chunk_id
                JOIN documents d ON d.id = ch.document_id
                JOIN projects p ON p.id = d.project_id
                WHERE p.user_id = %s
                GROUP BY d.id, d.filename
                ORDER BY citation_count DESC
                LIMIT 10
            """, (user_id,))
            top_cited_documents = [
                {"document_id": row[0], "filename": row[1], "citation_count": row[2]}
                for row in cur.fetchall()
            ]

            # --- Processing stats (job statuses) ---
            cur.execute("""
                SELECT j.status, COUNT(*) as cnt
                FROM jobs j
                JOIN documents d ON d.id = j.document_id
                JOIN projects p ON p.id = d.project_id
                WHERE p.user_id = %s
                GROUP BY j.status
            """, (user_id,))
            job_status_rows = {row[0]: row[1] for row in cur.fetchall()}
            processing_stats = {
                "completed": job_status_rows.get("completed", 0),
                "failed": job_status_rows.get("failed", 0),
                "processing": job_status_rows.get("processing", 0),
                "queued": job_status_rows.get("queued", 0),
                "dead_letter": job_status_rows.get("dead_letter", 0)
            }

    return {
        "summary": {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_queries": total_queries,
            "avg_latency_ms": avg_latency_ms,
            "total_conversations": total_conversations
        },
        "queries_per_day": queries_per_day,
        "document_types": document_types,
        "top_cited_documents": top_cited_documents,
        "processing_stats": processing_stats
    }

# Static files
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Startup event (bootstrap already ran at module load)
@app.on_event("startup")
async def startup():
    # Preload ML models to avoid cold-start latency on first request
    preload_models()
    logger.info("FastAPI application startup complete")


@app.on_event("shutdown")
async def shutdown():
    close_pool()
    logger.info("FastAPI application shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)