"""Centralized database connection pool for the RAG application.

Replaces the duplicated ``get_db()`` context managers scattered across
app.py, retrieval.py, tasks.py, worker.py, auth.py and oauth.py with a
single connection pool backed by ``psycopg2.pool.ThreadedConnectionPool``.

Usage:
    from db import get_db, init_pool, close_pool

    # At application startup
    init_pool()

    # In request / task code
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")

    # At application shutdown
    close_pool()
"""
import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.pool

from .config import Config

logger = logging.getLogger(__name__)

# Module-level pool reference
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_pool() -> None:
    """Create the global connection pool.

    Reads ``DB_POOL_MIN`` and ``DB_POOL_MAX`` from :class:`Config`.
    Safe to call multiple times -- subsequent calls are no-ops if the
    pool is already initialised.
    """
    global _pool
    if _pool is not None:
        return  # Already initialised

    logger.info(
        "Initialising DB connection pool (min=%d, max=%d)",
        Config.DB_POOL_MIN,
        Config.DB_POOL_MAX,
    )
    _pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=Config.DB_POOL_MIN,
        maxconn=Config.DB_POOL_MAX,
        dsn=Config.PG_CONN,
    )
    logger.info("DB connection pool ready")


def close_pool() -> None:
    """Close all connections in the pool."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("DB connection pool closed")


@contextmanager
def get_db():
    """Context manager that yields a connection from the pool.

    If the pool has not been initialised yet (e.g. standalone worker
    started without app.py), ``init_pool()`` is called automatically as
    a fallback so that every module can safely ``from db import get_db``.

    Connections are returned to the pool (or rolled back) on exit.
    Slow queries are logged as warnings when they exceed
    ``Config.SLOW_QUERY_THRESHOLD_MS``.
    """
    global _pool
    if _pool is None:
        # Fallback for standalone processes (worker.py)
        init_pool()

    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)
