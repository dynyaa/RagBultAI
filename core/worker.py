"""Background worker for processing document jobs.

This worker polls the jobs table for queued jobs and processes them
asynchronously. Run this as a separate process:

    python worker.py

The worker handles chunking, embedding, and storing documents without
blocking the API server. Multiple workers can run for parallel processing.
"""
import logging
import os
import time
import sys

# Allow running directly: python core/worker.py
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "core"

from .logging_config import setup_logging

# Configure logging before anything else
setup_logging()

logger = logging.getLogger("rag.worker")

from .tasks import process_document
from .config import Config
from .db import get_db, init_pool

# Validate config before starting
Config.validate()

# Initialise the connection pool for standalone worker mode.
# (When running inside app.py, the pool is already initialised.)
init_pool()

# Auto-run database migrations on startup
try:
    from migrate import run_migration
    logger.info("Checking for database migrations...")
    run_migration(silent=True)
    logger.info("Database migrations applied successfully")
except Exception as e:
    logger.warning("Migration check failed (this is OK if DB is already up to date): %s", e)


def process_next_job():
    """Poll for and process the next queued job.

    Returns True if a job was processed, False if none available.
    """
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
                return False  # No jobs available

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

    # Process the document (outside transaction to avoid long locks)
    logger.info("Processing job %d (document %d)...", job_id, document_id)

    try:
        process_document(document_id, job_id)
        logger.info("Job %d completed successfully", job_id)
        return True

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

        return True


def main():
    """Main worker loop - polls for jobs continuously."""
    logger.info("Document processing worker started")
    logger.info("Polling database: %s", Config.PG_CONN.split('@')[-1])
    logger.info("Press Ctrl+C to stop")

    poll_interval = 2  # seconds

    try:
        while True:
            try:
                processed = process_next_job()

                if processed:
                    # Job was processed, check immediately for next job
                    continue
                else:
                    # No jobs available, wait before polling again
                    time.sleep(poll_interval)

            except KeyboardInterrupt:
                raise  # Let outer handler catch it
            except Exception as e:
                logger.error("Worker error: %s", e)
                time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Worker stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
