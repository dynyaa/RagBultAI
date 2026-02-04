"""Document processing tasks for RAG system.

This module handles:
- Loading documents from various formats (PDF, DOCX, TXT, etc.)
- Automatic OCR for image-based PDFs
- Text chunking with configurable overlap
- Embedding generation via OpenAI
- Storage in PostgreSQL with pgvector

Tasks are executed by background worker threads/processes to prevent
HTTP timeouts on large documents.
"""
import logging
import os
import hashlib
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PyMuPDFReader

from .config import Config
from .db import get_db

logger = logging.getLogger("rag.tasks")

# Semantic chunking (optional - lazy loaded)
_semantic_splitter = None


def sanitize_text(text: str) -> str:
    """Remove null bytes and other problematic characters from text.

    PostgreSQL text columns cannot contain NUL (0x00) characters.
    Some PDFs contain these from binary data or corrupted extraction.
    """
    if not text:
        return text
    # Remove null bytes
    text = text.replace('\x00', '')
    # Remove other control characters (except newline, tab, carriage return)
    text = ''.join(char for char in text if char == '\n' or char == '\t' or char == '\r' or ord(char) >= 32)
    return text


def clean_pdf_artifacts(text: str) -> str:
    """Clean common PDF extraction artifacts for better chunk quality.

    Removes:
    - Table of contents dot leaders (............)
    - Standalone page numbers
    - Excessive whitespace
    - Repeated headers/footers patterns
    - TOC-style entries
    """
    import re

    if not text:
        return text

    # Remove dot leaders (table of contents pattern: "Chapter 1 ........... 5")
    text = re.sub(r'\.{3,}', ' ', text)

    # Remove TOC-style entries (text followed by dots and page number)
    # Matches patterns like "Introduction .................... 5"
    text = re.sub(r'^[^\n]+\s*\.{2,}\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # Remove lines that are just page numbers (standalone numbers 1-999)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # Remove page number patterns at end of lines ("... 15" or "...15")
    text = re.sub(r'\s+\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # Remove section numbers alone on a line (like "2.1", "5.3")
    text = re.sub(r'^\s*\d+\.\d+\s*$', '', text, flags=re.MULTILINE)

    # Join lines that were split mid-sentence (line ends without punctuation,
    # next line starts with lowercase)
    text = re.sub(r'([a-zа-яё,])\n([a-zа-яё])', r'\1 \2', text)

    # Collapse multiple consecutive newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces to single space
    text = re.sub(r' {2,}', ' ', text)

    # Remove lines that are just whitespace
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)

    # Remove repeated markers (confidential, headers, etc.)
    lines = text.split('\n')
    seen_markers = set()
    cleaned_lines = []

    for line in lines:
        line_lower = line.lower().strip()
        # Check for common repeated markers
        if line_lower in seen_markers:
            continue
        if re.match(r'^(confidential|confidentially|конфиденциально)$', line_lower):
            if line_lower not in seen_markers:
                seen_markers.add(line_lower)
                cleaned_lines.append(line)
        elif len(line_lower) < 30 and line_lower:
            # Short lines might be headers - track them
            if line_lower not in seen_markers:
                seen_markers.add(line_lower)
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def format_citation_text(text: str, max_length: int = 500) -> str:
    """Format text for citation display with better readability.

    Converts raw chunk text into clean, readable citation preview.
    """
    import re

    if not text:
        return text

    # First apply standard cleaning
    text = clean_pdf_artifacts(text)

    # Convert multiple newlines to paragraph breaks
    text = re.sub(r'\n\n+', '\n\n', text)

    # Convert single newlines to spaces (join paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Clean up any resulting double spaces
    text = re.sub(r' +', ' ', text)

    # Truncate if too long
    if len(text) > max_length:
        # Try to break at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')
        break_point = max(last_period, last_question, last_exclaim)

        if break_point > max_length * 0.5:  # At least half the content
            text = truncated[:break_point + 1]
        else:
            text = truncated.rsplit(' ', 1)[0] + '...'

    return text.strip()

# OCR dependencies (optional - gracefully degrades if not available)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configure LlamaIndex with OpenAI models
Settings.llm = OpenAI(model=Config.LLM_MODEL, api_key=Config.OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(
    model=Config.EMBEDDING_MODEL,
    api_key=Config.OPENAI_API_KEY
)
Settings.chunk_size = Config.CHUNK_SIZE
Settings.chunk_overlap = Config.CHUNK_OVERLAP


def report_progress(job_id: int, pct: int) -> None:
    """Update the progress percentage (0-100) for a job.

    Called at various stages of document processing to provide
    real-time progress feedback via the /api/jobs/{id}/progress endpoint.
    """
    pct = max(0, min(100, pct))  # Clamp to 0-100
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET progress = %s WHERE id = %s",
                    (pct, job_id)
                )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to report progress for job %d: %s", job_id, e)


def get_semantic_splitter():
    """Lazy load the semantic splitter for better chunk boundaries.

    Semantic chunking analyzes embedding similarity between sentences
    to find natural breakpoints, producing more coherent chunks.
    """
    global _semantic_splitter
    if _semantic_splitter is None:
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            logger.info("Initializing semantic splitter...")
            _semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=Config.SEMANTIC_BREAKPOINT_THRESHOLD,
                embed_model=Settings.embed_model
            )
            logger.info("Semantic splitter ready")
        except ImportError as e:
            logger.warning("Semantic chunking unavailable: %s", e)
            return None
        except Exception as e:
            logger.warning("Failed to initialize semantic splitter: %s", e)
            return None
    return _semantic_splitter


def get_splitter():
    """Get the appropriate text splitter based on configuration.

    Returns semantic splitter if enabled and available,
    otherwise returns the default SentenceSplitter.
    """
    if Config.USE_SEMANTIC_CHUNKING:
        semantic = get_semantic_splitter()
        if semantic is not None:
            return semantic
        logger.warning("Falling back to standard chunking")

    return SentenceSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )


def get_content_hash(text: str) -> str:
    """Generate SHA-256 hash of text content for cache lookup."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_cached_embedding(content_hash: str, conn) -> list | None:
    """Look up an embedding by content hash. Returns None if not cached."""
    if not Config.USE_EMBEDDING_CACHE:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM embedding_cache WHERE content_hash = %s",
                (content_hash,)
            )
            row = cur.fetchone()
            if row and row[0]:
                # pgvector returns a string representation; parse it
                emb = row[0]
                if isinstance(emb, str):
                    emb = [float(x) for x in emb.strip('[]').split(',')]
                return emb
    except Exception:
        pass  # Table may not exist yet
    return None


def store_cached_embedding(content_hash: str, embedding: list, conn):
    """Store an embedding in the cache. Idempotent via ON CONFLICT."""
    if not Config.USE_EMBEDDING_CACHE:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO embedding_cache (content_hash, embedding, model)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (content_hash) DO NOTHING""",
                (content_hash, embedding, Config.EMBEDDING_MODEL)
            )
    except Exception:
        pass  # Table may not exist yet; non-critical


def ocr_pdf_to_text(pdf_path: str) -> list:
    """Extract text from image-based PDF using OCR.

    Converts each PDF page to an image and runs Tesseract OCR with
    Russian + English language support. Useful for scanned documents,
    PowerPoint slides exported as PDFs, and other image-based PDFs.

    Args:
        pdf_path: Absolute path to PDF file

    Returns:
        List of Document objects with OCR'd text and metadata

    Raises:
        RuntimeError: If OCR dependencies not installed or conversion fails
    """
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "OCR dependencies not installed. "
            "Install: pip install pdf2image pytesseract"
        )

    logger.info("PDF contains no text layer - running OCR...")

    # Convert PDF pages to images using poppler
    # On Windows, may need POPPLER_PATH environment variable
    poppler_path = os.getenv("POPPLER_PATH")

    logger.info("Converting PDF to images...")
    try:
        if poppler_path:
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
        else:
            images = convert_from_path(pdf_path)
        logger.info("Converted PDF to %d images", len(images))
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert PDF to images: {e}\n"
            "Make sure poppler-utils is installed:\n"
            "- Linux: apt-get install poppler-utils\n"
            "- macOS: brew install poppler\n"
            "- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/"
        )

    # Run OCR on each page
    documents = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang=Config.OCR_LANGUAGES)

        if text.strip():
            doc = Document(
                text=text,
                metadata={
                    "page": i + 1,
                    "source": "ocr"
                }
            )
            documents.append(doc)
            logger.info("OCR page %d: %d chars", i + 1, len(text))
        else:
            logger.warning("OCR page %d: no text found", i + 1)

    return documents


# ---------- Sprint 3: Multi-Modal Document Understanding ----------

def extract_tables_from_pdf(pdf_path: str) -> list:
    """Extract tables from a PDF using pdfplumber and convert to markdown.

    Each table is returned as a dict with page number, table index on that
    page, and a markdown representation of the table content.  The function
    degrades gracefully when pdfplumber is not installed.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of dicts: [{"page": int, "table_markdown": str, "table_index": int}]
        Empty list if pdfplumber is unavailable or no tables are found.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed - skipping table extraction. "
                       "Install: pip install pdfplumber")
        return []

    tables_found = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                if not page_tables:
                    continue
                for tbl_idx, table in enumerate(page_tables):
                    if not table:
                        continue
                    # Build markdown table from rows
                    md_lines = []
                    for row_idx, row in enumerate(table):
                        # Replace None cells with empty string
                        cells = [str(cell).strip() if cell is not None else "" for cell in row]
                        md_lines.append("| " + " | ".join(cells) + " |")
                        # Add header separator after first row
                        if row_idx == 0:
                            md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                    table_md = "\n".join(md_lines)
                    if table_md.strip():
                        tables_found.append({
                            "page": page_num,
                            "table_markdown": table_md,
                            "table_index": tbl_idx,
                        })
        if tables_found:
            logger.info("Extracted %d tables from PDF", len(tables_found))
        else:
            logger.info("No tables found in PDF")
    except Exception as e:
        logger.warning("Table extraction failed: %s", e)

    return tables_found


def describe_pdf_images(pdf_path: str, pages: list = None) -> list:
    """Extract images from PDF pages and describe them via OpenAI vision API.

    Uses PyMuPDF (fitz) to pull images from every page (or a subset) and
    sends images larger than 10 KB to the gpt-4o-mini vision model for a
    textual description.  The result is a list of dicts suitable for creating
    LlamaIndex Document objects.

    This function is gated behind Config.EXTRACT_IMAGES (default: false)
    because each image incurs an OpenAI vision API call.

    Args:
        pdf_path: Absolute path to the PDF file.
        pages:    Optional list of 1-based page numbers to process.
                  If None, all pages are processed.

    Returns:
        List of dicts: [{"page": int, "image_index": int, "description": str}]
        Empty list on any failure.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF (fitz) not available - skipping image extraction")
        return []

    import base64

    descriptions = []
    try:
        doc = fitz.open(pdf_path)
        client = _get_context_client()  # reuse cached OpenAI client

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            if pages is not None and page_num not in pages:
                continue

            image_list = doc[page_idx].get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue
                image_bytes = base_image["image"]

                # Skip small images (likely icons/logos, < 10 KB)
                if len(image_bytes) < 10_240:
                    continue

                # Encode to base64 for the vision API
                b64_image = base64.b64encode(image_bytes).decode("utf-8")
                mime = base_image.get("ext", "png")
                if mime == "jpg":
                    mime = "jpeg"
                data_uri = f"data:image/{mime};base64,{b64_image}"

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "Describe this image from a document in detail. "
                                            "If it is a chart or graph, describe the data it "
                                            "shows, axes, trends, and key takeaways. "
                                            "If it is a diagram, describe its components and "
                                            "relationships. Keep the description concise but "
                                            "informative (max 200 words)."
                                        ),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_uri},
                                    },
                                ],
                            }
                        ],
                        max_tokens=300,
                        temperature=0,
                    )
                    desc = response.choices[0].message.content.strip()
                    if desc:
                        descriptions.append({
                            "page": page_num,
                            "image_index": img_idx,
                            "description": desc,
                        })
                        logger.info("Described image on page %d (idx %d)", page_num, img_idx)
                except Exception as e:
                    logger.warning("Vision API call failed for page %d image %d: %s", page_num, img_idx, e)

        doc.close()

        if descriptions:
            logger.info("Described %d images from PDF", len(descriptions))
        else:
            logger.info("No significant images found in PDF")
    except Exception as e:
        logger.warning("Image extraction failed: %s", e)

    return descriptions


_context_client = None

def _get_context_client():
    """Get or create a cached OpenAI client for contextual chunking."""
    global _context_client
    if _context_client is None:
        from openai import OpenAI as OpenAIClient
        _context_client = OpenAIClient(api_key=Config.OPENAI_API_KEY)
    return _context_client


def generate_chunk_context(chunk_text: str, document_text: str, filename: str) -> str:
    """Generate contextual prefix for a chunk using LLM (Anthropic method).

    Prepends a 1-2 sentence context that explains where this chunk fits
    within the overall document, improving retrieval accuracy by ~49%.
    """
    if not Config.USE_CONTEXTUAL_CHUNKING:
        return chunk_text

    # Truncate document text to fit in context window (keep first ~6000 chars)
    doc_preview = document_text[:6000]

    prompt = f"""<document>
{doc_preview}
</document>

Here is a chunk from the document "{filename}":
<chunk>
{chunk_text[:1500]}
</chunk>

Give a short succinct context (1-2 sentences) to situate this chunk within the overall document.
Focus on what section/topic this chunk covers and how it relates to the document's main subject.
Answer ONLY with the context, nothing else."""

    try:
        client = _get_context_client()
        response = client.chat.completions.create(
            model=Config.CONTEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )
        context = response.choices[0].message.content.strip()
        return f"{context}\n\n{chunk_text}"
    except Exception as e:
        logger.warning("Contextual chunking failed for chunk: %s", e)
        return chunk_text  # Fallback to raw text


def process_document(document_id: int, job_id: int):
    """Process uploaded document: load, chunk, embed, and store.

    Main document processing pipeline:
    1. Load document from disk using appropriate reader
    2. For PDFs: Auto-detect text layer, use OCR if needed
    3. Chunk text into segments with overlap
    4. Generate embeddings for each chunk
    5. Store chunks and embeddings in database
    6. Update job and document status

    Args:
        document_id: Database ID of document record
        job_id: Database ID of job record to track progress

    Raises:
        ValueError: If document not found or processing fails
        RuntimeError: If OCR needed but not available
    """
    # Fetch document metadata from database
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT project_id, filename, file_path FROM documents WHERE id = %s",
                (document_id,)
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Document {document_id} not found")
            project_id, filename, file_path = row

    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    report_progress(job_id, 5)

    # Load document using appropriate reader
    if file_path.lower().endswith('.pdf'):
        # Use PyMuPDF for PDFs (fast, handles text extraction well)
        reader = PyMuPDFReader()
        documents = reader.load(file_path=file_path)

        # Check if PDF has text layer or is image-based
        total_text = "".join(doc.text for doc in documents if doc.text)
        if not total_text.strip():
            # No text found - PDF is likely scanned or slide deck
            logger.info("PDF has no text layer, attempting OCR...")
            if OCR_AVAILABLE:
                documents = ocr_pdf_to_text(file_path)
            else:
                raise ValueError(
                    "PDF contains no text (likely scanned/image-based). "
                    "OCR dependencies not available. "
                    "Install: pip install pdf2image pytesseract"
                )
    else:
        # Use SimpleDirectoryReader for all other formats
        # Supports 100+ file types: DOCX, PPTX, TXT, MD, CSV, JSON, etc.
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

    if not documents:
        raise ValueError("No content could be extracted from the file")

    report_progress(job_id, 10)  # Document loaded

    # --- Sprint 3: Multi-Modal extraction (PDF only) ---
    is_pdf = file_path.lower().endswith('.pdf')

    # Table extraction: convert PDF tables to markdown Document objects
    table_documents = []
    if is_pdf and Config.EXTRACT_TABLES:
        logger.info("Extracting tables from PDF...")
        tables = extract_tables_from_pdf(file_path)
        for tbl in tables:
            table_doc = Document(
                text=tbl["table_markdown"],
                metadata={
                    "source": "table",
                    "page": tbl["page"],
                    "table_index": tbl["table_index"],
                    "content_type": "table",
                },
            )
            table_documents.append(table_doc)
        if table_documents:
            logger.info("Created %d table documents", len(table_documents))

    # Image/chart description: describe significant images via vision API
    image_documents = []
    if is_pdf and Config.EXTRACT_IMAGES:
        logger.info("Extracting and describing images from PDF...")
        image_descs = describe_pdf_images(pdf_path=file_path)
        for desc in image_descs:
            image_doc = Document(
                text=desc["description"],
                metadata={
                    "source": "image",
                    "page": desc["page"],
                    "image_index": desc["image_index"],
                    "content_type": "image",
                },
            )
            image_documents.append(image_doc)
        if image_documents:
            logger.info("Created %d image-description documents", len(image_documents))

    # Add metadata to documents
    for doc in documents:
        doc.metadata["filename"] = filename
        doc.metadata["doc_id"] = document_id
        # Mark original text documents
        if "content_type" not in doc.metadata:
            doc.metadata["content_type"] = "text"

    # Merge multi-modal documents into the main list so they go through
    # the same chunking -> embedding -> storage pipeline
    for extra_doc in table_documents + image_documents:
        extra_doc.metadata["filename"] = filename
        extra_doc.metadata["doc_id"] = document_id
    documents.extend(table_documents)
    documents.extend(image_documents)

    report_progress(job_id, 25)  # Extraction complete

    # Split documents into chunks
    # Uses semantic chunking if enabled, otherwise standard sentence splitting
    splitter = get_splitter()
    chunking_type = "semantic" if Config.USE_SEMANTIC_CHUNKING and get_semantic_splitter() else "sentence"
    logger.info("Chunking with %s splitter...", chunking_type)
    nodes = splitter.get_nodes_from_documents(documents)

    if not nodes:
        raise ValueError("No text chunks could be created from the document")

    report_progress(job_id, 40)  # Chunking complete

    # Build full document text for contextual chunking
    full_doc_text = "\n".join(doc.text for doc in documents if doc.text)

    # Generate embeddings and store chunks
    total_nodes = len(nodes)
    with get_db() as conn:
        with conn.cursor() as cur:
            valid_chunk_count = 0

            for i, node in enumerate(nodes):
                # Skip empty or whitespace-only chunks
                if not node.text or not node.text.strip():
                    logger.warning("Skipping empty chunk %d", i)
                    continue

                # Sanitize and clean chunk text
                chunk_text = sanitize_text(node.text)
                chunk_text = clean_pdf_artifacts(chunk_text)

                # Skip if cleaning left empty text
                if not chunk_text.strip():
                    logger.warning("Skipping chunk %d - empty after cleaning", i)
                    continue

                # Skip garbage chunks (OCR failures, encoding issues)
                stripped = chunk_text.strip()
                if len(stripped) < 50:
                    logger.warning("Skipping chunk %d - too short (%d chars)", i, len(stripped))
                    continue
                garbage_chars = sum(1 for c in stripped if c == '?' or (not c.isprintable() and c not in '\n\r\t'))
                if len(stripped) > 0 and garbage_chars / len(stripped) > 0.3:
                    logger.warning("Skipping chunk %d - %.0f%% garbage characters", i, garbage_chars / len(stripped) * 100)
                    continue

                # Generate contextual prefix for improved retrieval
                embedding_text = generate_chunk_context(chunk_text, full_doc_text, filename)

                # Try embedding cache first
                content_hash = get_content_hash(embedding_text)
                embedding = get_cached_embedding(content_hash, conn)

                if embedding is None:
                    # Cache miss - generate embedding via API
                    try:
                        embedding = Settings.embed_model.get_text_embedding(embedding_text)
                    except Exception as e:
                        logger.warning("Skipping chunk %d - embedding failed: %s", i, e)
                        continue
                    # Store in cache for future use
                    store_cached_embedding(content_hash, embedding, conn)
                else:
                    logger.debug("Cache hit for chunk %d embedding", i)

                # Extract page number from metadata
                page_num = None
                if hasattr(node, 'metadata') and node.metadata:
                    page_num = node.metadata.get('page') or node.metadata.get('page_number') or node.metadata.get('page_label')
                    # Convert to int if it's a string
                    if page_num is not None:
                        try:
                            page_num = int(page_num)
                        except (ValueError, TypeError):
                            page_num = None
                # Also check source_node for inherited metadata
                if page_num is None and hasattr(node, 'source_node') and node.source_node:
                    source_meta = getattr(node.source_node, 'metadata', None)
                    if source_meta:
                        page_num = source_meta.get('page') or source_meta.get('page_number') or source_meta.get('page_label')
                        if page_num is not None:
                            try:
                                page_num = int(page_num)
                            except (ValueError, TypeError):
                                page_num = None

                # Determine content_type from node metadata (text/table/image)
                content_type = "text"
                if hasattr(node, 'metadata') and node.metadata:
                    content_type = node.metadata.get('content_type', 'text')
                if content_type == "text" and hasattr(node, 'source_node') and node.source_node:
                    source_meta = getattr(node.source_node, 'metadata', None)
                    if source_meta:
                        content_type = source_meta.get('content_type', 'text')

                # Insert chunk with ON CONFLICT for idempotency
                # Safe to retry - won't create duplicates
                cur.execute(
                    """INSERT INTO chunks (document_id, project_id, content, embedding, chunk_index, page_number, content_type)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (document_id, chunk_index) DO NOTHING""",
                    (document_id, project_id, chunk_text, embedding, i, page_num, content_type)
                )
                valid_chunk_count += 1

                # Report embedding progress (40% -> 90% range)
                if total_nodes > 0:
                    embed_pct = 40 + int(50 * (i + 1) / total_nodes)
                    report_progress(job_id, embed_pct)

            # Ensure at least some chunks were successfully embedded
            if valid_chunk_count == 0:
                raise ValueError("No valid text chunks could be embedded from the document")

            # Update document status to ready
            cur.execute(
                "UPDATE documents SET status = 'ready', chunk_count = %s WHERE id = %s",
                (valid_chunk_count, document_id)
            )

            # Mark job as completed with 100% progress
            cur.execute(
                """UPDATE jobs SET status = 'completed', progress = 100, completed_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (job_id,)
            )
        conn.commit()

    report_progress(job_id, 100)  # Done
    logger.info("Processed document %d: %d/%d chunks embedded", document_id, valid_chunk_count, len(nodes))
