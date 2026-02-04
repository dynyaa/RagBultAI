"""Advanced retrieval module for production RAG.

This module implements:
- Hybrid search (BM25 keyword + vector semantic search)
- Cross-encoder reranking for improved precision
- HyDE (Hypothetical Document Embeddings) query transformation
- Configurable retrieval pipeline
- Faithfulness scoring using LLM-as-judge

These techniques combined can improve retrieval accuracy by 35-45%
compared to basic vector-only search.
"""
import logging
import os
import re
import math
from typing import List, Optional
from dataclasses import dataclass, field

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from .config import Config
from .db import get_db

logger = logging.getLogger("rag.retrieval")

# Cross-encoder reranking (lazy loaded)
_reranker = None
_cohere_client = None


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval pipeline.

    Attributes:
        hybrid_alpha: Balance between keyword (0) and vector (1) search
        initial_top_k: Number of candidates to retrieve before reranking
        final_top_k: Final number of results after reranking
        use_reranking: Whether to apply cross-encoder reranking
        use_hyde: Whether to use HyDE query transformation
        use_cohere: Whether to use Cohere reranking (requires API key)
    """
    hybrid_alpha: float = field(default_factory=lambda: Config.HYBRID_SEARCH_ALPHA)
    initial_top_k: int = field(default_factory=lambda: Config.RERANK_TOP_K)
    final_top_k: int = field(default_factory=lambda: Config.TOP_K)
    use_reranking: bool = field(default_factory=lambda: Config.USE_RERANKING)
    use_hyde: bool = field(default_factory=lambda: Config.USE_HYDE)
    use_cohere: bool = field(default_factory=lambda: Config.USE_COHERE_RERANK)


def get_reranker():
    """Lazy load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", Config.RERANKER_MODEL)
            _reranker = CrossEncoder(Config.RERANKER_MODEL, max_length=512)
            logger.info("Reranker model loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
            return None
        except Exception as e:
            logger.warning("Failed to load reranker: %s", e)
            return None
    return _reranker


def get_cohere_client():
    """Lazy load the Cohere client."""
    global _cohere_client
    if _cohere_client is None and Config.COHERE_API_KEY:
        try:
            import cohere
            _cohere_client = cohere.Client(Config.COHERE_API_KEY)
            logger.info("Cohere client initialized")
        except ImportError:
            logger.warning("cohere package not installed")
            return None
        except Exception as e:
            logger.warning("Failed to initialize Cohere: %s", e)
            return None
    return _cohere_client


def preload_models():
    """Preload ML models at startup to avoid cold-start latency.

    Call this function during application startup to load the reranker
    model before the first request. This eliminates the ~30 second
    delay on the first chat query.
    """
    if Config.USE_RERANKING:
        logger.info("Preloading reranker model at startup...")
        get_reranker()
    if Config.USE_COHERE_RERANK and Config.COHERE_API_KEY:
        get_cohere_client()


def vector_retrieve(
    project_id: int,
    query_embedding: List[float],
    top_k: int = 20
) -> List[dict]:
    """Pure vector similarity search using pgvector."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.content, c.chunk_index, c.document_id,
                       d.filename, 1 - (c.embedding <=> %s::vector) as similarity,
                       c.page_number
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.project_id = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, project_id, query_embedding, top_k))
            rows = cur.fetchall()

    return [
        {
            "chunk_id": r[0],
            "content": r[1],
            "chunk_index": r[2],
            "document_id": r[3],
            "filename": r[4],
            "vector_score": float(r[5]),
            "keyword_score": 0.0,
            "combined_score": float(r[5]),
            "page_number": r[6]
        }
        for r in rows
    ]


def hybrid_retrieve(
    project_id: int,
    query: str,
    query_embedding: List[float],
    alpha: float = 0.5,
    top_k: int = 20
) -> List[dict]:
    """Hybrid search combining BM25 keyword search with vector similarity.

    Args:
        project_id: Project to search within
        query: Original query text for keyword search
        query_embedding: Pre-computed query embedding for vector search
        alpha: Balance factor (0=keyword only, 1=vector only, 0.5=balanced)
        top_k: Number of results to return

    Returns:
        List of chunk dictionaries with combined scores
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            # Check if tsvector column exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'chunks' AND column_name = 'content_tsv'
                )
            """)
            has_tsvector = cur.fetchone()[0]

            if not has_tsvector:
                # Fall back to vector-only search
                return vector_retrieve(project_id, query_embedding, top_k)

            # Hybrid search with score normalization
            # Use 'simple' text search config for multilingual support
            # (works with Russian, English, and any language)
            cur.execute("""
                WITH vector_search AS (
                    SELECT id, 1 - (embedding <=> %s::vector) as vscore
                    FROM chunks
                    WHERE project_id = %s
                ),
                keyword_search AS (
                    SELECT id,
                           ts_rank_cd(content_tsv, plainto_tsquery('simple', %s), 32) as kscore
                    FROM chunks
                    WHERE project_id = %s
                      AND content_tsv @@ plainto_tsquery('simple', %s)
                ),
                -- Normalize scores to 0-1 range
                vector_normalized AS (
                    SELECT id, vscore,
                           (vscore - MIN(vscore) OVER()) /
                           NULLIF(MAX(vscore) OVER() - MIN(vscore) OVER(), 0) as vscore_norm
                    FROM vector_search
                ),
                keyword_normalized AS (
                    SELECT id, kscore,
                           (kscore - MIN(kscore) OVER()) /
                           NULLIF(MAX(kscore) OVER() - MIN(kscore) OVER(), 0) as kscore_norm
                    FROM keyword_search
                )
                SELECT c.id, c.content, c.chunk_index, c.document_id, d.filename,
                       COALESCE(v.vscore, 0) as vector_score,
                       COALESCE(k.kscore, 0) as keyword_score,
                       (COALESCE(v.vscore_norm, 0) * %s +
                        COALESCE(k.kscore_norm, 0) * %s) as combined_score,
                       c.page_number
                FROM chunks c
                LEFT JOIN vector_normalized v ON v.id = c.id
                LEFT JOIN keyword_normalized k ON k.id = c.id
                JOIN documents d ON d.id = c.document_id
                WHERE c.project_id = %s
                  AND (v.vscore IS NOT NULL OR k.kscore IS NOT NULL)
                ORDER BY combined_score DESC
                LIMIT %s
            """, (
                query_embedding, project_id,
                query, project_id, query,
                alpha, 1 - alpha,
                project_id, top_k
            ))
            rows = cur.fetchall()

    return [
        {
            "chunk_id": r[0],
            "content": r[1],
            "chunk_index": r[2],
            "document_id": r[3],
            "filename": r[4],
            "vector_score": float(r[5]) if r[5] else 0.0,
            "keyword_score": float(r[6]) if r[6] else 0.0,
            "combined_score": float(r[7]) if r[7] else 0.0,
            "page_number": r[8]
        }
        for r in rows
    ]


def normalize_scores_sigmoid(chunks: List[dict]) -> List[dict]:
    """Normalize cross-encoder scores using sigmoid for absolute relevance.

    Unlike min-max normalization (which always makes the best result 100%
    even if all results are irrelevant), sigmoid converts raw logits to
    meaningful probabilities:
    - Logit > 0 → model thinks relevant (>50%)
    - Logit < 0 → model thinks irrelevant (<50%)
    - Score reflects absolute relevance, not just relative ranking

    Cross-encoder/ms-marco models output logits roughly in [-10, 10] range.
    """
    if not chunks:
        return chunks

    for chunk in chunks:
        raw = chunk.get("rerank_score", 0)
        # Sigmoid: 1 / (1 + e^(-x))
        # This gives meaningful absolute probabilities
        chunk["rerank_score"] = 1.0 / (1.0 + math.exp(-raw))

    return chunks


def _apply_rerank_postprocessing(
    query: str,
    chunks: List[dict],
    diversity_penalty: float = None,
    keyword_boost: float = None,
) -> List[dict]:
    """Post-process reranked results with diversity penalty and keyword boost.

    This improves result quality by:
    1. Penalizing chunks from the same document section (same document_id and
       adjacent chunk_index) so that results cover more distinct sections.
    2. Boosting chunks that contain exact significant query terms, rewarding
       surface-level relevance on top of semantic reranking.

    The function operates on chunks that already have a ``rerank_score``
    (normalized 0-1 via sigmoid). Adjustments are additive/subtractive so
    the score remains interpretable.

    Args:
        query: Original user query (used for keyword matching).
        chunks: List of chunks with ``rerank_score`` already set.
        diversity_penalty: Score penalty for same-section neighbours.
            Defaults to ``Config.DIVERSITY_PENALTY``.
        keyword_boost: Score bonus for chunks containing query keywords.
            Defaults to ``Config.KEYWORD_BOOST``.

    Returns:
        The same list, re-sorted after adjustments.
    """
    if not chunks:
        return chunks

    if diversity_penalty is None:
        diversity_penalty = Config.DIVERSITY_PENALTY
    if keyword_boost is None:
        keyword_boost = Config.KEYWORD_BOOST

    # --- Keyword boost -----------------------------------------------------------
    is_cyrillic = _detect_has_cyrillic(query)
    stop_words = _STOP_WORDS_RU | _STOP_WORDS_EN if is_cyrillic else _STOP_WORDS_EN
    query_keywords = {
        w for w in query.lower().split()
        if w not in stop_words and len(w) > 2
    }

    if query_keywords:
        for chunk in chunks:
            content_lower = chunk.get("content", "").lower()
            matched = sum(1 for kw in query_keywords if kw in content_lower)
            if matched > 0:
                # Proportional boost based on fraction of keywords matched
                boost = keyword_boost * (matched / len(query_keywords))
                chunk["rerank_score"] = chunk.get("rerank_score", 0) + boost

    # --- Diversity penalty -------------------------------------------------------
    # Track which (document_id, chunk_index) regions have already been selected.
    # For each chunk, if a "neighbour" (same doc, chunk_index within +/-1) already
    # appeared higher in the ranking, apply a penalty.
    seen_sections: set = set()  # (document_id, chunk_index) tuples

    # Sort by current score before applying penalty (greedy selection)
    chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    for chunk in chunks:
        doc_id = chunk.get("document_id")
        c_idx = chunk.get("chunk_index")
        if doc_id is not None and c_idx is not None:
            # Check if a neighbouring chunk from the same document is already selected
            neighbours = {
                (doc_id, c_idx - 1),
                (doc_id, c_idx),
                (doc_id, c_idx + 1),
            }
            if neighbours & seen_sections:
                chunk["rerank_score"] = max(
                    0.0, chunk.get("rerank_score", 0) - diversity_penalty
                )
            seen_sections.add((doc_id, c_idx))

    # Re-sort after adjustments
    chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return chunks


def rerank_with_cross_encoder(
    query: str,
    chunks: List[dict],
    top_k: int = 5
) -> List[dict]:
    """Rerank chunks using a cross-encoder model.

    Cross-encoders are more accurate than bi-encoders for ranking
    because they see query and document together.

    Args:
        query: Original user query
        chunks: List of candidate chunks from initial retrieval
        top_k: Number of top results to return

    Returns:
        Reranked list of chunks with rerank_score added (normalized 0-1)
    """
    if not chunks:
        return chunks

    reranker = get_reranker()
    if reranker is None:
        return chunks[:top_k]

    # Prepare query-document pairs
    pairs = [[query, chunk["content"]] for chunk in chunks]

    # Score all pairs
    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.warning("Reranking failed: %s", e)
        return chunks[:top_k]

    # Add raw scores first
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # Sort by raw score
    reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)

    # Normalize with sigmoid for absolute relevance (on full set for better stats)
    normalize_scores_sigmoid(reranked)

    # Apply diversity penalty and keyword boost post-processing
    reranked = _apply_rerank_postprocessing(query, reranked)

    # Take top_k after post-processing
    return reranked[:top_k]


def rerank_with_cohere(
    query: str,
    chunks: List[dict],
    top_k: int = 5
) -> List[dict]:
    """Rerank chunks using Cohere's rerank API.

    Higher quality than local cross-encoder but requires API key and has cost.

    Args:
        query: Original user query
        chunks: List of candidate chunks
        top_k: Number of results to return

    Returns:
        Reranked chunks with rerank_score
    """
    if not chunks:
        return chunks

    client = get_cohere_client()
    if client is None:
        return rerank_with_cross_encoder(query, chunks, top_k)

    docs = [chunk["content"] for chunk in chunks]

    try:
        # Request more candidates than final top_k to allow post-processing to re-sort
        cohere_top_n = min(len(docs), max(top_k * 2, top_k + 10))
        response = client.rerank(
            query=query,
            documents=docs,
            top_n=cohere_top_n,
            model="rerank-english-v3.0"
        )

        reranked = []
        for result in response.results:
            chunk = chunks[result.index].copy()
            chunk["rerank_score"] = result.relevance_score
            reranked.append(chunk)

        # Apply diversity penalty and keyword boost post-processing
        reranked = _apply_rerank_postprocessing(query, reranked)
        return reranked[:top_k]

    except Exception as e:
        logger.warning("Cohere reranking failed: %s, falling back to local reranker", e)
        return rerank_with_cross_encoder(query, chunks, top_k)


async def hyde_transform(query: str) -> str:
    """Generate a hypothetical document for the query using HyDE.

    HyDE (Hypothetical Document Embeddings) generates a hypothetical
    answer to the query, which is then used for retrieval instead of
    the raw query. This improves retrieval for conceptual queries.

    Args:
        query: Original user query

    Returns:
        Hypothetical document text
    """
    llm = OpenAI(model=Config.LLM_MODEL, api_key=Config.OPENAI_API_KEY)

    prompt = f"""Write a short, detailed paragraph that would be a perfect answer
to this question. Be specific and factual. Do not say "I don't know" or
ask clarifying questions. Just provide the answer as if you know it.

Question: {query}

Answer:"""

    try:
        response = await llm.acomplete(prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning("HyDE transform failed: %s", e)
        return query  # Fall back to original query


async def generate_query_variations(query: str, count: int = 3) -> List[str]:
    """Generate paraphrased query variations for broader retrieval coverage.

    Multi-query retrieval reduces document miss rates by ~40% by searching
    with multiple phrasings of the same question.

    Args:
        query: Original user query
        count: Number of variations to generate (default 3)

    Returns:
        List of query variations (always includes original query)
    """
    if not Config.USE_MULTI_QUERY:
        return [query]

    llm = OpenAI(model=Config.LLM_MODEL, api_key=Config.OPENAI_API_KEY)

    prompt = f"""Generate {count} alternative phrasings of this search query.
Each variation should approach the topic from a different angle or use different keywords.
Keep each variation concise (under 30 words).

Original query: {query}

Return ONLY the variations, one per line, numbered 1-{count}. Do not include the original query."""

    try:
        response = await llm.acomplete(prompt)
        lines = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
        # Clean numbering (remove "1.", "1)", etc.)
        variations = []
        for line in lines[:count]:
            cleaned = re.sub(r'^\d+[\.\)\-\:]\s*', '', line).strip()
            if cleaned:
                variations.append(cleaned)

        # Always include original query first
        return [query] + variations
    except Exception as e:
        logger.warning("Multi-query generation failed: %s", e)
        return [query]


def is_complex_query(query: str) -> bool:
    """Heuristic classifier to detect multi-part / multi-hop questions.

    Checks for comparison keywords, conjunctions joining distinct sub-questions,
    multiple named entities, and other patterns that signal the query benefits
    from decomposition.

    Args:
        query: The user query string.

    Returns:
        True if the query is likely complex and should be decomposed.
    """
    q_lower = query.lower().strip()

    # Pattern 1: Explicit comparison / contrast language
    comparison_keywords = [
        "compare", "comparison", "contrast", "difference between",
        "differences between", "how does .* differ", "vs", "versus",
        "similarities between", "similar to",
        # Russian equivalents
        "сравни", "сравнение", "разница между", "различия между",
        "отличия между", "чем отличается",
    ]
    for kw in comparison_keywords:
        if re.search(kw, q_lower):
            return True

    # Pattern 2: Conjunction joining two distinct question parts
    # e.g., "What is X and how does Y work?"
    multi_part_patterns = [
        r'\b(and|и)\b.+\?',            # "X and Y?" style
        r'\b(both|оба|обе|оба)\b',     # "both A and B"
        r'\b(as well as|а также)\b',
        r'\b(in addition|кроме того)\b',
    ]
    for pat in multi_part_patterns:
        if re.search(pat, q_lower):
            return True

    # Pattern 3: Multiple question marks (compound question)
    if query.count('?') >= 2:
        return True

    # Pattern 4: Listing patterns ("first… second…", "1)… 2)…")
    if re.search(r'\b(first|second|third|firstly|secondly|thirdly)\b', q_lower):
        return True
    if re.search(r'(\d+[\.\)]\s)', query):
        return True

    # Pattern 5: Multiple proper nouns / entities (heuristic: uppercase words)
    # Skip the very first word (sentence start) and stop words
    words = query.split()
    capitalized = [
        w for w in words[1:]
        if w[0].isupper() and w.lower() not in _STOP_WORDS_EN and len(w) > 1
    ] if len(words) > 1 else []
    if len(capitalized) >= 3:
        return True

    return False


async def decompose_query(query: str) -> List[str]:
    """Use an LLM to break a complex query into 2-4 atomic sub-queries.

    Each sub-query should be self-contained and answerable independently.
    The final synthesis happens after retrieval for each sub-query.

    Args:
        query: The complex user query.

    Returns:
        List of sub-queries (2-4 items). Falls back to [query] on error.
    """
    llm = OpenAI(model=Config.LLM_MODEL, api_key=Config.OPENAI_API_KEY)

    prompt = f"""Break the following complex question into 2-4 simpler, independent sub-questions.
Each sub-question should target one specific piece of information needed to answer the original question.
Keep each sub-question concise and self-contained.

Original question: {query}

Return ONLY the sub-questions, one per line, numbered 1-4. Do not include explanations."""

    try:
        response = await llm.acomplete(prompt)
        lines = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
        sub_queries = []
        for line in lines[:4]:
            cleaned = re.sub(r'^\d+[\.\)\-\:]\s*', '', line).strip()
            if cleaned:
                sub_queries.append(cleaned)

        if len(sub_queries) >= 2:
            logger.info("Decomposed query into %d sub-queries", len(sub_queries))
            return sub_queries
        # If LLM returned fewer than 2, decomposition is not useful
        return [query]
    except Exception as e:
        logger.warning("Query decomposition failed: %s", e)
        return [query]


def reciprocal_rank_fusion(result_lists: List[List[dict]], k: int = 60) -> List[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF is a robust method for combining results from different queries/retrievers.
    Score = sum(1 / (k + rank)) across all lists where the item appears.

    Args:
        result_lists: List of ranked result lists from different queries
        k: Smoothing constant (default 60, standard in literature)

    Returns:
        Merged and re-ranked list of unique chunks
    """
    fused_scores = {}  # chunk_id -> (score, chunk_data)

    for results in result_lists:
        for rank, chunk in enumerate(results):
            chunk_id = chunk["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)

            if chunk_id in fused_scores:
                fused_scores[chunk_id] = (
                    fused_scores[chunk_id][0] + rrf_score,
                    fused_scores[chunk_id][1]  # Keep first occurrence's data
                )
            else:
                fused_scores[chunk_id] = (rrf_score, chunk)

    # Sort by fused score descending
    sorted_results = sorted(fused_scores.values(), key=lambda x: x[0], reverse=True)

    # Return chunks with updated combined scores
    merged = []
    for score, chunk in sorted_results:
        chunk_copy = chunk.copy()
        chunk_copy["combined_score"] = score
        merged.append(chunk_copy)

    return merged


def _detect_has_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters (Russian, etc.)."""
    return bool(re.search(r'[а-яА-ЯёЁ]', text))


# Stop words for supported languages
_STOP_WORDS_EN = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
    'who', 'how', 'when', 'where', 'why', 'do', 'does', 'did',
    'can', 'could', 'would', 'should', 'of', 'in', 'on', 'at',
    'to', 'for', 'with', 'by', 'from', 'as', 'into', 'about',
    'that', 'this', 'it', 'they', 'them', 'their', 'there', 'be',
    'been', 'being', 'have', 'has', 'had', 'will', 'not', 'but',
    'or', 'and', 'if', 'so', 'my', 'your', 'his', 'her', 'its'
}

_STOP_WORDS_RU = {
    'что', 'как', 'в', 'на', 'с', 'по', 'для', 'от', 'из', 'за', 'о',
    'об', 'до', 'у', 'к', 'и', 'а', 'но', 'или', 'не', 'ни', 'да',
    'это', 'то', 'он', 'она', 'оно', 'они', 'мы', 'вы', 'я', 'ты',
    'его', 'её', 'их', 'мой', 'наш', 'ваш', 'свой', 'этот', 'тот',
    'весь', 'все', 'вся', 'всё', 'быть', 'был', 'была', 'было', 'были',
    'есть', 'будет', 'бы', 'же', 'ли', 'вот', 'так', 'уже', 'тоже',
    'только', 'ещё', 'при', 'через', 'между', 'когда', 'где', 'кто',
    'какой', 'какая', 'какое', 'какие', 'чем', 'чего', 'кого',
    'который', 'которая', 'которое', 'которые', 'происходит'
}


def filter_citations(
    chunks: List[dict],
    query: str,
    min_score: float = 0.4,
    max_citations: int = 5
) -> List[dict]:
    """Filter citations for relevance and quality.

    Production RAG best practice: Don't show all retrieved chunks.
    Filter to only the most relevant ones.

    Args:
        chunks: Retrieved and reranked chunks
        query: Original query for relevance checking
        min_score: Minimum relevance score threshold (0-1)
        max_citations: Maximum number of citations to return

    Returns:
        Filtered list of high-quality citations
    """
    if not chunks:
        return chunks

    # Select stop words based on query language
    is_cyrillic = _detect_has_cyrillic(query)
    stop_words = _STOP_WORDS_RU | _STOP_WORDS_EN if is_cyrillic else _STOP_WORDS_EN

    # Extract query keywords for document relevance checking
    query_words = set(query.lower().split())
    query_keywords = query_words - stop_words

    # Minimum content length for informative citations
    MIN_CONTENT_LENGTH = 80  # Skip headers, TOC entries, etc.

    filtered = []
    for chunk in chunks:
        score = chunk.get("rerank_score", chunk.get("similarity", 0))

        # Filter 1: Minimum score threshold
        if score < min_score:
            continue

        content = chunk.get("content", "")

        # Filter 2: Minimum content length — skip uninformative headers/TOC
        if len(content.strip()) < MIN_CONTENT_LENGTH:
            continue

        # Filter 3: Document relevance check
        # Chunk content should contain at least one query keyword
        content_lower = content.lower()
        filename_lower = chunk.get("filename", "").lower()

        # Check if any query keyword appears in content or filename
        # Use stem-based matching (first 4+ chars) to handle inflected forms
        # e.g., "футболу" matches "футбол", "футбольный", "футбольного"
        has_keyword_match = False
        if query_keywords:
            for kw in query_keywords:
                stem = kw[:4] if len(kw) > 4 else kw
                if stem in content_lower or stem in filename_lower:
                    has_keyword_match = True
                    break
        else:
            has_keyword_match = True  # If no keywords extracted, don't filter

        if not has_keyword_match:
            # Mild penalty for chunks without keyword matches
            chunk["rerank_score"] = score * 0.8
            if chunk["rerank_score"] < min_score:
                continue

        filtered.append(chunk)

    # Sort by score and limit
    filtered.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return filtered[:max_citations]


async def advanced_retrieve(
    project_id: int,
    query: str,
    config: Optional[RetrievalConfig] = None
) -> List[dict]:
    """Production RAG retrieval pipeline.

    Combines hybrid search, reranking, and optional query transformation
    for maximum retrieval quality.

    Args:
        project_id: Project to search within
        query: User query
        config: Retrieval configuration (uses defaults if not provided)

    Returns:
        List of relevant chunks with scores
    """
    if config is None:
        config = RetrievalConfig()

    # Step 1: Query transformation (HyDE)
    search_text = query
    if config.use_hyde:
        search_text = await hyde_transform(query)

    # Step 1.5: Query decomposition for complex multi-hop questions
    # If enabled and the query is complex, decompose into sub-queries and
    # retrieve for each independently.  Results are merged via RRF.
    sub_queries: List[str] = []
    if Config.USE_QUERY_DECOMPOSITION and is_complex_query(search_text):
        sub_queries = await decompose_query(search_text)
        # Only use decomposition if we got multiple sub-queries
        if len(sub_queries) < 2:
            sub_queries = []

    # Step 2: Generate query variations for multi-query retrieval
    # When decomposition produced sub-queries, generate variations for EACH
    # sub-query independently (they complement each other).
    all_result_lists = []

    queries_to_expand = sub_queries if sub_queries else [search_text]
    for base_query in queries_to_expand:
        query_variations = await generate_query_variations(base_query)

        # Step 3: Retrieve for each query variation
        for q_variation in query_variations:
            # Generate embedding for this variation
            q_embedding = Settings.embed_model.get_text_embedding(q_variation)

            # Hybrid retrieve
            candidates = hybrid_retrieve(
                project_id=project_id,
                query=q_variation,
                query_embedding=q_embedding,
                alpha=config.hybrid_alpha,
                top_k=config.initial_top_k
            )
            all_result_lists.append(candidates)

    # Step 4: Merge results using Reciprocal Rank Fusion
    if len(all_result_lists) > 1:
        candidates = reciprocal_rank_fusion(all_result_lists)[:config.initial_top_k]
    else:
        candidates = all_result_lists[0] if all_result_lists else []

    # Step 5: Reranking
    if config.use_reranking and len(candidates) > config.final_top_k:
        if config.use_cohere and Config.COHERE_API_KEY:
            reranked = rerank_with_cohere(query, candidates, config.final_top_k)
        else:
            reranked = rerank_with_cross_encoder(query, candidates, config.final_top_k)
    else:
        reranked = candidates[:config.final_top_k]

    # Step 6: Citation filtering
    # Use adaptive threshold: cross-encoder models (especially multilingual)
    # produce low absolute scores. Fall back to top candidates if filtering
    # removes everything.
    results = filter_citations(
        reranked,
        query=query,
        min_score=0.15,
        max_citations=config.final_top_k
    )

    # Fallback: if filtering removed ALL results but we had candidates,
    # return the top hybrid search candidates by vector_score.
    if not results and candidates:
        fallback = sorted(
            candidates[:config.final_top_k],
            key=lambda x: x.get("vector_score", 0),
            reverse=True
        )[:5]
        for chunk in fallback:
            chunk["similarity"] = chunk.get("vector_score", 0)
        results = fallback
    else:
        # Add similarity score for backwards compatibility
        for chunk in results:
            if "rerank_score" in chunk:
                chunk["similarity"] = chunk["rerank_score"]
            elif "combined_score" in chunk:
                chunk["similarity"] = chunk["combined_score"]
            else:
                chunk["similarity"] = chunk.get("vector_score", 0)

    return results


def retrieve_context(project_id: int, query: str, top_k: int = None) -> List[dict]:
    """Synchronous wrapper for advanced_retrieve.

    Maintains backwards compatibility with existing code while using
    the advanced retrieval pipeline.
    """
    import asyncio

    if top_k is None:
        top_k = Config.TOP_K

    config = RetrievalConfig(final_top_k=top_k)

    # Run async function synchronously
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    advanced_retrieve(project_id, query, config)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                advanced_retrieve(project_id, query, config)
            )
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(advanced_retrieve(project_id, query, config))


async def calculate_faithfulness_score(
    response: str,
    context_chunks: List[dict],
    client  # OpenAI client
) -> dict:
    """
    Calculate faithfulness score using LLM-as-judge pattern.
    Evaluates how well the response is grounded in the provided sources.

    Returns:
        {
            "score": 0.0-1.0,
            "level": "high" | "medium" | "low",
            "reason": "Brief explanation"
        }
    """
    if not context_chunks:
        return {"score": 0.0, "level": "low", "reason": "No sources provided"}

    # Limit context to top 3 chunks for cost efficiency
    context_text = "\n\n---\n".join([c.get("content", "")[:1000] for c in context_chunks[:3]])

    prompt = f"""Evaluate if this response is faithful to the source documents.
A faithful response only contains information that can be verified from the sources.
Penalize responses that add information not found in sources or contradict them.

SOURCES:
{context_text}

RESPONSE TO EVALUATE:
{response[:2000]}

Rate faithfulness from 0 to 100 where:
- 90-100: Fully faithful, all claims supported by sources
- 70-89: Mostly faithful, minor unsupported details
- 50-69: Partially faithful, some claims unsupported
- 0-49: Unfaithful, significant hallucination

Respond in this exact format:
SCORE: [number]
REASON: [one sentence explanation]"""

    try:
        result = await client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap for evaluation
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )

        text = result.choices[0].message.content.strip()

        # Parse response
        score_match = re.search(r'SCORE:\s*(\d+)', text)
        reason_match = re.search(r'REASON:\s*(.+)', text, re.DOTALL)

        score = int(score_match.group(1)) / 100 if score_match else 0.5
        score = max(0.0, min(1.0, score))  # Clamp to 0-1
        reason = reason_match.group(1).strip() if reason_match else "Unable to evaluate"

        if score >= 0.8:
            level = "high"
        elif score >= 0.5:
            level = "medium"
        else:
            level = "low"

        return {"score": round(score, 2), "level": level, "reason": reason}

    except Exception as e:
        logger.warning("Faithfulness scoring failed: %s", e)
        return {"score": 0.5, "level": "medium", "reason": "Evaluation unavailable"}
