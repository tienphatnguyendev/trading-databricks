"""Pure silver transforms for the MAS (agents) domain."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def news_chunks_transform(cleaned: DataFrame) -> DataFrame:
    """Build hierarchical chunks from cleaned news articles.

    Applies chunk_article() via scalar UDF to split each article into:
      - headline chunk (always full)
      - lead paragraph chunk (first ~300 chars to sentence boundary)
      - body window chunks (384-token sliding windows with 64-token overlap)

    The UDF is self-contained (all logic inlined) so cloudpickle can
    serialize it to SDP serverless executors without external imports.

    Output schema: chunk_id, story_id, chunk_index, chunk_text,
    chunk_type, token_count, published_at, source.
    """
    from pyspark.sql.types import (
        ArrayType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    _chunk_schema = ArrayType(
        StructType([
            StructField("chunk_index", IntegerType(), nullable=False),
            StructField("chunk_text", StringType(), nullable=False),
            StructField("chunk_type", StringType(), nullable=False),
            StructField("token_count", IntegerType(), nullable=False),
        ])
    )

    # ── Chunking config (captured in closure, no external import needed) ──
    _MAX_CHUNK_TOKENS = 384
    _OVERLAP_TOKENS = 64
    _LEAD_CHARS = 300
    _MIN_CHUNK_CHARS = 50
    _CHARS_PER_TOKEN = 3.5
    _SEPARATORS = ("\n\n", "\n", ". ", ", ", " ")

    # ── Self-contained chunking logic (inlined for executor serialization) ──
    def _estimate_tokens(text):
        return int(len(text) / _CHARS_PER_TOKEN) + 1

    def _hard_split(text, max_chars, overlap_chars):
        chunks = []
        start = 0
        step = max(max_chars - overlap_chars, 1)
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start += step
        return chunks

    def _split_recursive(text, max_chars, overlap_chars, separators):
        if len(text) <= max_chars:
            return [text]
        for sep_idx, sep in enumerate(separators):
            parts = text.split(sep)
            if len(parts) <= 1:
                continue
            chunks = []
            current = ""
            for part in parts:
                candidate = f"{current}{sep}{part}" if current else part
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    if len(part) > max_chars:
                        remaining = separators[sep_idx + 1:]
                        if remaining:
                            sub = _split_recursive(part, max_chars, overlap_chars, remaining)
                        else:
                            sub = _hard_split(part, max_chars, overlap_chars)
                        chunks.extend(sub[:-1])
                        current = sub[-1] if sub else ""
                    else:
                        if overlap_chars > 0 and current:
                            overlap = current[-overlap_chars:]
                            current = overlap + sep + part
                            if len(current) > max_chars:
                                current = part
                        else:
                            current = part
            if current:
                if len(current) > max_chars:
                    chunks.extend(_hard_split(current, max_chars, overlap_chars))
                else:
                    chunks.append(current)
            if chunks:
                return chunks
        return _hard_split(text, max_chars, overlap_chars)

    def _chunk_article(headline, body):
        max_chars = int(_MAX_CHUNK_TOKENS * _CHARS_PER_TOKEN)
        overlap_chars = int(_OVERLAP_TOKENS * _CHARS_PER_TOKEN)
        chunks = []
        idx = 0
        # Chunk 0: Headline
        if headline and headline.strip():
            h = headline.strip()
            chunks.append({
                "chunk_index": idx, "chunk_text": h,
                "chunk_type": "headline", "token_count": _estimate_tokens(h),
            })
            idx += 1
        if not body or not body.strip():
            return chunks
        body = body.strip()
        # Chunk 1: Lead paragraph
        lead_end = min(_LEAD_CHARS, len(body))
        for i in range(lead_end, min(lead_end + 100, len(body))):
            if body[i] in ".!?\n":
                lead_end = i + 1
                break
        lead = body[:lead_end].strip()
        if len(lead) >= _MIN_CHUNK_CHARS:
            if len(lead) > max_chars:
                lead = lead[:max_chars].strip()
            chunks.append({
                "chunk_index": idx, "chunk_text": lead,
                "chunk_type": "lead", "token_count": _estimate_tokens(lead),
            })
            idx += 1
        # Chunks 2..N: Body windows
        remainder = body[lead_end:].strip()
        if len(remainder) >= _MIN_CHUNK_CHARS:
            windows = _split_recursive(remainder, max_chars, overlap_chars, _SEPARATORS)
            for window in windows:
                window = window.strip()
                if len(window) >= _MIN_CHUNK_CHARS:
                    chunks.append({
                        "chunk_index": idx, "chunk_text": window,
                        "chunk_type": "body_window",
                        "token_count": _estimate_tokens(window),
                    })
                    idx += 1
        return chunks

    @F.udf(returnType=_chunk_schema)
    def _chunk_udf(headline: str, body: str) -> list:
        h = str(headline) if headline is not None else ""
        b = str(body) if body is not None else ""
        return _chunk_article(h, b)

    return (
        cleaned
        .withColumn("_chunks", _chunk_udf(F.col("headline"), F.col("body")))
        .select(
            F.col("story_id"),
            F.col("published_at"),
            F.col("source"),
            F.explode("_chunks").alias("_chunk"),
        )
        .select(
            F.concat_ws("_", F.col("story_id"), F.col("_chunk.chunk_index").cast("string")).alias("chunk_id"),
            F.col("story_id"),
            F.col("_chunk.chunk_index").alias("chunk_index"),
            F.col("_chunk.chunk_text").alias("chunk_text"),
            F.col("_chunk.chunk_type").alias("chunk_type"),
            F.col("_chunk.token_count").alias("token_count"),
            F.col("published_at"),
            F.col("source"),
        )
    )
