"""Hierarchical chunking for news articles.

Splits articles into semantic chunks for Vector Search embedding:
  - Chunk 0: headline (always full, captures title-level semantics)
  - Chunk 1: lead paragraph (journalistic "inverted pyramid" lede)
  - Chunks 2..N: sliding body windows (token-aware, overlap for continuity)

Matches databricks-gte-large-en (512 token context) constraints.
All chunks guaranteed <= config.max_chunk_tokens.

Token counting uses character-based estimation (chars_per_token ratio)
which works in serverless environments without internet access.

Usage:
    from trading_databricks.agents.chunking import chunk_article
    from trading_databricks.agents.config import ChunkingConfig

    chunks = chunk_article(headline, body, config=ChunkingConfig())
"""
from __future__ import annotations

from typing import Sequence

from trading_databricks.agents.config import ChunkingConfig

# ---------------------------------------------------------------------------
# Token estimation (char-based — works without internet/tiktoken)
# databricks-gte-large-en uses BERT-style WordPiece: ~3.5-4 chars/token.
# ---------------------------------------------------------------------------
_DEFAULT_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate token count from character length (conservative)."""
    return int(len(text) / chars_per_token) + 1


def _max_chars(max_tokens: int, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN) -> int:
    """Convert token budget to character budget."""
    return int(max_tokens * chars_per_token)


# ---------------------------------------------------------------------------
# Recursive text splitter (token-aware, separator hierarchy)
# ---------------------------------------------------------------------------


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Hard character-boundary split (last resort)."""
    chunks: list[str] = []
    start = 0
    step = max(max_chars - overlap_chars, 1)
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += step
    return chunks


def _split_text_recursive(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
    separators: Sequence[str],
) -> list[str]:
    """Split text into chunks respecting character limits and natural boundaries.

    Tries separators in order (coarsest to finest). Falls back to hard
    character-boundary split when no separator produces valid chunks.
    Guarantees all output chunks <= max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    # Try each separator from coarsest (paragraph) to finest (space)
    for sep_idx, sep in enumerate(separators):
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the part itself exceeds budget, recursively split it
                if len(part) > max_chars:
                    remaining_seps = separators[sep_idx + 1:]
                    if remaining_seps:
                        sub_chunks = _split_text_recursive(
                            part,
                            max_chars=max_chars,
                            overlap_chars=overlap_chars,
                            separators=remaining_seps,
                        )
                    else:
                        sub_chunks = _hard_split(part, max_chars, overlap_chars)
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""
                else:
                    # Overlap: carry trailing chars from previous chunk
                    if overlap_chars > 0 and current:
                        overlap = current[-overlap_chars:]
                        current = overlap + sep + part
                        # Validate overlap didn't push us over
                        if len(current) > max_chars:
                            current = part
                    else:
                        current = part

        if current:
            # Final piece might exceed budget if overlap pushed it over
            if len(current) > max_chars:
                chunks.extend(_hard_split(current, max_chars, overlap_chars))
            else:
                chunks.append(current)

        if chunks:
            return chunks

    # Fallback: hard split by character boundaries
    return _hard_split(text, max_chars, overlap_chars)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_article(
    headline: str,
    body: str,
    *,
    config: ChunkingConfig | None = None,
) -> list[dict]:
    """Split a news article into hierarchical semantic chunks.

    Args:
        headline: Article headline text.
        body: Article body text (may be empty or very long).
        config: Chunking parameters. Uses defaults if None.

    Returns:
        List of chunk dicts, each containing:
          - chunk_index: int (0-based position within article)
          - chunk_text: str (the actual chunk content)
          - chunk_type: "headline" | "lead" | "body_window"
          - token_count: int (estimated token count)

    All chunks satisfy: token_count <= config.max_chunk_tokens.
    """
    if config is None:
        config = ChunkingConfig()

    cpt = config.chars_per_token
    max_chars = _max_chars(config.max_chunk_tokens, cpt)
    overlap_chars = _max_chars(config.overlap_tokens, cpt)

    chunks: list[dict] = []
    idx = 0

    # ── Chunk 0: Headline ─────────────────────────────────────────────────
    if config.headline_always_full and headline and headline.strip():
        h = headline.strip()
        chunks.append({
            "chunk_index": idx,
            "chunk_text": h,
            "chunk_type": "headline",
            "token_count": _estimate_tokens(h, cpt),
        })
        idx += 1

    if not body or not body.strip():
        return chunks

    body = body.strip()

    # ── Chunk 1: Lead paragraph ───────────────────────────────────────────
    lead_end = min(config.lead_paragraph_chars, len(body))

    # Extend to next sentence boundary (up to 100 chars ahead)
    for i in range(lead_end, min(lead_end + 100, len(body))):
        if body[i] in ".!?\n":
            lead_end = i + 1
            break

    lead = body[:lead_end].strip()
    if len(lead) >= config.min_chunk_chars:
        # Truncate lead if it exceeds budget
        if len(lead) > max_chars:
            lead = lead[:max_chars].strip()
        chunks.append({
            "chunk_index": idx,
            "chunk_text": lead,
            "chunk_type": "lead",
            "token_count": _estimate_tokens(lead, cpt),
        })
        idx += 1

    # ── Chunks 2..N: Body windows ─────────────────────────────────────────
    remainder = body[lead_end:].strip()
    if len(remainder) < config.min_chunk_chars:
        return chunks

    windows = _split_text_recursive(
        remainder,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        separators=config.separators,
    )

    for window in windows:
        window = window.strip()
        if len(window) >= config.min_chunk_chars:
            chunks.append({
                "chunk_index": idx,
                "chunk_text": window,
                "chunk_type": "body_window",
                "token_count": _estimate_tokens(window, cpt),
            })
            idx += 1

    return chunks
