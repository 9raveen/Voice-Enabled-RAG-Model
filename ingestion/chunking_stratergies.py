"""
Phase 2, Step 2 — Chunking strategy implementations.
Each function takes the raw corpus (list of {passage_id, text}) and
returns a list of chunks: {chunk_id, source_passage_id, text, metadata}
"""

import hashlib


def chunk_id(text: str, source_id: str, idx: int) -> str:
    raw = f"{source_id}_{idx}_{text[:50]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def strategy_a_baseline(corpus: list[dict]) -> list[dict]:
    """No splitting — one chunk per passage, as-is."""
    chunks = []
    for row in corpus:
        chunks.append({
            "chunk_id": chunk_id(row["text"], row["passage_id"], 0),
            "source_passage_id": row["passage_id"],
            "text": row["text"],
            "metadata": {},
        })
    return chunks


def recursive_split(text: str, max_len: int = 800, overlap: int = 100) -> list[str]:
    """Simple recursive-ish splitter: break on sentence boundaries first,
    fall back to hard split if a single sentence still exceeds max_len."""
    if len(text) <= max_len:
        return [text]

    # naive sentence split on common Hindi/English sentence enders
    import re
    sentences = re.split(r'(?<=[।.!?])\s+', text)

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_len:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
            # overlap: carry last `overlap` chars forward
            current = current[-overlap:] + " " + sent if current else sent
    if current:
        chunks.append(current)

    # fallback: if any chunk is still too long (single giant sentence), hard-split it
    final_chunks = []
    for c in chunks:
        if len(c) <= max_len * 1.5:
            final_chunks.append(c)
        else:
            for i in range(0, len(c), max_len - overlap):
                final_chunks.append(c[i:i + max_len])

    return final_chunks


def strategy_b_longtail_split(corpus: list[dict], threshold: int = 800) -> list[dict]:
    """Passages under threshold stay whole; longer ones get recursively split."""
    chunks = []
    for row in corpus:
        text = row["text"]
        if len(text) <= threshold:
            pieces = [text]
        else:
            pieces = recursive_split(text, max_len=threshold)

        for idx, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": chunk_id(piece, row["passage_id"], idx),
                "source_passage_id": row["passage_id"],
                "text": piece,
                "metadata": {"split": len(pieces) > 1, "part": idx, "total_parts": len(pieces)},
            })
    return chunks


def strategy_c_metadata_enriched(corpus: list[dict], passage_meta: dict) -> list[dict]:
    """Same as baseline, but attach metadata (e.g. query_type) per chunk."""
    chunks = []
    for row in corpus:
        meta = passage_meta.get(row["passage_id"], {})
        chunks.append({
            "chunk_id": chunk_id(row["text"], row["passage_id"], 0),
            "source_passage_id": row["passage_id"],
            "text": row["text"],
            "metadata": meta,
        })
    return chunks


def strategy_d_adaptive(corpus: list[dict], passage_meta: dict, threshold: int = 800) -> list[dict]:
    """Long-tail split AND metadata enrichment combined."""
    chunks = []
    for row in corpus:
        text = row["text"]
        meta = passage_meta.get(row["passage_id"], {})

        if len(text) <= threshold:
            pieces = [text]
        else:
            pieces = recursive_split(text, max_len=threshold)

        for idx, piece in enumerate(pieces):
            chunk_meta = dict(meta)
            chunk_meta.update({"split": len(pieces) > 1, "part": idx, "total_parts": len(pieces)})
            chunks.append({
                "chunk_id": chunk_id(piece, row["passage_id"], idx),
                "source_passage_id": row["passage_id"],
                "text": piece,
                "metadata": chunk_meta,
            })
    return chunks