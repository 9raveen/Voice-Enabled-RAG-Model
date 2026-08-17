"""
Phase 1 (cont.) — Build deduplicated corpus + eval query set from
MSMARCO-XI Hindi validation data.

Outputs:
  data/corpus.jsonl        - unique passages, each with a stable ID
  data/eval_queries.jsonl  - queries + ground-truth relevant passage IDs
"""

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import json
import hashlib
import os

REPO_ID = "ai4bharat/MSMARCO-XI"
FILENAME = "validation/hinval.parquet"
OUTPUT_DIR = "data"
NUM_QUERIES_SAMPLE = 4000   # subsample size, adjust if needed
RANDOM_SEED = 42


def passage_id(text: str) -> str:
    """Stable ID derived from passage content, so duplicates collapse naturally."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    table = pq.read_table(local_path)
    df = table.to_pandas()

    df = df.sample(n=NUM_QUERIES_SAMPLE, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Subsampled to {len(df)} queries (from {table.num_rows} total) for manageable corpus size")

    corpus = {}          # passage_id -> passage text
    eval_queries = []

    for _, row in df.iterrows():
        passages = row["passages"]
        if not passages:
            continue

        translated = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])

        row_passage_ids = []
        relevant_ids = []

        for i, text in enumerate(translated):
            if not text or not str(text).strip():
                continue
            pid = passage_id(text)
            corpus[pid] = text
            row_passage_ids.append(pid)
            if i < len(is_selected) and is_selected[i] == 1:
                relevant_ids.append(pid)

        eval_queries.append({
            "query_id": int(row["query_id"]),
            "query": row["query"],
            "query_type": row["query_type"],
            "candidate_passage_ids": row_passage_ids,
            "relevant_passage_ids": relevant_ids,
        })

    print(f"Unique passages after dedup: {len(corpus)}")
    print(f"Eval queries written: {len(eval_queries)}")

    with open(os.path.join(OUTPUT_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for pid, text in corpus.items():
            f.write(json.dumps({"passage_id": pid, "text": text}, ensure_ascii=False) + "\n")

    with open(os.path.join(OUTPUT_DIR, "eval_queries.jsonl"), "w", encoding="utf-8") as f:
        for q in eval_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print("Done. Files written to data/corpus.jsonl and data/eval_queries.jsonl")


if __name__ == "__main__":
    build()