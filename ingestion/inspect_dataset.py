"""
Phase 1 — Dataset Inspection for MSMARCO-XI (Hindi validation split)
Downloads the parquet file directly and reads it with pyarrow,
bypassing the datasets library's streaming iterator (which has a
known bug with nested struct columns like our `passages` field).
"""

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import statistics

REPO_ID = "ai4bharat/MSMARCO-XI"
FILENAME = "validation/hinval.parquet"
SAMPLE_SIZE = 500
NUM_QUERIES_SAMPLE = 4000   # subsample size, adjust if needed
RANDOM_SEED = 42


def inspect():
    print(f"Downloading {FILENAME} from {REPO_ID} (one-time, cached after this)...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
    )
    print(f"Downloaded to: {local_path}")

    table = pq.read_table(local_path)
    print(f"\nTotal rows in hinval.parquet: {table.num_rows}")
    print(f"Columns: {table.column_names}")

    df = table.slice(0, SAMPLE_SIZE).to_pandas()

    print("\n=== SAMPLE ROW ===")
    print(df.iloc[0])

    passage_counts, translated_lengths, english_lengths = [], [], []
    query_types = {}
    selected_ratios = []
    empty_translation_count = 0

    for _, row in df.iterrows():
        passages = row["passages"]
        translated = passages.get("Translated_passages", []) if passages else []
        english = passages.get("English_passages", []) if passages else []
        is_selected = passages.get("is_selected", []) if passages else []

        passage_counts.append(len(translated))
        if len(is_selected) > 0:
            selected_ratios.append(sum(is_selected) / len(is_selected))

        for t in translated:
            if not t or not str(t).strip():
                empty_translation_count += 1
            else:
                translated_lengths.append(len(t))

        for e in english:
            english_lengths.append(len(e))

        qt = row.get("query_type", "UNKNOWN")
        query_types[qt] = query_types.get(qt, 0) + 1

    print(f"\n=== STATS over {len(df)} rows ===")
    print(f"Avg passages per query: {statistics.mean(passage_counts):.2f}")
    print(f"Avg translated passage length (chars): {statistics.mean(translated_lengths):.1f}")
    print(f"Median translated passage length (chars): {statistics.median(translated_lengths):.1f}")
    print(f"Max translated passage length (chars): {max(translated_lengths)}")
    print(f"Avg English passage length (chars): {statistics.mean(english_lengths):.1f}")
    print(f"Empty/blank translated passages: {empty_translation_count}")
    print(f"Avg fraction of passages marked relevant: {statistics.mean(selected_ratios):.3f}")
    print(f"Query type distribution: {query_types}")


if __name__ == "__main__":
    inspect()