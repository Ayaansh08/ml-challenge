"""Runner script to execute pair_construction end-to-end on a 10k S1 sample."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.pipeline.cleaning import clean_dataframe, normalize_missing, normalize_whitespace
from src.pipeline.pair_construction import (
    build_positive_pairs,
    write_pair_dataset,
)
from src.utils.config import DATA_DIR, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_needed_source_records(file_path: Path, needed_ids: Set[str], chunksize: int = 200_000) -> pd.DataFrame:
    """Read a large TSV in chunks, extracting only rows with entity_id in needed_ids."""
    matched_chunks = []
    found_count = 0
    total_needed = len(needed_ids)

    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, chunksize=chunksize, low_memory=False):
        # Normalize whitespace on entity_id for accurate set membership
        norm_ids = chunk["entity_id"].astype(str).map(normalize_whitespace)
        mask = norm_ids.isin(needed_ids)
        if mask.any():
            matched_chunks.append(chunk[mask])
            found_count += mask.sum()
            if found_count >= total_needed:
                break

    if matched_chunks:
        return pd.concat(matched_chunks, ignore_index=True)
    return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])


def run_sample_pair_construction(
    sample_size: int = 10_000,
    output_dir: Path = OUTPUTS_DIR / "sample_pairs",
) -> None:
    """Run end-to-end pair construction on a sample of Source 1 and matching S2/S3 records."""
    train_dir = DATA_DIR / "train"
    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    print(f"=== Starting Pair Construction on {sample_size:,} S1 sample ===")
    start_time = time.time()

    # 1. Load S1 Sample
    print(f"[1/5] Loading first {sample_size:,} rows of Source 1...")
    raw_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, nrows=sample_size)
    cleaned_s1 = clean_dataframe(raw_s1)

    s1_ids = set(cleaned_s1["cleaned_entity_id"].dropna().unique())
    print(f"      Loaded {len(cleaned_s1):,} S1 rows ({len(s1_ids):,} unique S1 IDs).")

    # 2. Extract Ground Truth matches for S1 sample
    print("[2/5] Filtering Ground Truth for S1 sample entities...")
    raw_gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt_s1_col = "source1_entity_id" if "source1_entity_id" in raw_gt.columns else raw_gt.columns[0]
    gt_match_col = "matched_entity_ids" if "matched_entity_ids" in raw_gt.columns else raw_gt.columns[1]

    raw_gt[gt_s1_col] = raw_gt[gt_s1_col].astype(str).map(normalize_whitespace)
    sample_gt = raw_gt[raw_gt[gt_s1_col].isin(s1_ids)].copy()

    needed_s2_ids: Set[str] = set()
    needed_s3_ids: Set[str] = set()
    singleton_count = 0

    for _, row in sample_gt.iterrows():
        raw_m = normalize_missing(row[gt_match_col])
        if not raw_m:
            singleton_count += 1
            continue
        m_list = [x.strip() for x in raw_m.split(",") if x.strip()]
        if not m_list:
            singleton_count += 1
            continue
        for mid in m_list:
            if mid.startswith("S2-"):
                needed_s2_ids.add(mid)
            elif mid.startswith("S3-"):
                needed_s3_ids.add(mid)

    print(f"      Found {len(sample_gt):,} GT records ({singleton_count:,} singletons).")
    print(f"      Need to extract {len(needed_s2_ids):,} S2 matches and {len(needed_s3_ids):,} S3 matches.")

    # 3. Stream-extract needed S2 and S3 records
    print("[3/5] Extracting matching records from Source 2 and Source 3...")
    s2_df = extract_needed_source_records(s2_path, needed_s2_ids)
    s3_df = extract_needed_source_records(s3_path, needed_s3_ids)

    cleaned_s2 = clean_dataframe(s2_df)
    cleaned_s3 = clean_dataframe(s3_df)

    # Prefix validations
    for name, df, pfx in [("S1", cleaned_s1, "S1-"), ("S2", cleaned_s2, "S2-"), ("S3", cleaned_s3, "S3-")]:
        invalid = ~df["cleaned_entity_id"].fillna("").str.startswith(pfx)
        if invalid.any():
            raise ValueError(f"Invalid prefix found in {name}: {df[invalid]['cleaned_entity_id'].tolist()[:5]}")

    print(f"      Cleaned and validated: S2 rows={len(cleaned_s2):,}, S3 rows={len(cleaned_s3):,}.")

    # 4. Build positive pairs
    print("[4/5] Constructing positive pair records...")
    temp_gt_path = output_dir / "temp_sample_gt.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_gt.to_csv(temp_gt_path, sep="\t", index=False)

    pairs_df = build_positive_pairs(
        ground_truth_path=temp_gt_path,
        cleaned_s1=cleaned_s1,
        cleaned_s2=cleaned_s2,
        cleaned_s3=cleaned_s3,
    )
    if temp_gt_path.exists():
        temp_gt_path.unlink()

    print(f"      Constructed {len(pairs_df):,} total positive pairs.")

    # 5. Write dataset and generate JSON summary
    print(f"[5/5] Writing Parquet datasets and summary to '{output_dir}'...")
    summary = write_pair_dataset(
        positive_pairs_df=pairs_df,
        output_dir=output_dir,
        val_fraction=0.15,
        random_seed=42,
        cleaned_s1=cleaned_s1,
        singleton_count=singleton_count,
    )

    elapsed = time.time() - start_time
    print(f"\n=== Completed in {elapsed:.2f} seconds ===\n")
    print("JSON Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_sample_pair_construction()
