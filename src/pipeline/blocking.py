"""Candidate pair generation and blocking module for entity resolution.

Implements inverted-index blocking with disk sharding to handle millions of records.
Generates candidate pairs between Source 1 and (Source 2, Source 3).
"""

import gc
import hashlib
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from src.utils.config import OUTPUTS_DIR


CHAR_NGRAM_SIZE = 3
WORD_NGRAM_SIZE = 2


def generate_char_ngrams(text: str, n: int = CHAR_NGRAM_SIZE) -> List[str]:
    if not text or len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def generate_word_ngrams(text: str, n: int = WORD_NGRAM_SIZE) -> List[str]:
    words = text.split()
    if len(words) < n:
        return ["_".join(words)] if words else []
    return ["_".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def get_blocking_keys(
    cleaned_name: str,
    cleaned_country: str,
    cleaned_address: str,
) -> List[str]:
    """Generate blocking keys for a single entity."""
    keys = []
    has_name = bool(cleaned_name)
    has_country = bool(cleaned_country)
    has_address = bool(cleaned_address)

    if has_name:
        # N-grams
        char_ngrams = generate_char_ngrams(cleaned_name)
        for ng in char_ngrams[:20]:
            keys.append(f"char_{ng}")
        word_ngrams = generate_word_ngrams(cleaned_name)
        for ng in word_ngrams[:10]:
            keys.append(f"word_{ng}")
            
        # Legal suffix
        legal_suffixes = [
            "limited", "incorporated", "corporation", "company", "llc", 
            "gmbh", "sarl", "sa", "srl", "spa", "bv", "nv", 
            "private limited", "public limited company", 
            "limited liability company", "limited liability partnership", 
            "proprietary limited"
        ]
        for suffix in legal_suffixes:
            if cleaned_name.endswith(f" {suffix}") or cleaned_name == suffix:
                keys.append(f"leg_{suffix}")
                break
                
        # Composite
        first_token = cleaned_name.split()[0] if cleaned_name.split() else ""
        if first_token and has_country:
            keys.append(f"comp_{cleaned_country}_{first_token}")

    if has_country:
        keys.append(f"ctry_{cleaned_country}")
        if has_name and char_ngrams:
            keys.append(f"ctrychar_{cleaned_country}_{char_ngrams[0]}")
            if len(char_ngrams) > 1:
                keys.append(f"ctrychar_{cleaned_country}_{char_ngrams[1]}")

    if has_address:
        postal_match = re.search(r"\b([1-9][0-9]{4,5})\b", cleaned_address)
        if postal_match:
            keys.append(f"post_{postal_match.group(1)}")

    return keys


def get_shard_id(key: str, num_shards: int) -> int:
    """Hash a blocking key to a shard ID."""
    return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16) % num_shards


def extract_keys_vectorized(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Extract blocking keys for a chunk of records."""
    records = []
    
    # Fix 3: fast iteration with itertuples over rows
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        
        # Handle fallback if cleaned columns are missing
        entity_id = row_dict.get("cleaned_entity_id", row_dict.get("entity_id"))
        if pd.isna(entity_id):
            continue
            
        name = row_dict.get("cleaned_name", "")
        if pd.isna(name): name = ""
        
        country = row_dict.get("cleaned_country", "")
        if pd.isna(country): country = ""
        
        address = row_dict.get("cleaned_address", "")
        if pd.isna(address): address = ""
        
        bkeys = get_blocking_keys(str(name), str(country), str(address))
        
        # Deduplicate keys per entity
        for bk in set(bkeys):
            records.append((bk, entity_id, source_name, name))
            
    return pd.DataFrame(records, columns=["blocking_key", "entity_id", "source", "name"])


def build_shards(
    cleaned_dir: Path, 
    shard_dir: Path, 
    source_files: Optional[List[str]] = None,
    chunksize: int = 200_000, 
    num_shards: int = 32
):
    """Read parquet files in chunks and write out key->entity associations to shards."""
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)
    
    for i in range(num_shards):
        (shard_dir / f"shard_{i}").mkdir()
        
    if source_files:
        parquet_files = [cleaned_dir / f for f in source_files]
    else:
        parquet_files = sorted(cleaned_dir.glob("*_cleaned.parquet"))

    for pf in parquet_files:
        if not pf.exists():
            continue
            
        source_name = pf.stem.replace("_cleaned", "")
        print(f"[INFO] Sharding {source_name}...")
        
        # Fix 1 & 2: True chunked reading and sharding out to disk
        pf_reader = pq.ParquetFile(pf)
        for chunk_idx, batch in enumerate(pf_reader.iter_batches(batch_size=chunksize)):
            df = batch.to_pandas()
            keys_df = extract_keys_vectorized(df, source_name)
            
            if keys_df.empty:
                continue
                
            keys_df["shard_id"] = keys_df["blocking_key"].apply(lambda k: get_shard_id(k, num_shards))
            
            for shard_id, group in keys_df.groupby("shard_id"):
                out_path = shard_dir / f"shard_{shard_id}" / f"{source_name}_chunk_{chunk_idx}.parquet"
                group.drop(columns=["shard_id"]).to_parquet(out_path, index=False)
                
            del df, keys_df
            gc.collect()


def generate_cross_source_pairs(
    s1_df: pd.DataFrame, 
    s2_df: pd.DataFrame, 
    max_pairs: int
) -> List[Tuple[str, str]]:
    """Generate candidate pairs from S1 to S2/S3 with oversized block handling."""
    if s1_df.empty or s2_df.empty:
        return []
    
    s1_records = s1_df[["entity_id", "name"]].to_dict("records")
    s2_records = s2_df[["entity_id", "name"]].to_dict("records")
    
    total_possible = len(s1_records) * len(s2_records)
    pairs = []
    
    if total_possible <= max_pairs:
        # Standard: emit all cross pairs
        for r1 in s1_records:
            for r2 in s2_records:
                # Fix 6: entity_id_1 is strictly source_1
                pairs.append((r1["entity_id"], r2["entity_id"]))
    else:
        # Fix 4: RapidFuzz token_sort_ratio for oversized blocks
        # Keep top-K matches per S1 record
        k_per_s1 = max(1, max_pairs // len(s1_records))
        for r1 in s1_records:
            name1 = r1["name"]
            scored = []
            for r2 in s2_records:
                score = fuzz.token_sort_ratio(name1, r2["name"])
                scored.append((score, r2["entity_id"]))
                
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, e2_id in scored[:k_per_s1]:
                pairs.append((r1["entity_id"], e2_id))
                
    return pairs


def process_shard(shard_path: Path, max_pairs_per_block: int = 100_000) -> pd.DataFrame:
    """Read a shard, group by blocking key, and generate cross-source pairs."""
    files = list(shard_path.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["entity_id_1", "entity_id_2"])
    
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    all_pairs = []
    
    for _, group in df.groupby("blocking_key"):
        # Fix 5: We only care about matching source_1 against others
        s1 = group[group["source"].str.contains("source1|source_1")]
        if s1.empty:
            continue
            
        s2 = group[group["source"].str.contains("source2|source_2")]
        s3 = group[group["source"].str.contains("source3|source_3")]
        
        if not s2.empty:
            all_pairs.extend(generate_cross_source_pairs(s1, s2, max_pairs_per_block))
        if not s3.empty:
            all_pairs.extend(generate_cross_source_pairs(s1, s3, max_pairs_per_block))
            
    return pd.DataFrame(all_pairs, columns=["entity_id_1", "entity_id_2"])


def run_blocking(
    cleaned_dir: Path = OUTPUTS_DIR / "cleaned",
    output_dir: Path = OUTPUTS_DIR / "blocked",
    source_files: Optional[List[str]] = None,
    chunksize: int = 200_000,
    max_pairs_per_block: int = 100_000,
    num_shards: int = 32,
) -> Path:
    """Run the blocking pipeline using a disk-sharded inverted index."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    
    print("[INFO] Building shards from cleaned data...")
    build_shards(cleaned_dir, shard_dir, source_files, chunksize, num_shards)
    
    print("[INFO] Processing shards and generating pairs...")
    all_pairs_files = []
    for i in range(num_shards):
        pairs_df = process_shard(shard_dir / f"shard_{i}", max_pairs_per_block)
        if not pairs_df.empty:
            out_file = output_dir / f"pairs_shard_{i}.parquet"
            pairs_df.to_parquet(out_file, index=False)
            all_pairs_files.append(out_file)
            
    print("[INFO] Deduplicating cross-shard pairs...")
    if all_pairs_files:
        df_all = pd.concat([pd.read_parquet(f) for f in all_pairs_files], ignore_index=True)
        # Drop fully identical pairs
        df_all.drop_duplicates(inplace=True)
        
        final_out = output_dir / "candidate_pairs.parquet"
        df_all.to_parquet(final_out, index=False)
        
        # Cleanup
        shutil.rmtree(shard_dir)
        for f in all_pairs_files:
            f.unlink()
            
        mb = final_out.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Wrote {len(df_all):,} unique candidate pairs ({mb:.2f} MB)")
        print(f"[INFO] Time elapsed: {time.time() - t0:.2f}s")
        return final_out
    else:
        print("[WARNING] No candidate pairs generated.")
        return output_dir / "candidate_pairs.parquet"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: Blocking (Disk-Sharded)")
    parser.add_argument("--cleaned-dir", type=Path, default=OUTPUTS_DIR / "cleaned")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "blocked")
    parser.add_argument("--source-files", nargs="*", default=None)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-pairs", type=int, default=100_000)
    parser.add_argument("--shards", type=int, default=32)
    args = parser.parse_args()

    run_blocking(
        cleaned_dir=args.cleaned_dir,
        output_dir=args.output_dir,
        source_files=args.source_files,
        chunksize=args.chunksize,
        max_pairs_per_block=args.max_pairs,
        num_shards=args.shards
    )


if __name__ == "__main__":
    main()
