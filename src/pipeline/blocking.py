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
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
        # Fix 4: Removed `keys.append(f"ctry_{cleaned_country}")` to prevent pathological oversized blocks
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
    
    for row in df.itertuples(index=False):
        # Avoid relying on the dynamically generated namedtuple returned by
        # ``itertuples``; type checkers can interpret ``_asdict`` as a column
        # value when column names overlap with namedtuple members.
        row_dict = dict(zip(df.columns, row))
        
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
    s1_data: Union[pd.DataFrame, List[Tuple[str, str]]], 
    s2_data: Union[pd.DataFrame, List[Tuple[str, str]]], 
    max_pairs: int,
    stats: Dict[str, int],
    blocking_key: str = ""
) -> List[Tuple[str, str]]:
    """Generate candidate pairs from S1 to S2/S3 with oversized block handling."""
    if s1_data is None or s2_data is None:
        return []
    
    if isinstance(s1_data, pd.DataFrame):
        if s1_data.empty:
            return []
        s1_records = [(r["entity_id"], r.get("name", "")) for r in s1_data.to_dict("records")]
    else:
        if not s1_data:
            return []
        s1_records = s1_data

    if isinstance(s2_data, pd.DataFrame):
        if s2_data.empty:
            return []
        s2_records = [(r["entity_id"], r.get("name", "")) for r in s2_data.to_dict("records")]
    else:
        if not s2_data:
            return []
        s2_records = s2_data
    
    total_possible = len(s1_records) * len(s2_records)
    pairs = []
    
    if total_possible <= max_pairs:
        for r1 in s1_records:
            eid1 = r1[0]
            for r2 in s2_records:
                pairs.append((eid1, r2[0]))
    else:
        # Strict ceiling on max_pairs for oversized blocks
        stats["oversized_blocks"] += 1
        print(f"      [WARN] Oversized block:\n  key={blocking_key}\n  S1={len(s1_records)}\n  S2={len(s2_records)}\n  potential={total_possible}\n  limit={max_pairs}")
        
        if len(s1_records) > max_pairs:
            s1_records = s1_records[:max_pairs]
            
        k_per_s1 = max_pairs // len(s1_records)
        remainder = max_pairs % len(s1_records)
        
        emitted = 0
        n_s2 = len(s2_records)
        for i, r1 in enumerate(s1_records):
            add_count = k_per_s1 + (1 if i < remainder else 0)
            eid1 = r1[0]
            
            for j in range(add_count):
                if emitted >= max_pairs:
                    break
                idx = (j * n_s2) // add_count if add_count > 0 else 0
                idx = min(idx, n_s2 - 1)
                pairs.append((eid1, s2_records[idx][0]))
                emitted += 1
                
    return pairs


def process_shard(shard_path: Path, output_dir: Path, max_pairs_per_block: int, stats: Dict[str, int]) -> int:
    """Read a shard, group by blocking key (streaming out-of-core), and generate cross-source pairs to disk."""
    print(f"  [INFO] Processing {shard_path.name}...")
    files = list(shard_path.glob("*.parquet"))
    if not files:
        return 0
    
    # Read all shard files with pandas and group by blocking_key
    # Avoids DuckDB/PyArrow interoperability issues (TProtocolException: Invalid data)
    dfs = [pd.read_parquet(f) for f in files]
    if not dfs:
        return 0
    shard_df = pd.concat(dfs, ignore_index=True)
    
    all_pairs = []
    total_shard_pairs = 0
    chunk_idx = 0
    
    for blocking_key, group in shard_df.groupby("blocking_key", sort=False):
        stats["blocks_processed"] += 1
        eids = group["entity_id"].tolist()
        srcs = group["source"].tolist()
        names = group["name"].tolist()
        if len(eids) < 2:
            continue
            
        prefix_map: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        for eid, src, name in zip(eids, srcs, names):
            pfx = src.split("_")[0]
            if pfx not in prefix_map:
                prefix_map[pfx] = {"s1": [], "s2": [], "s3": []}
            if "source1" in src or "source_1" in src:
                prefix_map[pfx]["s1"].append((eid, name))
            elif "source2" in src or "source_2" in src:
                prefix_map[pfx]["s2"].append((eid, name))
            elif "source3" in src or "source_3" in src:
                prefix_map[pfx]["s3"].append((eid, name))
                
        for pfx, grp in prefix_map.items():
            s1 = grp["s1"]
            if not s1:
                continue
            s2 = grp["s2"]
            s3 = grp["s3"]
            if s2:
                p = generate_cross_source_pairs(s1, s2, max_pairs_per_block, stats, blocking_key)
                stats["s1s2_pairs"] += len(p)
                all_pairs.extend(p)
            if s3:
                p = generate_cross_source_pairs(s1, s3, max_pairs_per_block, stats, blocking_key)
                stats["s1s3_pairs"] += len(p)
                all_pairs.extend(p)
                
        # Flush if memory is growing
        if len(all_pairs) >= 500_000:
            pdf = pd.DataFrame(all_pairs, columns=["entity_id_1", "entity_id_2"])
            out_chunk = output_dir / f"pairs_{shard_path.name}_{chunk_idx}.parquet"
            pdf.to_parquet(out_chunk, index=False)
            total_shard_pairs += len(pdf)
            all_pairs = []
            chunk_idx += 1
            
    if all_pairs:
        pdf = pd.DataFrame(all_pairs, columns=["entity_id_1", "entity_id_2"])
        out_chunk = output_dir / f"pairs_{shard_path.name}_{chunk_idx}.parquet"
        pdf.to_parquet(out_chunk, index=False)
        total_shard_pairs += len(pdf)
        
    return total_shard_pairs


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_blocking(
    cleaned_dir: Path = OUTPUTS_DIR / "cleaned",
    output_dir: Path = OUTPUTS_DIR / "blocked",
    source_files: Optional[List[str]] = None,
    chunksize: int = 200_000,
    max_pairs_per_block: int = 100_000,
    num_shards: int = 32,
) -> Path:
    """Run the blocking pipeline using a disk-sharded inverted index."""
    t_start = time.time()
    mem_start = get_memory_mb()
    peak_mem = mem_start
    
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    
    print("[INFO] Building shards from cleaned data...")
    t_shard_start = time.time()
    build_shards(cleaned_dir, shard_dir, source_files, chunksize, num_shards)
    t_shard_end = time.time()
    peak_mem = max(peak_mem, get_memory_mb())
    
    print(f"[STATS] Sharding completed in {t_shard_end - t_shard_start:.2f}s")
    
    print("[INFO] Processing shards and generating pairs...")
    t_pair_start = time.time()
    total_pairs = 0
    
    stats = {
        "s1s2_pairs": 0,
        "s1s3_pairs": 0,
        "s2s3_pairs": 0,
        "blocks_processed": 0,
        "oversized_blocks": 0
    }
    
    for i in range(num_shards):
        shard_path = shard_dir / f"shard_{i}"
        if not shard_path.exists():
            continue
        shard_pairs = process_shard(shard_path, output_dir, max_pairs_per_block, stats)
        total_pairs += shard_pairs
        peak_mem = max(peak_mem, get_memory_mb())
            
    t_pair_end = time.time()
    print(f"[STATS] Pair generation completed in {t_pair_end - t_pair_start:.2f}s")
    
    print("[INFO] Deduplicating cross-shard pairs...")
    t_dedup_start = time.time()
    
    final_out = output_dir / "candidate_pairs.parquet"
    pair_files = list(output_dir.glob("pairs_shard_*.parquet"))
    
    if pair_files:
        import duckdb
        # Fix 2: Disk-safe out-of-core deduplication via DuckDB instead of pd.concat
        con = duckdb.connect()
        files_pattern = str(output_dir / "pairs_shard_*.parquet").replace("\\", "/")
        final_out_str = str(final_out).replace("\\", "/")
        
        con.execute(f"COPY (SELECT DISTINCT entity_id_1, entity_id_2 FROM read_parquet('{files_pattern}')) TO '{final_out_str}' (FORMAT PARQUET)")
        
        count_row = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{final_out_str}')"
        ).fetchone()
        final_pairs = count_row[0] if count_row is not None else 0
        con.close()
        
        peak_mem = max(peak_mem, get_memory_mb())
        
        # Cleanup
        shutil.rmtree(shard_dir)
        for f in pair_files:
            f.unlink()
            
        mb = final_out.stat().st_size / (1024 * 1024)
        t_total = time.time() - t_start
        
        print("\n" + "="*50)
        print(" PHASE 3: BLOCKING EMPIRICAL STATS ")
        print("="*50)
        print(f" Total Time:          {t_total:.2f} s")
        print(f" Sharding Time:       {t_shard_end - t_shard_start:.2f} s")
        print(f" Pair Gen Time:       {t_pair_end - t_pair_start:.2f} s")
        print(f" Dedup Time:          {time.time() - t_dedup_start:.2f} s")
        print(f" Peak Memory (RSS):   {peak_mem:.2f} MB")
        print(f" Blocks Processed:    {stats['blocks_processed']:,}")
        print(f" Oversized Blocks:    {stats['oversized_blocks']:,}")
        print(f" S1-S2 Pairs Gen:     {stats['s1s2_pairs']:,}")
        print(f" S1-S3 Pairs Gen:     {stats['s1s3_pairs']:,}")
        print(f" S2-S3 Pairs Gen:     {stats['s2s3_pairs']:,}  (Should be 0)")
        print(f" Raw Pairs Generated: {total_pairs:,}")
        print(f" Final Unique Pairs:  {final_pairs:,}")
        if (t_pair_end - t_pair_start) > 0:
            print(f" Throughput:          {final_pairs / (t_pair_end - t_pair_start):,.0f} pairs/sec")
        print(f" Output File Size:    {mb:.2f} MB")
        print("="*50 + "\n")
        
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
