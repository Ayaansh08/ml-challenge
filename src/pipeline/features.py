"""Feature engineering module for pairwise similarity computation.

Computes fuzzy matching, exact match, and embedding-based features
on candidate pairs from blocking for the LightGBM classifier (Phase 5).
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein


def levenshtein_ratio(s1: Optional[str], s2: Optional[str]) -> float:
    """Normalized Levenshtein similarity ratio [0, 1]."""
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = Levenshtein.distance(s1, s2)
    return 1.0 - (dist / max_len)


def jaro_winkler_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """Jaro-Winkler similarity [0, 1] with prefix scaling."""
    if not s1 or not s2:
        return 0.0
    return fuzz.WRatio(s1, s2) / 100.0


def token_set_ratio(s1: Optional[str], s2: Optional[str]) -> float:
    """Token set ratio for fuzzy matching [0, 1]."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


def token_sort_ratio(s1: Optional[str], s2: Optional[str]) -> float:
    """Token sort ratio for fuzzy matching [0, 1]."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def partial_ratio(s1: Optional[str], s2: Optional[str]) -> float:
    """Partial ratio for substring matching [0, 1]."""
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1, s2) / 100.0


def exact_match(s1: Optional[str], s2: Optional[str]) -> int:
    """Binary exact match flag after normalization."""
    if not s1 or not s2:
        return 0
    return int(s1.strip().lower() == s2.strip().lower())


def prefix_match(s1: Optional[str], s2: Optional[str], n: int = 3) -> int:
    """Binary flag: first n tokens match exactly."""
    if not s1 or not s2:
        return 0
    t1 = s1.strip().lower().split()[:n]
    t2 = s2.strip().lower().split()[:n]
    return int(t1 == t2 and len(t1) > 0)


def common_token_count(s1: Optional[str], s2: Optional[str]) -> int:
    """Count of common tokens between two strings."""
    if not s1 or not s2:
        return 0
    set1 = set(s1.strip().lower().split())
    set2 = set(s2.strip().lower().split())
    return len(set1 & set2)


def jaccard_token_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """Jaccard similarity on token sets [0, 1]."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.strip().lower().split())
    set2 = set(s2.strip().lower().split())
    union = set1 | set2
    if not union:
        return 1.0
    return len(set1 & set2) / len(union)


def cosine_token_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """Cosine similarity on token frequency vectors [0, 1]."""
    if not s1 or not s2:
        return 0.0
    from collections import Counter
    c1 = Counter(s1.strip().lower().split())
    c2 = Counter(s2.strip().lower().split())
    dot = sum(c1[t] * c2[t] for t in c1 if t in c2)
    norm1 = sum(v * v for v in c1.values()) ** 0.5
    norm2 = sum(v * v for v in c2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def address_postal_match(addr1: Optional[str], addr2: Optional[str]) -> int:
    """Binary flag: both addresses contain the same postal code."""
    if not addr1 or not addr2:
        return 0
    post1 = set(re.findall(r"\b([1-9][0-9]{4,5})\b", addr1))
    post2 = set(re.findall(r"\b([1-9][0-9]{4,5})\b", addr2))
    return int(bool(post1 & post2))


def address_street_number_match(addr1: Optional[str], addr2: Optional[str]) -> int:
    """Binary flag: both addresses contain the same street/unit number."""
    if not addr1 or not addr2:
        return 0
    num1 = set(re.findall(r"\b([0-9]+[a-zA-Z]?(?:[/-][0-9]+[a-zA-Z]?)?)\b", addr1))
    num2 = set(re.findall(r"\b([0-9]+[a-zA-Z]?(?:[/-][0-9]+[a-zA-Z]?)?)\b", addr2))
    return int(bool(num1 & num2))


def address_landmark_overlap(addr1: Optional[str], addr2: Optional[str]) -> float:
    """Token overlap ratio for landmark phrases in addresses."""
    if not addr1 or not addr2:
        return 0.0
    lm1 = set(re.findall(r"\b(?:near|opp|opposite|behind|next to|adjacent to|in front of|beside)\s+([^,;]+)", addr1, re.IGNORECASE))
    lm2 = set(re.findall(r"\b(?:near|opp|opposite|behind|next to|adjacent to|in front of|beside)\s+([^,;]+)", addr2, re.IGNORECASE))
    if not lm1 or not lm2:
        return 0.0
    # Flatten landmark phrases into tokens
    tokens1 = set(" ".join(lm1).lower().split())
    tokens2 = set(" ".join(lm2).lower().split())
    union = tokens1 | tokens2
    if not union:
        return 0.0
    return len(tokens1 & tokens2) / len(union)


def country_match(c1: Optional[str], c2: Optional[str]) -> int:
    """Binary flag: countries match exactly (case-insensitive)."""
    if not c1 or not c2:
        return 0
    return int(c1.strip().upper() == c2.strip().upper())


def is_bare_domain_flag(name: Optional[str]) -> int:
    """Binary flag: name is a bare domain."""
    if not name:
        return 0
    return int(bool(re.match(r"^(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.)+(com|net|org|co\.in|in|io|ai|biz|info|edu|gov|me|app|dev|tech|store|online|co|uk|us|ca|de|fr|au|jp|cn|sg|hk|ae|sa|eu|xyz|site)(?:\/[^\s]*)?$", name, re.IGNORECASE)))


def legal_suffix_match(name1: Optional[str], name2: Optional[str]) -> int:
    """Binary flag: both names share the same legal suffix type."""
    if not name1 or not name2:
        return 0
    suffixes = [
        "limited", "incorporated", "corporation", "company", "llc",
        "gmbh", "sarl", "sa", "srl", "spa", "bv", "nv",
        "private limited", "public limited company",
        "limited liability company", "limited liability partnership",
        "proprietary limited"
    ]
    suf1 = next((s for s in suffixes if name1.endswith(f" {s}") or name1 == s), None)
    suf2 = next((s for s in suffixes if name2.endswith(f" {s}") or name2 == s), None)
    return int(suf1 is not None and suf1 == suf2)


def name_length_ratio(s1: Optional[str], s2: Optional[str]) -> float:
    """Ratio of shorter to longer name length [0, 1]."""
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    return min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0


def compute_pair_features(
    row1: pd.Series,
    row2: pd.Series,
    embedding_sim: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all pairwise similarity features for a candidate pair.

    Parameters
    ----------
    row1 : pd.Series
        First entity record with cleaned fields.
    row2 : pd.Series
        Second entity record with cleaned fields.
    embedding_sim : Optional[float]
        Precomputed bi-encoder cosine similarity if available.

    Returns
    -------
    Dict[str, float]
        Dictionary of feature name -> value.
    """
    def _clean_field(val: Any) -> str:
        if val is None or pd.isna(val):
            return ""
        s = str(val).strip()
        return "" if s.lower() in ("nan", "none", "null") else s

    name1 = _clean_field(row1.get("cleaned_name"))
    name2 = _clean_field(row2.get("cleaned_name"))
    addr1 = _clean_field(row1.get("cleaned_address"))
    addr2 = _clean_field(row2.get("cleaned_address"))
    country1 = _clean_field(row1.get("cleaned_country"))
    country2 = _clean_field(row2.get("cleaned_country"))

    features = {}

    # Name similarity features
    features["name_levenshtein"] = levenshtein_ratio(name1, name2)
    features["name_jaro_winkler"] = jaro_winkler_similarity(name1, name2)
    features["name_token_set_ratio"] = token_set_ratio(name1, name2)
    features["name_token_sort_ratio"] = token_sort_ratio(name1, name2)
    features["name_partial_ratio"] = partial_ratio(name1, name2)
    features["name_exact_match"] = exact_match(name1, name2)
    features["name_prefix3_match"] = prefix_match(name1, name2, 3)
    features["name_prefix2_match"] = prefix_match(name1, name2, 2)
    features["name_common_tokens"] = common_token_count(name1, name2)
    features["name_jaccard"] = jaccard_token_similarity(name1, name2)
    features["name_cosine"] = cosine_token_similarity(name1, name2)
    features["name_length_ratio"] = name_length_ratio(name1, name2)
    features["name_legal_suffix_match"] = legal_suffix_match(name1, name2)
    features["name_bare_domain_1"] = is_bare_domain_flag(name1)
    features["name_bare_domain_2"] = is_bare_domain_flag(name2)

    # Address similarity features
    features["addr_levenshtein"] = levenshtein_ratio(addr1, addr2)
    features["addr_jaro_winkler"] = jaro_winkler_similarity(addr1, addr2)
    features["addr_token_set_ratio"] = token_set_ratio(addr1, addr2)
    features["addr_exact_match"] = exact_match(addr1, addr2)
    features["addr_postal_match"] = address_postal_match(addr1, addr2)
    features["addr_street_number_match"] = address_street_number_match(addr1, addr2)
    features["addr_landmark_overlap"] = address_landmark_overlap(addr1, addr2)
    features["addr_jaccard"] = jaccard_token_similarity(addr1, addr2)

    # Country match
    features["country_match"] = country_match(country1, country2)

    # Composite features
    features["name_addr_both_exact"] = int(
        features["name_exact_match"] == 1 and features["addr_exact_match"] == 1
    )
    features["name_country_both_match"] = int(
        features["name_exact_match"] == 1 and features["country_match"] == 1
    )

    # Embedding similarity (if provided)
    if embedding_sim is not None:
        features["embedding_cosine"] = embedding_sim
    else:
        features["embedding_cosine"] = 0.0

    return features


def compute_features_batch(
    pairs_df: pd.DataFrame,
    combined_lookup: pd.DataFrame,
    embedding_sims: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute features for a batch of candidate pairs.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame with columns [entity_id_1, entity_id_2] for candidate pairs.
    cleaned_s1 : pd.DataFrame
        Cleaned Source 1 data.
    cleaned_s2 : pd.DataFrame
        Cleaned Source 2 data.
    cleaned_s3 : pd.DataFrame
        Cleaned Source 3 data.
    embedding_sims : Optional[np.ndarray]
        Precomputed embedding similarities aligned with pairs_df rows.
    id_col : str
        Column name for entity ID in cleaned DataFrames.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per candidate pair.
    """
    recs1 = combined_lookup.reindex(pairs_df["entity_id_1"].values)
    recs2 = combined_lookup.reindex(pairs_df["entity_id_2"].values)

    recs1_dicts = recs1.to_dict("records")
    recs2_dicts = recs2.to_dict("records")

    feature_rows = []
    for i in range(len(pairs_df)):
        r1 = recs1_dicts[i]
        r2 = recs2_dicts[i]

        if pd.isna(r1.get("cleaned_name")) and pd.isna(r1.get("cleaned_address")) and pd.isna(r1.get("cleaned_country")):
            continue
        if pd.isna(r2.get("cleaned_name")) and pd.isna(r2.get("cleaned_address")) and pd.isna(r2.get("cleaned_country")):
            continue

        emb_sim = embedding_sims[i] if embedding_sims is not None else None
        feats = compute_pair_features(r1, r2, emb_sim)
        feats["entity_id_1"] = pairs_df.iloc[i]["entity_id_1"]
        feats["entity_id_2"] = pairs_df.iloc[i]["entity_id_2"]
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


def run_feature_engineering(
    candidate_pairs_path: str,
    cleaned_dir: str,
    output_path: str,
    embedding_sims_path: Optional[str] = None,
    chunksize: int = 100_000,
) -> Path:
    """Run feature engineering on candidate pairs from blocking.

    Parameters
    ----------
    candidate_pairs_path : str
        Path to candidate_pairs.parquet from Phase 3 blocking.
    cleaned_dir : str
        Directory containing *_cleaned.parquet files.
    output_path : str
        Output path for feature matrix parquet.
    embedding_sims_path : Optional[str]
        Optional path to precomputed embedding similarities (numpy .npy).
    chunksize : int
        Chunk size for memory-bounded processing.

    Returns
    -------
    Path
        Path to the output feature matrix parquet.
    """
    from pathlib import Path
    import pyarrow.parquet as pq

    cleaned_path = Path(cleaned_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cleaned data
    if cleaned_path.name in ("train", "test"):
        search_path = cleaned_path.parent
    else:
        search_path = cleaned_path

    s1_files = list(search_path.rglob("*source1*_cleaned.parquet"))
    s2_files = list(search_path.rglob("*source2*_cleaned.parquet"))
    s3_files = list(search_path.rglob("*source3*_cleaned.parquet"))

    if not s1_files or not s2_files or not s3_files:
        raise FileNotFoundError("Missing cleaned parquet files for one or more sources")

    print("[INFO] Loading cleaned sources into memory-efficient DataFrame...")
    cols = ["cleaned_entity_id", "cleaned_name", "cleaned_address", "cleaned_country"]
    
    s1_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s1_files], ignore_index=True)
    s2_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s2_files], ignore_index=True)
    s3_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s3_files], ignore_index=True)

    print("[INFO] Building unified lookup DataFrame...")
    combined_lookup = pd.concat([s1_df, s2_df, s3_df], ignore_index=True).set_index("cleaned_entity_id")

    # Load candidate pairs
    pairs_pf = pq.ParquetFile(candidate_pairs_path)
    total_pairs = pairs_pf.metadata.num_rows
    print(f"[INFO] Total candidate pairs: {total_pairs:,}")

    # Load embedding similarities if provided
    embedding_sims = None
    if embedding_sims_path and Path(embedding_sims_path).exists():
        embedding_sims = np.load(embedding_sims_path)
        print(f"[INFO] Loaded embedding similarities: {embedding_sims.shape}")

    # Process in chunks and stream to disk
    emb_idx = 0
    import pyarrow as pa
    writer = None
    total_written = 0
    num_cols = 0

    try:
        for batch_idx, batch in enumerate(pairs_pf.iter_batches(batch_size=chunksize)):
            pairs_chunk = batch.to_pandas()
            
            chunk_emb = None
            if embedding_sims is not None:
                chunk_end = min(emb_idx + len(pairs_chunk), len(embedding_sims))
                chunk_emb = embedding_sims[emb_idx:chunk_end]
                emb_idx = chunk_end

            print(f"[INFO] Computing features for chunk {batch_idx + 1} ({len(pairs_chunk):,} pairs)...")
            features_df = compute_features_batch(
                pairs_chunk, combined_lookup, chunk_emb
            )
            
            if features_df.empty:
                continue
                
            table = pa.Table.from_pandas(features_df, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
                num_cols = len(features_df.columns)
                
            writer.write_table(table)
            total_written += len(features_df)
            break
    finally:
        if writer is not None:
            writer.close()

    if total_written > 0:
        print(f"[INFO] Saved feature matrix: {out_path} ({total_written:,} rows, {num_cols} features)")
    else:
        print("[WARNING] No features computed")

    return out_path


if __name__ == "__main__":
    import argparse
    from src.utils.config import OUTPUTS_DIR

    parser = argparse.ArgumentParser(description="Phase 4: Feature Engineering")
    parser.add_argument("--candidate-pairs", type=Path, default=OUTPUTS_DIR / "blocked" / "candidate_pairs.parquet")
    parser.add_argument("--cleaned-dir", type=Path, default=OUTPUTS_DIR / "cleaned")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "features" / "features.parquet")
    parser.add_argument("--embedding-sims", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    run_feature_engineering(
        candidate_pairs_path=str(args.candidate_pairs),
        cleaned_dir=str(args.cleaned_dir),
        output_path=str(args.output),
        embedding_sims_path=str(args.embedding_sims) if args.embedding_sims else None,
        chunksize=args.chunksize,
    )