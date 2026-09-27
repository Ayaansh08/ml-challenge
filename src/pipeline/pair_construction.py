"""Pair construction and dataset generation module for Model 2 (Bi-Encoder + Cross-Encoder).

Pure functions designed for entity resolution pair generation:
1. `load_and_clean_sources`: Ingests S1, S2, S3 TSVs and applies `clean_dataframe`.
2. `build_positive_pairs`: Maps ground-truth matches into positive training pairs.
3. `grouped_train_val_split`: Performs entity-grouped train/val partitioning.
4. `build_country_stratified_proxy_split`: Produces France-proxy normalized evaluation sets.
5. `mine_hard_negatives`: Cosine-similarity hard negative mining scaffold.
6. `write_pair_dataset`: Writes Parquet datasets and summary metrics.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.pipeline.cleaning import clean_dataframe, normalize_missing, normalize_whitespace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# French-Specific Synonym & Street Token Dictionaries (Local to module)
# ---------------------------------------------------------------------------

FRENCH_LEGAL_SYNONYMS: List[Tuple[str, str]] = [
    (r"\bs\.?a\.?r\.?l\.?\b|\bsoci[eé]t[eé]\s+[aà]\s+responsabilit[eé]\s+limit[eé]e\b", "societe a responsabilite limitee"),
    (r"\bs\.?a\.?s\.?u\.?\b|\bsoci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e\s+unipersonnelle\b", "societe par actions simplifiee unipersonnelle"),
    (r"\bs\.?a\.?s\.?\b|\bsoci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e\b", "societe par actions simplifiee"),
    (r"\bs\.?a\.?\b|\bsoci[eé]t[eé]\s+anonyme\b", "societe anonyme"),
    (r"\be\.?u\.?r\.?l\.?\b|\bentreprise\s+unipersonnelle\s+[aà]\s+responsabilit[eé]\s+limit[eé]e\b", "entreprise unipersonnelle a responsabilite limitee"),
    (r"\bs\.?c\.?i\.?\b|\bsoci[eé]t[eé]\s+civile\s+immobili[eè]re\b", "societe civile immobiliere"),
    (r"\bs\.?n\.?c\.?\b|\bsoci[eé]t[eé]\s+en\s+nom\s+collectif\b", "societe en nom collectif"),
    (r"\bg\.?i\.?e\.?\b|\bgroupement\s+d['’]int[eé]r[eê]t\s+[eé]conomique\b", "groupement d interet economique"),
    (r"\bassoc\.?\b|\bassociation\b", "association"),
]

FRENCH_ADDRESS_TOKENS: List[Tuple[str, str]] = [
    (r"\bbvd\.?\b|\bbd\.?\b", "boulevard"),
    (r"\bav\.?\b|\bave\.?\b", "avenue"),
    (r"\br\.?\b|\brte\.?\b", "route"),
    (r"\ball\.?\b|\ball[eé]e\.?\b", "allee"),
    (r"\bpl\.?\b|\bplc\.?\b", "place"),
    (r"\bch\.?\b|\bche\.?\b", "chemin"),
    (r"\bimp\.?\b", "impasse"),
    (r"\bst\.?\b", "saint"),
    (r"\bste\.?\b", "sainte"),
    (r"\bfaubourg\b|\bfbg\.?\b", "faubourg"),
    (r"\bsq\.?\b", "square"),
    (r"\brue\b", "rue"),
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def format_entity_text(cleaned_name: Optional[str], cleaned_address: Optional[str]) -> str:
    """Format cleaned entity fields into model input string: '{cleaned_name} [SEP] {cleaned_address}'."""
    name_part = str(cleaned_name).strip() if pd.notna(cleaned_name) else ""
    addr_part = str(cleaned_address).strip() if pd.notna(cleaned_address) else ""

    if name_part and addr_part:
        return f"{name_part} [SEP] {addr_part}"
    if name_part:
        return name_part
    if addr_part:
        return addr_part
    return ""


def apply_french_proxy_normalization(name: Optional[str], address: Optional[str]) -> str:
    """Apply French-specific corporate legal and address expansions for proxy validation."""
    clean_n = normalize_whitespace(normalize_missing(name)) or ""
    clean_a = normalize_whitespace(normalize_missing(address)) or ""

    if clean_n:
        clean_n = unicodedata.normalize("NFKC", clean_n).casefold()
        for pat, repl in FRENCH_LEGAL_SYNONYMS:
            clean_n = re.sub(pat, repl, clean_n, flags=re.IGNORECASE)
        clean_n = normalize_whitespace(clean_n) or ""

    if clean_a:
        clean_a = unicodedata.normalize("NFKC", clean_a).casefold()
        for pat, repl in FRENCH_ADDRESS_TOKENS:
            clean_a = re.sub(pat, repl, clean_a, flags=re.IGNORECASE)
        clean_a = normalize_whitespace(clean_a) or ""

    return format_entity_text(clean_n, clean_a)


# ---------------------------------------------------------------------------
# Core Pipeline Pure Functions
# ---------------------------------------------------------------------------


def load_and_clean_sources(
    s1_path: Union[str, Path],
    s2_path: Union[str, Path],
    s3_path: Union[str, Path],
    nrows: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Load S1, S2, and S3 TSV files, apply clean_dataframe, and validate entity_id prefixes.

    Parameters
    ----------
    s1_path : Union[str, Path]
        Path to Source 1 TSV.
    s2_path : Union[str, Path]
        Path to Source 2 TSV.
    s3_path : Union[str, Path]
        Path to Source 3 TSV.
    nrows : Optional[int], default None
        Optional limit on number of rows to load per file for fast sampling/testing.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary keyed "s1", "s2", "s3" with cleaned DataFrames.

    Raises
    ------
    ValueError
        If entity_id values in any source do not match the expected prefix (S1-, S2-, S3-).
    """
    sources_spec = [
        ("s1", Path(s1_path), "S1-"),
        ("s2", Path(s2_path), "S2-"),
        ("s3", Path(s3_path), "S3-"),
    ]

    cleaned_dict: Dict[str, pd.DataFrame] = {}

    for key, path, expected_prefix in sources_spec:
        if not path.is_file():
            raise FileNotFoundError(f"Source file not found at: {path.resolve()}")

        raw_df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
        cleaned_df = clean_dataframe(raw_df)

        # Validate entity_id prefixes
        id_col = "cleaned_entity_id" if "cleaned_entity_id" in cleaned_df.columns else "entity_id"
        invalid_mask = ~cleaned_df[id_col].fillna("").str.startswith(expected_prefix)
        invalid_rows = cleaned_df[invalid_mask]

        if not invalid_rows.empty:
            offending_ids = invalid_rows[id_col].head(10).tolist()
            total_offending = len(invalid_rows)
            raise ValueError(
                f"Prefix validation failed for source '{key}' in '{path.name}'. "
                f"Expected all entity IDs to start with '{expected_prefix}'. "
                f"Found {total_offending} offending rows. Sample offending IDs: {offending_ids}"
            )

        cleaned_dict[key] = cleaned_df

    return cleaned_dict


def build_positive_pairs(
    ground_truth_path: Union[str, Path],
    cleaned_s1: pd.DataFrame,
    cleaned_s2: pd.DataFrame,
    cleaned_s3: pd.DataFrame,
    filter_to_s1_ids: bool = True,
) -> pd.DataFrame:
    """Construct positive matching pairs from ground truth and cleaned source DataFrames.

    Parameters
    ----------
    ground_truth_path : Union[str, Path]
        Path to the ground truth TSV file (source1_entity_id, matched_entity_ids).
    cleaned_s1 : pd.DataFrame
        Cleaned S1 DataFrame.
    cleaned_s2 : pd.DataFrame
        Cleaned S2 DataFrame.
    cleaned_s3 : pd.DataFrame
        Cleaned S3 DataFrame.
    filter_to_s1_ids : bool, default True
        Whether to filter ground truth to only S1 entity IDs present in cleaned_s1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [s1_entity_id, match_entity_id, source, s1_text, match_text].
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found at: {gt_path.resolve()}")

    # Load ground truth
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)

    # Standardize column naming
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
    matches_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else gt_df.columns[1]

    gt_df[s1_col] = gt_df[s1_col].astype(str).map(normalize_whitespace)
    gt_df[matches_col] = gt_df[matches_col].astype(str).map(normalize_missing)

    # Build lookup dictionaries for fast O(1) text retrieval
    s1_text_map: Dict[str, str] = {}
    for _, row in cleaned_s1.iterrows():
        eid = row.get("cleaned_entity_id") or row.get("entity_id")
        if eid:
            s1_text_map[eid] = format_entity_text(row.get("cleaned_name"), row.get("cleaned_address"))

    s2_text_map: Dict[str, str] = {}
    for _, row in cleaned_s2.iterrows():
        eid = row.get("cleaned_entity_id") or row.get("entity_id")
        if eid:
            s2_text_map[eid] = format_entity_text(row.get("cleaned_name"), row.get("cleaned_address"))

    s3_text_map: Dict[str, str] = {}
    for _, row in cleaned_s3.iterrows():
        eid = row.get("cleaned_entity_id") or row.get("entity_id")
        if eid:
            s3_text_map[eid] = format_entity_text(row.get("cleaned_name"), row.get("cleaned_address"))

    # Optionally filter GT to S1 entities in cleaned_s1
    if filter_to_s1_ids:
        gt_df = gt_df[gt_df[s1_col].isin(s1_text_map)].copy()

    records: List[Dict[str, str]] = []
    singleton_count = 0
    total_gt_rows = len(gt_df)

    for _, row in gt_df.iterrows():
        s1_id = row[s1_col]
        raw_matches = row[matches_col]

        if not raw_matches or pd.isna(raw_matches):
            singleton_count += 1
            continue

        match_ids = [m.strip() for m in raw_matches.split(",") if m.strip()]
        if not match_ids:
            singleton_count += 1
            continue

        s1_text = s1_text_map.get(s1_id, "")
        for match_id in match_ids:
            if match_id.startswith("S2-"):
                src = "s2"
                match_text = s2_text_map.get(match_id, "")
            elif match_id.startswith("S3-"):
                src = "s3"
                match_text = s3_text_map.get(match_id, "")
            else:
                src = "unknown"
                match_text = s2_text_map.get(match_id, s3_text_map.get(match_id, ""))

            records.append({
                "s1_entity_id": s1_id,
                "match_entity_id": match_id,
                "source": src,
                "s1_text": s1_text,
                "match_text": match_text,
            })

    logger.info(
        f"Built {len(records):,} positive pairs across {total_gt_rows - singleton_count:,} non-singleton S1 entities. "
        f"Skipped {singleton_count:,} singletons ({singleton_count / max(total_gt_rows, 1) * 100:.2f}% of GT)."
    )

    columns = ["s1_entity_id", "match_entity_id", "source", "s1_text", "match_text"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns)


def grouped_train_val_split(
    positive_pairs_df: pd.DataFrame,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split positive pairs by unique s1_entity_id so no entity's pairs span both splits.

    Parameters
    ----------
    positive_pairs_df : pd.DataFrame
        Positive pair DataFrame containing 's1_entity_id'.
    val_fraction : float, default 0.15
        Proportion of S1 entity groups allocated to validation.
    random_seed : int, default 42
        Random seed for reproducible partitioning.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, val_df) partitioned dataframes.

    Raises
    ------
    AssertionError
        If there is any s1_entity_id overlap between train and val splits.
    """
    if positive_pairs_df.empty:
        return positive_pairs_df.copy(), positive_pairs_df.copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_seed)
    groups = positive_pairs_df["s1_entity_id"].values
    train_idx, val_idx = next(gss.split(positive_pairs_df, groups=groups))

    train_df = positive_pairs_df.iloc[train_idx].reset_index(drop=True)
    val_df = positive_pairs_df.iloc[val_idx].reset_index(drop=True)

    # Assert strict zero entity overlap
    train_entities: Set[str] = set(train_df["s1_entity_id"].unique())
    val_entities: Set[str] = set(val_df["s1_entity_id"].unique())
    overlap = train_entities.intersection(val_entities)

    if overlap:
        raise AssertionError(
            f"Grouped train/val split assertion failed: {len(overlap)} S1 entity IDs "
            f"overlap across train and validation sets! Sample overlaps: {list(overlap)[:5]}"
        )

    logger.info(
        f"Grouped split complete: Train pairs={len(train_df):,} ({len(train_entities):,} unique S1) | "
        f"Val pairs={len(val_df):,} ({len(val_entities):,} unique S1)"
    )
    return train_df, val_df


def build_country_stratified_proxy_split(
    cleaned_s1: pd.DataFrame,
    val_entity_ids: Union[Set[str], List[str], pd.Series],
    target_countries: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Construct France-proxy validation set by applying French legal/street expansions to US/IN records.

    Parameters
    ----------
    cleaned_s1 : pd.DataFrame
        Cleaned S1 DataFrame.
    val_entity_ids : Union[Set[str], List[str], pd.Series]
        Set or collection of validation S1 entity IDs.
    target_countries : Optional[List[str]], default None
        Countries to include as France proxy candidates (defaults to ['US', 'USA', 'IN', 'IND', 'INDIA']).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [entity_id, country, raw_name, raw_address, normal_text, france_proxy_text].
    """
    if target_countries is None:
        target_countries = ["US", "USA", "IN", "IND", "INDIA"]
    target_set = {c.upper() for c in target_countries}

    val_set = set(val_entity_ids)
    id_col = "cleaned_entity_id" if "cleaned_entity_id" in cleaned_s1.columns else "entity_id"

    # Filter to validation entity IDs
    val_s1 = cleaned_s1[cleaned_s1[id_col].isin(val_set)].copy()

    # Filter to target proxy countries
    country_col = "cleaned_country" if "cleaned_country" in val_s1.columns else "country"
    proxy_candidates = val_s1[val_s1[country_col].fillna("").str.upper().isin(target_set)].copy()

    records: List[Dict[str, Any]] = []
    for _, row in proxy_candidates.iterrows():
        eid = row.get(id_col)
        cntry = row.get(country_col)
        raw_n = row.get("business_name")
        raw_a = row.get("business_address")
        clean_n = row.get("cleaned_name")
        clean_a = row.get("cleaned_address")

        normal_text = format_entity_text(clean_n, clean_a)
        france_proxy_text = apply_french_proxy_normalization(raw_n, raw_a)

        records.append({
            "entity_id": eid,
            "country": cntry,
            "raw_name": raw_n,
            "raw_address": raw_a,
            "normal_text": normal_text,
            "france_proxy_text": france_proxy_text,
        })

    cols = ["entity_id", "country", "raw_name", "raw_address", "normal_text", "france_proxy_text"]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records, columns=cols)


def mine_hard_negatives(
    anchor_entity_id: str,
    anchor_text: str,
    candidate_pool_df: pd.DataFrame,
    current_embeddings: np.ndarray,
    anchor_embedding: Optional[np.ndarray] = None,
    anchor_idx: Optional[int] = None,
    known_positive_ids: Optional[Set[str]] = None,
    k: int = 6,
) -> List[Dict[str, Any]]:
    """Mine top-k hard negative candidate records by cosine similarity from precomputed embeddings.

    Parameters
    ----------
    anchor_entity_id : str
        Anchor S1 entity ID.
    anchor_text : str
        Anchor representation text.
    candidate_pool_df : pd.DataFrame
        Candidate pool DataFrame with entity IDs and texts.
    current_embeddings : np.ndarray
        Precomputed candidate pool embedding matrix (shape: [N, D]).
    anchor_embedding : Optional[np.ndarray], default None
        Embedding vector for the anchor entity (shape: [D] or [1, D]).
    anchor_idx : Optional[int], default None
        Index of anchor in current_embeddings if anchor is part of candidate pool.
    known_positive_ids : Optional[Set[str]], default None
        Set of true positive entity IDs for this anchor to exclude from negatives.
    k : int, default 6
        Number of hard negatives to retrieve.

    Returns
    -------
    List[Dict[str, Any]]
        List of top-k hard negative dictionaries with entity ID, text, and cosine score.
    """
    if candidate_pool_df.empty or len(current_embeddings) == 0:
        return []

    pos_set = set(known_positive_ids) if known_positive_ids else set()

    # Extract anchor embedding vector
    if anchor_embedding is not None:
        anchor_vec = np.asarray(anchor_embedding, dtype=np.float32).reshape(1, -1)
    elif anchor_idx is not None and 0 <= anchor_idx < len(current_embeddings):
        anchor_vec = current_embeddings[anchor_idx : anchor_idx + 1]
    else:
        raise ValueError("Must provide either `anchor_embedding` vector or valid `anchor_idx`.")

    # Compute normalized cosine similarities
    cand_norms = np.linalg.norm(current_embeddings, axis=1, keepdims=True) + 1e-10
    anchor_norm = np.linalg.norm(anchor_vec, axis=1, keepdims=True) + 1e-10

    normed_cands = current_embeddings / cand_norms
    normed_anchor = anchor_vec / anchor_norm

    sims = np.dot(normed_cands, normed_anchor.T).flatten()

    id_col = "match_entity_id" if "match_entity_id" in candidate_pool_df.columns else "entity_id"
    if id_col not in candidate_pool_df.columns and "cleaned_entity_id" in candidate_pool_df.columns:
        id_col = "cleaned_entity_id"

    text_col = "match_text" if "match_text" in candidate_pool_df.columns else "text"
    if text_col not in candidate_pool_df.columns and "s1_text" in candidate_pool_df.columns:
        text_col = "s1_text"

    # Sort candidates by descending similarity
    ranked_indices = np.argsort(-sims)

    negatives: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        cand_row = candidate_pool_df.iloc[idx]
        cand_id = cand_row.get(id_col)
        cand_text = cand_row.get(text_col, "")

        # Exclude anchor itself and known true positive matches
        if cand_id == anchor_entity_id or cand_id in pos_set:
            continue

        negatives.append({
            "anchor_entity_id": anchor_entity_id,
            "anchor_text": anchor_text,
            "negative_entity_id": cand_id,
            "negative_text": cand_text,
            "similarity_score": float(sims[idx]),
        })

        if len(negatives) >= k:
            break

    return negatives


def write_pair_dataset(
    positive_pairs_df: pd.DataFrame,
    output_dir: Union[str, Path],
    val_fraction: float = 0.15,
    random_seed: int = 42,
    cleaned_s1: Optional[pd.DataFrame] = None,
    singleton_count: int = 0,
) -> Dict[str, Any]:
    """Split positive pairs, write train/val Parquet files, generate France proxy, and output JSON summary.

    Parameters
    ----------
    positive_pairs_df : pd.DataFrame
        Full positive pairs DataFrame.
    output_dir : Union[str, Path]
        Target directory to write datasets.
    val_fraction : float, default 0.15
        Fraction of S1 entities allocated to validation.
    random_seed : int, default 42
        Random seed for reproducibility.
    cleaned_s1 : Optional[pd.DataFrame], default None
        Cleaned S1 DataFrame for France-proxy generation and country breakdown.
    singleton_count : int, default 0
        Count of ground truth singletons skipped.

    Returns
    -------
    Dict[str, Any]
        JSON summary statistics dictionary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df, val_df = grouped_train_val_split(
        positive_pairs_df=positive_pairs_df,
        val_fraction=val_fraction,
        random_seed=random_seed,
    )

    train_parquet = out_path / "train_positives.parquet"
    val_parquet = out_path / "val_positives.parquet"

    train_df.to_parquet(train_parquet, index=False, engine="pyarrow")
    val_df.to_parquet(val_parquet, index=False, engine="pyarrow")

    # France proxy validation set
    france_proxy_count = 0
    if cleaned_s1 is not None and not val_df.empty:
        val_eids = set(val_df["s1_entity_id"].unique())
        france_proxy_df = build_country_stratified_proxy_split(
            cleaned_s1=cleaned_s1,
            val_entity_ids=val_eids,
        )
        france_parquet = out_path / "france_proxy_val.parquet"
        france_proxy_df.to_parquet(france_parquet, index=False, engine="pyarrow")
        france_proxy_count = len(france_proxy_df)

    # Compute breakdown statistics
    train_src_counts = train_df["source"].value_counts().to_dict() if not train_df.empty else {}
    val_src_counts = val_df["source"].value_counts().to_dict() if not val_df.empty else {}

    country_breakdown: Dict[str, int] = {}
    if cleaned_s1 is not None:
        id_col = "cleaned_entity_id" if "cleaned_entity_id" in cleaned_s1.columns else "entity_id"
        cntry_col = "cleaned_country" if "cleaned_country" in cleaned_s1.columns else "country"
        if cntry_col in cleaned_s1.columns:
            cntry_map = dict(zip(cleaned_s1[id_col], cleaned_s1[cntry_col].fillna("UNKNOWN")))
            if not positive_pairs_df.empty:
                s1_countries = positive_pairs_df["s1_entity_id"].map(cntry_map).fillna("UNKNOWN")
                country_breakdown = s1_countries.value_counts().to_dict()

    summary: Dict[str, Any] = {
        "dataset_summary": {
            "total_positive_pairs": len(positive_pairs_df),
            "train_positive_pairs": len(train_df),
            "val_positive_pairs": len(val_df),
            "singleton_s1_entities_skipped": singleton_count,
            "unique_s1_entities_total": int(positive_pairs_df["s1_entity_id"].nunique()) if not positive_pairs_df.empty else 0,
            "unique_s1_entities_train": int(train_df["s1_entity_id"].nunique()) if not train_df.empty else 0,
            "unique_s1_entities_val": int(val_df["s1_entity_id"].nunique()) if not val_df.empty else 0,
            "france_proxy_val_entities": france_proxy_count,
            "val_fraction_requested": val_fraction,
            "random_seed": random_seed,
        },
        "source_breakdown": {
            "train": train_src_counts,
            "val": val_src_counts,
        },
        "country_breakdown": country_breakdown,
    }

    summary_file = out_path / "pair_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
