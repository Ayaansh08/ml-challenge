"""End-to-end integration runner for the Business Entity Resolution Pipeline.

Chains together all 6 phases:
1. Ingestion / Data Cleaning (cleaning.py)
2. Inverted-Index Blocking (blocking.py)
3. Pair Construction & Feature Engineering (features.py)
4. Matcher Model Training & Test Prediction (matching.py)
5. Aggregation, Transitive Closure & Final Submission (aggregation.py)

Supports full-scale dataset execution as well as sampled subsets (via --nrows)
for rapid end-to-end integration verification.
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline.aggregation import load_all_entity_ids, run_aggregation
from src.pipeline.blocking import run_blocking
from src.pipeline.cleaning import clean_dataframe
from src.pipeline.features import run_feature_engineering
from src.pipeline.matching import (
    load_model,
    predict_matches,
    prepare_training_data,
    save_model,
    train_matcher,
)
from src.pipeline.run_cleaning import process_file_in_chunks
from src.utils.config import DATA_DIR, OUTPUTS_DIR

logger = logging.getLogger("run_pipeline")


def clean_source_file(
    file_path: Path,
    output_file: Path,
    nrows: Optional[int] = None,
    chunksize: int = 200_000,
    force: bool = False,
) -> int:
    """Clean a single raw TSV file to a Parquet output file."""
    if not force and output_file.exists() and output_file.stat().st_size > 0:
        logger.info(
            "Skipping '%s' — output '%s' already exists (%0.2f MB).",
            file_path.name,
            output_file.name,
            output_file.stat().st_size / (1024 * 1024),
        )
        # Read row count from existing parquet metadata
        return pq.ParquetFile(output_file).metadata.num_rows

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if nrows is not None:
        logger.info("Reading first %d rows of '%s'...", nrows, file_path.name)
        raw_df = pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            nrows=nrows,
            low_memory=False,
            encoding="utf-8",
        )
        cleaned_df = clean_dataframe(raw_df)
        cleaned_df.to_parquet(output_file, index=False)
        total_rows = len(cleaned_df)
        logger.info(
            "Wrote %d cleaned sample records for '%s' to '%s'.",
            total_rows,
            file_path.name,
            output_file.name,
        )
        return total_rows
    else:
        logger.info("Streaming '%s' in chunks of %d...", file_path.name, chunksize)
        return process_file_in_chunks(
            file_path=file_path,
            output_file=output_file,
            chunksize=chunksize,
        )


def run_cleaning_stage(
    train_dir: Path,
    test_dir: Path,
    cleaned_dir: Path,
    nrows: Optional[int] = None,
    chunksize: int = 200_000,
    force: bool = False,
) -> Dict[str, int]:
    """Execute Phase 2 cleaning on both train and test sources."""
    logger.info("=" * 60)
    logger.info("STAGE 1 & 2: INGESTION AND CLEANING")
    logger.info("=" * 60)

    train_out_dir = cleaned_dir / "train"
    test_out_dir = cleaned_dir / "test"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}

    # Clean train sources (source1, source2, source3)
    train_sources = sorted(train_dir.glob("train_source*.tsv"))
    if not train_sources:
        raise FileNotFoundError(f"No train_source*.tsv found in {train_dir}")

    for src in train_sources:
        out_file = train_out_dir / f"{src.stem}_cleaned.parquet"
        counts[src.name] = clean_source_file(
            file_path=src,
            output_file=out_file,
            nrows=nrows,
            chunksize=chunksize,
            force=force,
        )

    # Clean test sources (source1, source2, source3)
    test_sources = sorted(test_dir.glob("test_source*.tsv"))
    if not test_sources:
        raise FileNotFoundError(f"No test_source*.tsv found in {test_dir}")

    for src in test_sources:
        out_file = test_out_dir / f"{src.stem}_cleaned.parquet"
        counts[src.name] = clean_source_file(
            file_path=src,
            output_file=out_file,
            nrows=nrows,
            chunksize=chunksize,
            force=force,
        )

    return counts


def run_blocking_stage(
    cleaned_dir: Path,
    blocked_dir: Path,
    num_shards: int = 16,
    max_pairs_per_block: int = 100_000,
    chunksize: int = 200_000,
    force: bool = False,
) -> Tuple[Path, Path]:
    """Execute Phase 3 blocking separately for train and test splits."""
    logger.info("=" * 60)
    logger.info("STAGE 3: BLOCKING & CANDIDATE PAIR GENERATION")
    logger.info("=" * 60)

    train_cleaned = cleaned_dir / "train"
    test_cleaned = cleaned_dir / "test"
    train_blocked = blocked_dir / "train"
    test_blocked = blocked_dir / "test"

    train_pairs_path = train_blocked / "candidate_pairs.parquet"
    test_pairs_path = test_blocked / "candidate_pairs.parquet"

    # Train blocking
    if not force and train_pairs_path.exists() and train_pairs_path.stat().st_size > 0:
        logger.info("Train candidate pairs already exist: %s", train_pairs_path)
    else:
        logger.info("Running blocking on train sources...")
        train_pairs_path = run_blocking(
            cleaned_dir=train_cleaned,
            output_dir=train_blocked,
            chunksize=chunksize,
            max_pairs_per_block=max_pairs_per_block,
            num_shards=num_shards,
        )

    # Test blocking
    if not force and test_pairs_path.exists() and test_pairs_path.stat().st_size > 0:
        logger.info("Test candidate pairs already exist: %s", test_pairs_path)
    else:
        logger.info("Running blocking on test sources...")
        test_pairs_path = run_blocking(
            cleaned_dir=test_cleaned,
            output_dir=test_blocked,
            chunksize=chunksize,
            max_pairs_per_block=max_pairs_per_block,
            num_shards=num_shards,
        )

    return train_pairs_path, test_pairs_path


def run_features_stage(
    train_pairs_path: Path,
    test_pairs_path: Path,
    cleaned_dir: Path,
    features_dir: Path,
    chunksize: int = 100_000,
    force: bool = False,
) -> Tuple[Path, Path]:
    """Execute Phase 4 feature engineering for candidate pairs."""
    logger.info("=" * 60)
    logger.info("STAGE 4: FEATURE ENGINEERING")
    logger.info("=" * 60)

    features_dir.mkdir(parents=True, exist_ok=True)
    train_features_path = features_dir / "train_features.parquet"
    test_features_path = features_dir / "test_features.parquet"

    # Train features
    if not force and train_features_path.exists() and train_features_path.stat().st_size > 0:
        logger.info("Train feature matrix already exists: %s", train_features_path)
    else:
        logger.info("Computing features for train candidate pairs...")
        run_feature_engineering(
            candidate_pairs_path=str(train_pairs_path),
            cleaned_dir=str(cleaned_dir / "train"),
            output_path=str(train_features_path),
            chunksize=chunksize,
        )

    # Test features
    if not force and test_features_path.exists() and test_features_path.stat().st_size > 0:
        logger.info("Test feature matrix already exists: %s", test_features_path)
    else:
        logger.info("Computing features for test candidate pairs...")
        run_feature_engineering(
            candidate_pairs_path=str(test_pairs_path),
            cleaned_dir=str(cleaned_dir / "test"),
            output_path=str(test_features_path),
            chunksize=chunksize,
        )

    return train_features_path, test_features_path


def run_matching_stage(
    train_features_path: Path,
    test_features_path: Path,
    ground_truth_path: Path,
    matching_dir: Path,
    threshold: float = 0.5,
    model_params: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Execute Phase 5: train LightGBM classifier on train and predict matches on test."""
    logger.info("=" * 60)
    logger.info("STAGE 5: MATCHING & CLASSIFICATION (LightGBM)")
    logger.info("=" * 60)

    matching_dir.mkdir(parents=True, exist_ok=True)
    model_base_path = matching_dir / "matcher_model"
    test_preds_path = matching_dir / "test_predictions.parquet"
    metrics_path = matching_dir / "match_metrics.json"

    # Load or train model
    model = None
    metrics = {}
    if not force and (model_base_path.with_suffix(".txt").exists() or model_base_path.with_suffix(".pkl").exists()):
        logger.info("Loading existing matcher model from %s...", model_base_path)
        model = load_model(model_base_path)
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
    else:
        logger.info("Loading train features from %s...", train_features_path)
        train_features_df = pd.read_parquet(train_features_path)
        logger.info("Loaded %d train candidate pairs.", len(train_features_df))

        logger.info("Preparing training labels using ground truth: %s...", ground_truth_path)
        X_train, y_train = prepare_training_data(train_features_df, ground_truth_path)

        pos_count = int(y_train.sum())
        logger.info(
            "Training label distribution: %d positive, %d negative.",
            pos_count,
            len(y_train) - pos_count,
        )

        if pos_count == 0:
            raise ValueError(
                "No positive match pairs found between train candidate pairs and ground truth! "
                "Ensure sufficient nrows or valid ground truth alignment."
            )

        logger.info("Training LightGBM matcher...")
        model, metrics = train_matcher(X_train, y_train, params=model_params)
        save_model(model, model_base_path)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved model metrics to %s.", metrics_path)

    # Predict on test features
    logger.info("Predicting matches on test candidate pairs: %s...", test_features_path)
    if test_features_path.exists() and test_features_path.stat().st_size > 0:
        test_features_df = pd.read_parquet(test_features_path)
        if len(test_features_df) > 0:
            test_preds_df = predict_matches(model, test_features_df, threshold=threshold)
        else:
            test_preds_df = pd.DataFrame(columns=["entity_id_1", "entity_id_2", "score", "prediction"])
    else:
        test_preds_df = pd.DataFrame(columns=["entity_id_1", "entity_id_2", "score", "prediction"])

    test_preds_df.to_parquet(test_preds_path, index=False)
    pred_matches = int((test_preds_df["prediction"] == 1).sum()) if "prediction" in test_preds_df.columns else 0
    logger.info(
        "Test predictions saved to %s (%d candidate pairs, %d predicted matches at threshold=%.2f).",
        test_preds_path,
        len(test_preds_df),
        pred_matches,
        threshold,
    )

    return test_preds_path, metrics


def run_aggregation_stage(
    test_preds_path: Path,
    cleaned_test_dir: Path,
    submission_path: Path,
    threshold: float = 0.5,
) -> Path:
    """Execute Phase 6: aggregation, transitive closure, and submission generation."""
    logger.info("=" * 60)
    logger.info("STAGE 6: AGGREGATION & SUBMISSION")
    logger.info("=" * 60)

    logger.info("Aggregating test predictions into clusters: %s -> %s...", test_preds_path, submission_path)
    out_path = run_aggregation(
        predictions_path=test_preds_path,
        entities_dir=cleaned_test_dir,
        output_path=submission_path,
        threshold=threshold,
    )
    return out_path


def validate_submission_integrity(submission_path: Path, cleaned_test_dir: Path) -> Dict[str, Any]:
    """Validate that the generated submission file conforms strictly to the requirements."""
    logger.info("=" * 60)
    logger.info("VALIDATING SUBMISSION INTEGRITY")
    logger.info("=" * 60)

    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file does not exist: {submission_path}")

    sub_df = pd.read_csv(submission_path, sep="\t", dtype=str)

    # 1. Check columns
    required_cols = ["entity_id", "cluster_id"]
    if list(sub_df.columns) != required_cols:
        raise ValueError(f"Submission columns mismatch! Expected {required_cols}, found {list(sub_df.columns)}")

    # 2. Check no nulls
    if sub_df["entity_id"].isna().any():
        raise ValueError("Submission contains null entity_id entries!")
    if sub_df["cluster_id"].isna().any():
        raise ValueError("Submission contains null cluster_id entries!")

    # 3. Check uniqueness of entity_id
    if not sub_df["entity_id"].is_unique:
        duplicates = sub_df[sub_df["entity_id"].duplicated()]["entity_id"].tolist()[:10]
        raise ValueError(f"Duplicate entity_ids detected in submission! Sample: {duplicates}")

    # 4. Check that all test entities are present
    expected_ids = set(load_all_entity_ids(cleaned_test_dir))
    sub_ids = set(sub_df["entity_id"])
    missing_ids = expected_ids - sub_ids
    extra_ids = sub_ids - expected_ids

    if missing_ids:
        raise ValueError(f"{len(missing_ids)} expected test entities missing from submission! Sample: {list(missing_ids)[:5]}")
    if extra_ids:
        raise ValueError(f"{len(extra_ids)} unexpected entities present in submission! Sample: {list(extra_ids)[:5]}")

    # 5. Cluster statistics
    cluster_counts = sub_df["cluster_id"].value_counts()
    n_singletons = int((cluster_counts == 1).sum())
    n_multi = int((cluster_counts > 1).sum())
    max_cluster_size = int(cluster_counts.max()) if len(cluster_counts) > 0 else 0

    stats = {
        "total_records": len(sub_df),
        "total_unique_clusters": int(sub_df["cluster_id"].nunique()),
        "singleton_clusters": n_singletons,
        "multi_entity_clusters": n_multi,
        "max_cluster_size": max_cluster_size,
        "file_size_bytes": submission_path.stat().st_size,
    }

    logger.info("Submission validation PASSED:")
    for k, v in stats.items():
        logger.info("  %s: %s", k, f"{v:,}" if isinstance(v, int) else v)

    return stats


def run_pipeline(
    train_dir: Path = DATA_DIR / "train",
    test_dir: Path = DATA_DIR / "test",
    output_dir: Path = OUTPUTS_DIR,
    nrows: Optional[int] = None,
    chunksize: int = 100_000,
    num_shards: int = 16,
    max_pairs_per_block: int = 100_000,
    threshold: float = 0.5,
    skip_cleaning: bool = False,
    skip_blocking: bool = False,
    skip_features: bool = False,
    skip_matching: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute the full entity resolution pipeline end-to-end.

    Parameters
    ----------
    train_dir : Path
        Directory containing raw train TSVs and train_ground_truth.tsv.
    test_dir : Path
        Directory containing raw test TSVs.
    output_dir : Path
        Base output directory for artifacts.
    nrows : Optional[int]
        Number of rows to sample from each TSV. If None, runs on full data.
    chunksize : int
        Chunk size for reading and processing.
    num_shards : int
        Number of shards for blocking index.
    max_pairs_per_block : int
        Maximum candidate pairs generated per blocking key.
    threshold : float
        Decision threshold for binary match predictions.
    skip_cleaning : bool
        If True, skip cleaning stage and use existing parquet files.
    skip_blocking : bool
        If True, skip blocking stage and use existing candidate pairs.
    skip_features : bool
        If True, skip featurization stage and use existing feature matrices.
    skip_matching : bool
        If True, skip model training and predict with existing model.
    force : bool
        If True, recompute all intermediate artifacts even if they exist.

    Returns
    -------
    Dict[str, Any]
        Summary report containing stage runtimes and validation metrics.
    """
    total_start_time = time.time()
    logger.info("Starting Business Entity Resolution Pipeline")
    logger.info("  Train directory:   %s", train_dir.resolve())
    logger.info("  Test directory:    %s", test_dir.resolve())
    logger.info("  Output directory:  %s", output_dir.resolve())
    logger.info("  Sample size:       %s", f"{nrows:,} rows" if nrows else "FULL DATASET")
    logger.info("  Blocking shards:   %d", num_shards)
    logger.info("  Max pairs/block:   %d", max_pairs_per_block)
    logger.info("  Threshold:         %.2f", threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = output_dir / "cleaned"
    blocked_dir = output_dir / "blocked"
    features_dir = output_dir / "features"
    matching_dir = output_dir / "matching"
    submission_path = output_dir / "submission.tsv"

    ground_truth_path = train_dir / "train_ground_truth.tsv"
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {ground_truth_path}")

    stage_times: Dict[str, float] = {}

    # Stage 1 & 2: Cleaning
    t0 = time.time()
    if not skip_cleaning:
        run_cleaning_stage(
            train_dir=train_dir,
            test_dir=test_dir,
            cleaned_dir=cleaned_dir,
            nrows=nrows,
            chunksize=chunksize,
            force=force,
        )
    stage_times["cleaning"] = time.time() - t0

    # Stage 3: Blocking
    t0 = time.time()
    if not skip_blocking:
        train_pairs_path, test_pairs_path = run_blocking_stage(
            cleaned_dir=cleaned_dir,
            blocked_dir=blocked_dir,
            num_shards=num_shards,
            max_pairs_per_block=max_pairs_per_block,
            chunksize=chunksize,
            force=force,
        )
    else:
        train_pairs_path = blocked_dir / "train" / "candidate_pairs.parquet"
        test_pairs_path = blocked_dir / "test" / "candidate_pairs.parquet"
    stage_times["blocking"] = time.time() - t0

    # Stage 4: Feature Engineering
    t0 = time.time()
    if not skip_features:
        train_feats_path, test_feats_path = run_features_stage(
            train_pairs_path=train_pairs_path,
            test_pairs_path=test_pairs_path,
            cleaned_dir=cleaned_dir,
            features_dir=features_dir,
            chunksize=chunksize,
            force=force,
        )
    else:
        train_feats_path = features_dir / "train_features.parquet"
        test_feats_path = features_dir / "test_features.parquet"
    stage_times["features"] = time.time() - t0

    # Stage 5: Matching / LightGBM
    t0 = time.time()
    test_preds_path, match_metrics = run_matching_stage(
        train_features_path=train_feats_path,
        test_features_path=test_feats_path,
        ground_truth_path=ground_truth_path,
        matching_dir=matching_dir,
        threshold=threshold,
        force=force,
    )
    stage_times["matching"] = time.time() - t0

    # Stage 6: Aggregation & Submission
    t0 = time.time()
    submission_file = run_aggregation_stage(
        test_preds_path=test_preds_path,
        cleaned_test_dir=cleaned_dir / "test",
        submission_path=submission_path,
        threshold=threshold,
    )
    stage_times["aggregation"] = time.time() - t0

    # Validation
    sub_stats = validate_submission_integrity(submission_file, cleaned_dir / "test")

    total_time = time.time() - total_start_time
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY IN %.2fs", total_time)
    logger.info("Stage timings:")
    for stage, elapsed in stage_times.items():
        logger.info("  %-15s: %8.2fs (%4.1f%%)", stage, elapsed, 100 * elapsed / max(total_time, 0.001))
    logger.info("Final submission written to: %s", submission_file.resolve())
    logger.info("=" * 60)

    return {
        "status": "success",
        "total_time_seconds": total_time,
        "stage_times": stage_times,
        "match_metrics": match_metrics,
        "submission_stats": sub_stats,
        "submission_file": str(submission_file.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end integration runner for business entity resolution pipeline."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DATA_DIR / "train",
        help="Directory containing raw train TSVs and train_ground_truth.tsv.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DATA_DIR / "test",
        help="Directory containing raw test TSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "pipeline_run",
        help="Target output directory for intermediate artifacts and submission.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row sampling limit (e.g. 50000) for fast integration verification.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Batch chunk size for streaming operations.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=16,
        help="Number of shards for blocking index.",
    )
    parser.add_argument(
        "--max-pairs-per-block",
        type=int,
        default=100_000,
        help="Ceiling on candidate pairs generated per blocking key.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Match classification decision threshold.",
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip cleaning stage and reuse existing cleaned parquet files.",
    )
    parser.add_argument(
        "--skip-blocking",
        action="store_true",
        help="Skip blocking stage and reuse existing candidate pairs.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering and reuse existing feature matrices.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of all stages even if outputs already exist.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        nrows=args.nrows,
        chunksize=args.chunksize,
        num_shards=args.num_shards,
        max_pairs_per_block=args.max_pairs_per_block,
        threshold=args.threshold,
        skip_cleaning=args.skip_cleaning,
        skip_blocking=args.skip_blocking,
        skip_features=args.skip_features,
        force=args.force,
    )


if __name__ == "__main__":
    main()
