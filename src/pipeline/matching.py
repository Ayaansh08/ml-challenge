"""Phase 5: LightGBM Matcher for Entity Resolution.

Trains and applies a binary classifier on pairwise feature matrices
to predict entity matches between sources.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils.config import DATA_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)

# Feature columns (excluding entity IDs)
FEATURE_COLUMNS = [
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_exact_match",
    "name_prefix3_match",
    "name_prefix2_match",
    "name_common_tokens",
    "name_jaccard",
    "name_cosine",
    "name_length_ratio",
    "name_legal_suffix_match",
    "name_bare_domain_1",
    "name_bare_domain_2",
    "addr_levenshtein",
    "addr_jaro_winkler",
    "addr_token_set_ratio",
    "addr_exact_match",
    "addr_postal_match",
    "addr_street_number_match",
    "addr_landmark_overlap",
    "addr_jaccard",
    "country_match",
    "name_addr_both_exact",
    "name_country_both_match",
    "embedding_cosine",
]

ID_COLUMNS = ["entity_id_1", "entity_id_2"]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature column names from DataFrame (excluding ID columns)."""
    return [c for c in df.columns if c not in ID_COLUMNS]


def prepare_training_data(
    features_df: pd.DataFrame,
    ground_truth_path: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build binary labels from ground truth and join with feature matrix.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with columns [entity_id_1, entity_id_2, ...features...].
    ground_truth_path : Union[str, Path]
        Path to train_ground_truth.tsv (columns: source1_entity_id, matched_entity_ids).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) where X is feature matrix (no ID columns) and y is binary labels.
    """
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found at: {gt_path}")

    # Load ground truth
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)

    # Build positive pair set
    positive_pairs = set()
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
    matches_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else gt_df.columns[1]

    # Optimization: Filter GT to relevant entity IDs in candidate pairs
    if "entity_id_1" in features_df.columns:
        relevant_ids = set(features_df["entity_id_1"])
        if "entity_id_2" in features_df.columns:
            relevant_ids.update(features_df["entity_id_2"])
        gt_df = gt_df[gt_df[s1_col].isin(relevant_ids)]

    for row in gt_df.itertuples(index=False):
        row_dict = dict(zip(gt_df.columns, row))
        s1_id = str(row_dict[s1_col]).strip()
        raw_matches = str(row_dict[matches_col]).strip()
        if not raw_matches or raw_matches.lower() in ("nan", "none", "null", ""):
            continue
        match_ids = [m.strip() for m in raw_matches.split(",") if m.strip()]
        for match_id in match_ids:
            # Order doesn't matter for matching, store both directions
            positive_pairs.add((s1_id, match_id))
            positive_pairs.add((match_id, s1_id))

    logger.info(f"Loaded {len(positive_pairs):,} positive pair records from ground truth")

    # Create labels for all candidate pairs
    def is_positive(row):
        pair = (row["entity_id_1"], row["entity_id_2"])
        return 1 if pair in positive_pairs else 0

    features_df = features_df.copy()
    labels = features_df.apply(is_positive, axis=1)

    pos_count = labels.sum()
    neg_count = len(labels) - pos_count
    logger.info(f"Labels: {pos_count:,} positive, {neg_count:,} negative (ratio: {neg_count/max(pos_count,1):.1f}:1)")

    # Extract feature columns (excluding ID columns and label)
    feature_cols = get_feature_columns(features_df)
    X = features_df[feature_cols]
    y = labels

    return X, y


def train_matcher(
    features_df: pd.DataFrame,
    labels: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """Train LightGBM binary classifier on feature matrix.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix (no ID columns).
    labels : pd.Series
        Binary labels (1=match, 0=non-match).
    params : Optional[Dict[str, Any]]
        LightGBM parameters. If None, uses defaults tuned for imbalanced classification.
    test_size : float, default 0.2
        Fraction for validation split.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[lgb.Booster, Dict[str, float]]
        (trained_model, metrics_dict with precision, recall, f1, auc_roc, accuracy)
    """
    if params is None:
        # Default params for imbalanced binary classification
        pos_count = labels.sum()
        neg_count = len(labels) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "scale_pos_weight": scale_pos_weight,
            "verbose": -1,
            "random_state": random_state,
            "n_jobs": -1,
        }

    # Stratified train/val split (safely fallback if class has < 2 samples)
    min_class_count = labels.value_counts().min() if hasattr(labels, "value_counts") else pd.Series(labels).value_counts().min()
    stratify_arg = labels if min_class_count >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        features_df, labels, test_size=test_size, stratify=stratify_arg, random_state=random_state
    )

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with early stopping
    logger.info(f"Training LightGBM: {len(X_train):,} train, {len(X_val):,} val samples")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    # Evaluate on validation set
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_val, y_pred_proba),
        "accuracy": accuracy_score(y_val, y_pred),
        "best_iteration": model.best_iteration,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "pos_train": int(y_train.sum()),
        "neg_train": int(len(y_train) - y_train.sum()),
        "pos_val": int(y_val.sum()),
        "neg_val": int(len(y_val) - y_val.sum()),
    }

    logger.info(f"Validation metrics: {metrics}")
    return model, metrics


def predict_matches(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict match probabilities and binary predictions on feature matrix.

    Parameters
    ----------
    model : lgb.Booster
        Trained LightGBM model.
    features_df : pd.DataFrame
        Feature matrix with ID columns [entity_id_1, entity_id_2].
    threshold : float, default 0.5
        Decision threshold for binary prediction.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [entity_id_1, entity_id_2, score, prediction].
    """
    X = features_df[get_feature_columns(features_df)]

    scores = model.predict(X, num_iteration=model.best_iteration)
    predictions = (scores >= threshold).astype(int)

    result = pd.DataFrame({
        "entity_id_1": features_df["entity_id_1"].values,
        "entity_id_2": features_df["entity_id_2"].values,
        "score": scores,
        "prediction": predictions,
    })

    return result


def save_model(model: lgb.Booster, path: Union[str, Path]) -> None:
    """Save LightGBM model to disk using native format + joblib wrapper."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path.with_suffix(".txt")))
    # Also save with joblib for sklearn compatibility
    import joblib
    joblib.dump(model, str(path.with_suffix(".pkl")))


def load_model(path: Union[str, Path]) -> lgb.Booster:
    """Load LightGBM model from disk."""
    path = Path(path)
    if path.with_suffix(".txt").exists():
        return lgb.Booster(model_file=str(path.with_suffix(".txt")))
    elif path.with_suffix(".pkl").exists():
        import joblib
        return joblib.load(str(path.with_suffix(".pkl")))
    else:
        raise FileNotFoundError(f"No model found at {path} (.txt or .pkl)")


def run_matching(
    features_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
    output_dir: Union[str, Path],
    threshold: float = 0.5,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Full matching pipeline: load features, build labels, train, evaluate, save.

    Parameters
    ----------
    features_path : Union[str, Path]
        Path to features.parquet from Phase 4.
    ground_truth_path : Union[str, Path]
        Path to train_ground_truth.tsv.
    output_dir : Union[str, Path]
        Output directory for model, predictions, and metrics.
    threshold : float, default 0.5
        Decision threshold for binary predictions.
    model_params : Optional[Dict[str, Any]]
        Optional LightGBM parameter overrides.

    Returns
    -------
    Dict[str, Any]
        Dictionary with model path, predictions path, metrics, and summary stats.
    """
    features_path = Path(features_path)
    ground_truth_path = Path(ground_truth_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading features from {features_path}...")
    features_df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(features_df):,} pairs with {len(get_feature_columns(features_df))} features")

    # Prepare labels from ground truth
    logger.info("Building labels from ground truth...")
    X, y = prepare_training_data(features_df, ground_truth_path)

    # Train model
    logger.info("Training LightGBM matcher...")
    model, metrics = train_matcher(X, y, params=model_params)

    # Save model
    model_path = output_dir / "matcher_model"
    save_model(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Predict on full feature matrix
    logger.info("Generating predictions on full feature matrix...")
    predictions_df = predict_matches(model, features_df, threshold=threshold)

    # Save predictions
    preds_path = output_dir / "match_predictions.parquet"
    predictions_df.to_parquet(preds_path, index=False)
    logger.info(f"Predictions saved to {preds_path} ({len(predictions_df):,} rows)")

    # Save metrics
    import json
    metrics_path = output_dir / "match_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Summary
    pred_pos = predictions_df["prediction"].sum()
    pred_neg = len(predictions_df) - pred_pos
    logger.info(f"Predictions: {pred_pos:,} matches, {pred_neg:,} non-matches")

    return {
        "model_path": str(model_path),
        "predictions_path": str(preds_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "num_predictions": len(predictions_df),
        "predicted_matches": int(pred_pos),
        "predicted_non_matches": int(pred_neg),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5: LightGBM Matcher")
    parser.add_argument("--features", type=Path, default=OUTPUTS_DIR / "features" / "features.parquet")
    parser.add_argument("--ground-truth", type=Path, default=DATA_DIR / "train" / "train_ground_truth.tsv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "matching")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-rounds", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    custom_params = {
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
    } if args.learning_rate != 0.05 or args.num_leaves != 63 else None

    run_matching(
        features_path=args.features,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        threshold=args.threshold,
        model_params=custom_params,
    )