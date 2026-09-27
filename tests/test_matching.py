"""Unit tests for Phase 5 LightGBM matching module."""

import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.pipeline.matching import (
    get_feature_columns,
    load_model,
    predict_matches,
    prepare_training_data,
    save_model,
    train_matcher,
)


class TestMatching(unittest.TestCase):
    """Test suite for LightGBM matcher functions."""

    def setUp(self):
        """Create synthetic feature data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_pos = 100

        # Create synthetic features
        self.features_df = pd.DataFrame({
            "entity_id_1": [f"S1-{i}" for i in range(n_samples)],
            "entity_id_2": [f"S2-{i}" for i in range(n_samples)],
            "name_levenshtein": np.random.beta(2, 5, n_samples).astype(float),
            "name_jaro_winkler": np.random.beta(2, 5, n_samples).astype(float),
            "name_token_set_ratio": np.random.beta(2, 5, n_samples).astype(float),
            "name_token_sort_ratio": np.random.beta(2, 5, n_samples).astype(float),
            "name_partial_ratio": np.random.beta(2, 5, n_samples).astype(float),
            "name_exact_match": np.random.binomial(1, 0.05, n_samples).astype(float),
            "name_prefix3_match": np.random.binomial(1, 0.1, n_samples).astype(float),
            "name_prefix2_match": np.random.binomial(1, 0.2, n_samples).astype(float),
            "name_common_tokens": np.random.poisson(1, n_samples).astype(float),
            "name_jaccard": np.random.beta(2, 5, n_samples).astype(float),
            "name_cosine": np.random.beta(2, 5, n_samples).astype(float),
            "name_length_ratio": np.random.beta(2, 5, n_samples).astype(float),
            "name_legal_suffix_match": np.random.binomial(1, 0.1, n_samples).astype(float),
            "name_bare_domain_1": np.random.binomial(1, 0.02, n_samples).astype(float),
            "name_bare_domain_2": np.random.binomial(1, 0.02, n_samples).astype(float),
            "addr_levenshtein": np.random.beta(2, 5, n_samples).astype(float),
            "addr_jaro_winkler": np.random.beta(2, 5, n_samples).astype(float),
            "addr_token_set_ratio": np.random.beta(2, 5, n_samples).astype(float),
            "addr_exact_match": np.random.binomial(1, 0.03, n_samples).astype(float),
            "addr_postal_match": np.random.binomial(1, 0.05, n_samples).astype(float),
            "addr_street_number_match": np.random.binomial(1, 0.05, n_samples).astype(float),
            "addr_landmark_overlap": np.random.beta(1, 10, n_samples).astype(float),
            "addr_jaccard": np.random.beta(2, 5, n_samples).astype(float),
            "country_match": np.random.binomial(1, 0.8, n_samples).astype(float),
            "name_addr_both_exact": np.random.binomial(1, 0.01, n_samples).astype(float),
            "name_country_both_match": np.random.binomial(1, 0.04, n_samples).astype(float),
            "embedding_cosine": np.random.beta(2, 5, n_samples).astype(float),
        })

        # Boost features for positive samples
        pos_idx = np.random.choice(n_samples, n_pos, replace=False)
        for col in ["name_levenshtein", "name_jaro_winkler", "name_token_set_ratio",
                    "name_exact_match", "addr_exact_match", "country_match"]:
            self.features_df.loc[pos_idx, col] = np.random.beta(5, 2, n_pos)

        # Create ground truth file
        self.gt_path = Path(tempfile.mktemp(suffix=".tsv"))
        gt_rows = []
        for idx in pos_idx:
            s1 = self.features_df.loc[idx, "entity_id_1"]
            s2 = self.features_df.loc[idx, "entity_id_2"]
            gt_rows.append(f"{s1}\t{s2}")
        with open(self.gt_path, "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("\n".join(gt_rows))

    def tearDown(self):
        """Clean up temp files."""
        if self.gt_path.exists():
            self.gt_path.unlink()

    def test_get_feature_columns(self):
        """Test feature column extraction excludes ID columns."""
        cols = get_feature_columns(self.features_df)
        self.assertNotIn("entity_id_1", cols)
        self.assertNotIn("entity_id_2", cols)
        self.assertIn("name_levenshtein", cols)
        self.assertIn("country_match", cols)
        self.assertEqual(len(cols), 27)  # 29 total - 2 ID columns

    def test_prepare_training_data(self):
        """Test label construction from ground truth."""
        X, y = prepare_training_data(self.features_df, self.gt_path)

        # Check shapes
        self.assertEqual(len(X), len(self.features_df))
        self.assertEqual(len(y), len(self.features_df))
        self.assertEqual(list(X.columns), get_feature_columns(self.features_df))

        # Check labels are binary
        self.assertTrue(set(y.unique()).issubset({0, 1}))

        # Check positive count matches ground truth
        self.assertEqual(y.sum(), 100)  # 100 positive pairs in ground truth

    def test_train_matcher(self):
        """Test LightGBM training on synthetic data."""
        X, y = prepare_training_data(self.features_df, self.gt_path)

        model, metrics = train_matcher(X, y, test_size=0.3, random_state=42)

        # Check model type
        self.assertIsInstance(model, lgb.Booster)

        # Check metrics keys
        expected_keys = {"precision", "recall", "f1", "auc_roc", "accuracy",
                         "best_iteration", "train_samples", "val_samples",
                         "pos_train", "neg_train", "pos_val", "neg_val"}
        self.assertEqual(set(metrics.keys()), expected_keys)

        # Check metric ranges
        for key in ["precision", "recall", "f1", "auc_roc", "accuracy"]:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)

        # Check best_iteration is positive
        self.assertGreater(metrics["best_iteration"], 0)

    def test_predict_matches(self):
        """Test prediction on feature matrix."""
        X, y = prepare_training_data(self.features_df, self.gt_path)
        model, _ = train_matcher(X, y, test_size=0.3, random_state=42)

        predictions = predict_matches(model, self.features_df, threshold=0.5)

        # Check output structure
        self.assertEqual(len(predictions), len(self.features_df))
        self.assertListEqual(list(predictions.columns), ["entity_id_1", "entity_id_2", "score", "prediction"])

        # Check score range
        self.assertTrue((predictions["score"] >= 0).all())
        self.assertTrue((predictions["score"] <= 1).all())

        # Check prediction is binary
        self.assertTrue(set(predictions["prediction"].unique()).issubset({0, 1}))

        # Check threshold behavior
        high_thresh = predict_matches(model, self.features_df, threshold=0.9)
        low_thresh = predict_matches(model, self.features_df, threshold=0.1)
        self.assertLessEqual(high_thresh["prediction"].sum(), low_thresh["prediction"].sum())

    def test_save_load_model(self):
        """Test model save/load roundtrip."""
        X, y = prepare_training_data(self.features_df, self.gt_path)
        model, _ = train_matcher(X, y, test_size=0.3, random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            save_model(model, model_path)

            # Check files created
            self.assertTrue(model_path.with_suffix(".txt").exists())
            self.assertTrue(model_path.with_suffix(".pkl").exists())

            # Load and verify predictions match
            loaded_model = load_model(model_path)
            self.assertIsInstance(loaded_model, lgb.Booster)

            pred_orig = predict_matches(model, self.features_df)
            pred_loaded = predict_matches(loaded_model, self.features_df)

            # Scores should be nearly identical
            np.testing.assert_allclose(pred_orig["score"].values, pred_loaded["score"].values, rtol=1e-5)

    def test_threshold_behavior(self):
        """Test that threshold correctly controls positive predictions."""
        X, y = prepare_training_data(self.features_df, self.gt_path)
        model, _ = train_matcher(X, y, test_size=0.3, random_state=42)

        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        prev_pos = len(self.features_df) + 1

        for thresh in thresholds:
            preds = predict_matches(model, self.features_df, threshold=thresh)
            pos_count = preds["prediction"].sum()
            self.assertLessEqual(pos_count, prev_pos)  # Higher threshold -> fewer positives
            prev_pos = pos_count


if __name__ == "__main__":
    unittest.main()