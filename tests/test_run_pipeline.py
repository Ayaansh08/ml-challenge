"""Unit and integration tests for src/pipeline/run_pipeline.py."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.run_pipeline import (
    clean_source_file,
    run_aggregation_stage,
    run_pipeline,
    validate_submission_integrity,
)


class TestRunPipelineHelpers(unittest.TestCase):
    """Test helper functions in run_pipeline."""

    def test_clean_source_file_with_nrows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            raw_tsv = tmp / "train_source1.tsv"
            out_parquet = tmp / "train_source1_cleaned.parquet"

            df = pd.DataFrame({
                "entity_id": ["S1-1", "S1-2", "S1-3"],
                "business_name": ["Acme Corp Ltd", "Beta Inc", "Gamma LLC"],
                "business_address": ["123 Main St", "456 Oak Rd", "789 Pine Ave"],
                "country": ["US", "US", "UK"],
            })
            df.to_csv(raw_tsv, sep="\t", index=False)

            # Test reading with nrows=2
            count = clean_source_file(raw_tsv, out_parquet, nrows=2, force=True)
            self.assertEqual(count, 2)
            self.assertTrue(out_parquet.exists())

            cleaned = pd.read_parquet(out_parquet)
            self.assertEqual(len(cleaned), 2)
            self.assertIn("cleaned_name", cleaned.columns)

    def test_validate_submission_integrity_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cleaned_test_dir = tmp / "cleaned" / "test"
            cleaned_test_dir.mkdir(parents=True)

            # Create synthetic test cleaned parquet
            df_test = pd.DataFrame({
                "cleaned_entity_id": ["S1-10", "S2-20", "S3-30"],
                "cleaned_name": ["Alpha", "Beta", "Gamma"],
            })
            df_test.to_parquet(cleaned_test_dir / "test_source1_cleaned.parquet")

            # Create matching submission
            sub_file = tmp / "submission.tsv"
            sub_df = pd.DataFrame({
                "entity_id": ["S1-10", "S2-20", "S3-30"],
                "cluster_id": [1, 1, 2],
            })
            sub_df.to_csv(sub_file, sep="\t", index=False)

            stats = validate_submission_integrity(sub_file, cleaned_test_dir)
            self.assertEqual(stats["total_records"], 3)
            self.assertEqual(stats["total_unique_clusters"], 2)
            self.assertEqual(stats["singleton_clusters"], 1)
            self.assertEqual(stats["multi_entity_clusters"], 1)

    def test_validate_submission_integrity_missing_entity_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cleaned_test_dir = tmp / "cleaned" / "test"
            cleaned_test_dir.mkdir(parents=True)

            df_test = pd.DataFrame({
                "cleaned_entity_id": ["S1-10", "S2-20"],
            })
            df_test.to_parquet(cleaned_test_dir / "test_source1_cleaned.parquet")

            sub_file = tmp / "submission.tsv"
            sub_df = pd.DataFrame({
                "entity_id": ["S1-10"],  # S2-20 missing
                "cluster_id": [1],
            })
            sub_df.to_csv(sub_file, sep="\t", index=False)

            with self.assertRaises(ValueError):
                validate_submission_integrity(sub_file, cleaned_test_dir)


if __name__ == "__main__":
    unittest.main()
