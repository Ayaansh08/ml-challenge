"""Unit tests for pair_construction.py module."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.pair_construction import (
    apply_french_proxy_normalization,
    build_country_stratified_proxy_split,
    build_positive_pairs,
    format_entity_text,
    grouped_train_val_split,
    load_and_clean_sources,
    mine_hard_negatives,
    write_pair_dataset,
)


class TestPairConstruction(unittest.TestCase):
    """Test suite for pair construction pure functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_format_entity_text(self):
        """Test formatting name and address with [SEP] token."""
        self.assertEqual(format_entity_text("Acme Corp", "123 Main St"), "Acme Corp [SEP] 123 Main St")
        self.assertEqual(format_entity_text("Acme Corp", None), "Acme Corp")
        self.assertEqual(format_entity_text("Acme Corp", ""), "Acme Corp")

    def test_load_and_clean_sources_valid(self):
        """Test loading and cleaning valid sources with proper prefixes."""
        s1_file = self.temp_path / "s1.tsv"
        s2_file = self.temp_path / "s2.tsv"
        s3_file = self.temp_path / "s3.tsv"

        pd.DataFrame([
            {"entity_id": "S1-101", "business_name": "Acme Inc", "business_address": "123 Main St", "country": "US"}
        ]).to_csv(s1_file, sep="\t", index=False)

        pd.DataFrame([
            {"entity_id": "S2-201", "business_name": "Acme LLC", "business_address": "123 Main Street", "country": "US"}
        ]).to_csv(s2_file, sep="\t", index=False)

        pd.DataFrame([
            {"entity_id": "S3-301", "business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"}
        ]).to_csv(s3_file, sep="\t", index=False)

        sources = load_and_clean_sources(s1_file, s2_file, s3_file)
        self.assertIn("s1", sources)
        self.assertIn("s2", sources)
        self.assertIn("s3", sources)
        self.assertEqual(sources["s1"].iloc[0]["cleaned_entity_id"], "S1-101")
        self.assertEqual(sources["s2"].iloc[0]["cleaned_entity_id"], "S2-201")
        self.assertEqual(sources["s3"].iloc[0]["cleaned_entity_id"], "S3-301")

    def test_load_and_clean_sources_invalid_prefix(self):
        """Test that invalid entity_id prefixes raise a descriptive ValueError."""
        s1_file = self.temp_path / "s1.tsv"
        s2_file = self.temp_path / "s2.tsv"
        s3_file = self.temp_path / "s3.tsv"

        # Invalid prefix for S1 (starts with S2-)
        pd.DataFrame([
            {"entity_id": "S2-999", "business_name": "Invalid Co", "business_address": "Rd", "country": "US"}
        ]).to_csv(s1_file, sep="\t", index=False)
        pd.DataFrame([{"entity_id": "S2-201", "business_name": "B", "business_address": "A", "country": "US"}]).to_csv(s2_file, sep="\t", index=False)
        pd.DataFrame([{"entity_id": "S3-301", "business_name": "C", "business_address": "A", "country": "US"}]).to_csv(s3_file, sep="\t", index=False)

        with self.assertRaises(ValueError) as ctx:
            load_and_clean_sources(s1_file, s2_file, s3_file)
        self.assertIn("Prefix validation failed for source 's1'", str(ctx.exception))

    def test_build_positive_pairs_and_singleton_skipping(self):
        """Test positive pairs creation and singleton counting."""
        gt_file = self.temp_path / "gt.tsv"
        gt_data = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10,S3-20"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},  # Singleton
            {"source1_entity_id": "S1-3", "matched_entity_ids": "NaN"},  # Singleton
            {"source1_entity_id": "S1-4", "matched_entity_ids": "S2-40"},
        ])
        gt_data.to_csv(gt_file, sep="\t", index=False)

        s1_df = pd.DataFrame([
            {"cleaned_entity_id": "S1-1", "cleaned_name": "Alpha Corp", "cleaned_address": "10 High St", "country": "US"},
            {"cleaned_entity_id": "S1-2", "cleaned_name": "Beta LLC", "cleaned_address": "20 Main Rd", "country": "US"},
            {"cleaned_entity_id": "S1-3", "cleaned_name": "Gamma Inc", "cleaned_address": None, "country": "US"},
            {"cleaned_entity_id": "S1-4", "cleaned_name": "Delta Co", "cleaned_address": "30 Park Ave", "country": "US"},
        ])
        s2_df = pd.DataFrame([
            {"cleaned_entity_id": "S2-10", "cleaned_name": "Alpha Corporation", "cleaned_address": "10 High Street", "country": "US"},
            {"cleaned_entity_id": "S2-40", "cleaned_name": "Delta Company", "cleaned_address": "30 Park Avenue", "country": "US"},
        ])
        s3_df = pd.DataFrame([
            {"cleaned_entity_id": "S3-20", "cleaned_name": "Alpha", "cleaned_address": "10 High St", "country": "US"},
        ])

        pairs = build_positive_pairs(gt_file, s1_df, s2_df, s3_df)
        self.assertEqual(len(pairs), 3)  # S1-1 -> S2-10, S1-1 -> S3-20, S1-4 -> S2-40
        self.assertListEqual(list(pairs.columns), ["s1_entity_id", "match_entity_id", "source", "s1_text", "match_text"])
        self.assertEqual(set(pairs["source"]), {"s2", "s3"})
        self.assertIn("Alpha Corp [SEP] 10 High St", pairs["s1_text"].values)

    def test_grouped_train_val_split_zero_overlap(self):
        """Test grouped split strictly prevents any S1 entity from being in both train and val."""
        pairs_df = pd.DataFrame([
            {"s1_entity_id": "S1-1", "match_entity_id": "S2-10", "source": "s2", "s1_text": "A", "match_text": "A"},
            {"s1_entity_id": "S1-1", "match_entity_id": "S3-11", "source": "s3", "s1_text": "A", "match_text": "A"},
            {"s1_entity_id": "S1-2", "match_entity_id": "S2-20", "source": "s2", "s1_text": "B", "match_text": "B"},
            {"s1_entity_id": "S1-3", "match_entity_id": "S2-30", "source": "s2", "s1_text": "C", "match_text": "C"},
            {"s1_entity_id": "S1-4", "match_entity_id": "S3-40", "source": "s3", "s1_text": "D", "match_text": "D"},
            {"s1_entity_id": "S1-5", "match_entity_id": "S2-50", "source": "s2", "s1_text": "E", "match_text": "E"},
        ])

        train_df, val_df = grouped_train_val_split(pairs_df, val_fraction=0.4, random_seed=42)
        train_s1 = set(train_df["s1_entity_id"])
        val_s1 = set(val_df["s1_entity_id"])

        self.assertEqual(len(train_s1.intersection(val_s1)), 0)
        self.assertEqual(len(train_s1) + len(val_s1), pairs_df["s1_entity_id"].nunique())
        self.assertEqual(len(train_df) + len(val_df), len(pairs_df))

    def test_build_country_stratified_proxy_split(self):
        """Test French proxy transformations on US/India entities."""
        s1_df = pd.DataFrame([
            {
                "cleaned_entity_id": "S1-US1",
                "cleaned_country": "US",
                "business_name": "SARL Lafayette Tech",
                "business_address": "15 BVD SAINT GERMAIN",
                "cleaned_name": "sarl lafayette tech",
                "cleaned_address": "15 bvd saint germain",
            },
            {
                "cleaned_entity_id": "S1-FR1",
                "cleaned_country": "FR",  # Not target country
                "business_name": "Paris Co",
                "business_address": "Rue 1",
                "cleaned_name": "paris co",
                "cleaned_address": "rue 1",
            },
        ])

        proxy_df = build_country_stratified_proxy_split(s1_df, val_entity_ids=["S1-US1", "S1-FR1"])
        self.assertEqual(len(proxy_df), 1)
        self.assertEqual(proxy_df.iloc[0]["entity_id"], "S1-US1")
        self.assertIn("societe a responsabilite limitee", proxy_df.iloc[0]["france_proxy_text"])
        self.assertIn("boulevard", proxy_df.iloc[0]["france_proxy_text"])

    def test_mine_hard_negatives(self):
        """Test cosine similarity ranking and positive match exclusion in hard negative mining."""
        cand_df = pd.DataFrame([
            {"match_entity_id": "S2-001", "match_text": "Alpha Technologies"},
            {"match_entity_id": "S2-002", "match_text": "Beta Solutions"},
            {"match_entity_id": "S2-003", "match_text": "Gamma Global"},
            {"match_entity_id": "S2-004", "match_text": "Delta Inc"},
        ])

        # Artificial embeddings: anchor is close to S2-001 and S2-002
        anchor_vec = np.array([1.0, 0.0, 0.0])
        cand_embeddings = np.array([
            [0.99, 0.01, 0.0],   # S2-001 (Sim ~0.99) - True positive
            [0.95, 0.05, 0.0],   # S2-002 (Sim ~0.95) - Hard negative!
            [0.20, 0.90, 0.0],   # S2-003
            [0.10, 0.0, 0.90],   # S2-004
        ])

        # S2-001 is a known positive, so top hard negative should be S2-002
        negatives = mine_hard_negatives(
            anchor_entity_id="S1-100",
            anchor_text="Alpha Tech",
            candidate_pool_df=cand_df,
            current_embeddings=cand_embeddings,
            anchor_embedding=anchor_vec,
            known_positive_ids={"S2-001"},
            k=2,
        )

        self.assertEqual(len(negatives), 2)
        self.assertEqual(negatives[0]["negative_entity_id"], "S2-002")
        self.assertAlmostEqual(negatives[0]["similarity_score"], 0.95, places=1)
        self.assertEqual(negatives[1]["negative_entity_id"], "S2-003")

    def test_write_pair_dataset(self):
        """Test end-to-end writing of parquet files and json summary."""
        pairs_df = pd.DataFrame([
            {"s1_entity_id": f"S1-{i}", "match_entity_id": f"S2-{i}", "source": "s2", "s1_text": f"A{i}", "match_text": f"B{i}"}
            for i in range(10)
        ])
        s1_df = pd.DataFrame([
            {"cleaned_entity_id": f"S1-{i}", "cleaned_country": "US", "business_name": f"A{i}", "business_address": f"St {i}", "cleaned_name": f"a{i}", "cleaned_address": f"st {i}"}
            for i in range(10)
        ])

        out_dir = self.temp_path / "pairs_out"
        summary = write_pair_dataset(
            positive_pairs_df=pairs_df,
            output_dir=out_dir,
            val_fraction=0.2,
            cleaned_s1=s1_df,
            singleton_count=5,
        )

        self.assertTrue((out_dir / "train_positives.parquet").exists())
        self.assertTrue((out_dir / "val_positives.parquet").exists())
        self.assertTrue((out_dir / "france_proxy_val.parquet").exists())
        self.assertTrue((out_dir / "pair_summary.json").exists())
        self.assertEqual(summary["dataset_summary"]["total_positive_pairs"], 10)
        self.assertEqual(summary["dataset_summary"]["singleton_s1_entities_skipped"], 5)


if __name__ == "__main__":
    unittest.main()
