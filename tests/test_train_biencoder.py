"""Unit tests for train_biencoder.py pure evaluation and dataset loading functions."""

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from src.pipeline.train_biencoder import (
    compute_recall_at_k,
    load_pair_dataset,
    run_comparison,
)


class DummyMockModel:
    """Mock SentenceTransformer model returning deterministic normalized embeddings."""

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def encode(self, texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            vecs.append(self.mapping.get(t, np.array([0.1, 0.1, 0.1], dtype=np.float32)))
        arr = np.array(vecs, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
            arr = arr / norms
        return arr


class TestTrainBiencoder(unittest.TestCase):
    """Test suite for bi-encoder evaluation, recall computation, and dataset loading."""

    def test_load_pair_dataset(self):
        """Test loading dataframe pairs into InputExample objects."""
        train_df = pd.DataFrame([
            {"s1_text": "Acme Inc [SEP] 123 Main St", "match_text": "Acme LLC [SEP] 123 Main Street"},
            {"s1_text": "Beta Corp [SEP] 456 Elm St", "match_text": "Beta Co [SEP] 456 Elm St"},
        ])
        val_df = pd.DataFrame([
            {"s1_text": "Gamma [SEP] 789 Oak Ave", "match_text": "Gamma Inc [SEP] 789 Oak Avenue"},
        ])

        train_ex, val_ex = load_pair_dataset(train_df, val_df)
        self.assertEqual(len(train_ex), 2)
        self.assertEqual(len(val_ex), 1)
        self.assertEqual(train_ex[0].texts, ["Acme Inc [SEP] 123 Main St", "Acme LLC [SEP] 123 Main Street"])
        self.assertEqual(val_ex[0].texts, ["Gamma [SEP] 789 Oak Ave", "Gamma Inc [SEP] 789 Oak Avenue"])

    def test_compute_recall_at_k(self):
        """Test Recall@k calculation against a known candidate pool with mock embeddings."""
        val_df = pd.DataFrame([
            {"s1_entity_id": "S1-A", "match_entity_id": "S2-1", "s1_text": "Anchor A", "match_text": "Match 1"},
            {"s1_entity_id": "S1-B", "match_entity_id": "S2-2", "s1_text": "Anchor B", "match_text": "Match 2"},
        ])

        cand_pool_df = pd.DataFrame([
            {"match_entity_id": "S2-1", "match_text": "Match 1"},  # Matches Anchor A
            {"match_entity_id": "S2-2", "match_text": "Match 2"},  # Matches Anchor B
            {"match_entity_id": "S2-3", "match_text": "Match 3 (Distractor)"},
        ])

        # Define mock embeddings: Anchor A close to Match 1, Anchor B close to Match 2
        mapping = {
            "Anchor A": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "Match 1": np.array([0.99, 0.01, 0.0], dtype=np.float32),
            "Anchor B": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "Match 2": np.array([0.0, 0.99, 0.01], dtype=np.float32),
            "Match 3 (Distractor)": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        mock_model = DummyMockModel(mapping)

        recalls = compute_recall_at_k(mock_model, val_df, candidate_pool_df=cand_pool_df, k_values=[1, 2, 3])
        self.assertEqual(recalls[1], 1.0)
        self.assertEqual(recalls[2], 1.0)
        self.assertEqual(recalls[3], 1.0)


if __name__ == "__main__":
    unittest.main()
