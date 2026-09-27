"""Unit tests for Phase 4 feature engineering functions."""

import unittest
import pandas as pd
import numpy as np
from src.pipeline.features import (
    levenshtein_ratio,
    jaro_winkler_similarity,
    token_set_ratio,
    token_sort_ratio,
    partial_ratio,
    exact_match,
    prefix_match,
    common_token_count,
    jaccard_token_similarity,
    cosine_token_similarity,
    address_postal_match,
    address_street_number_match,
    address_landmark_overlap,
    country_match,
    is_bare_domain_flag,
    legal_suffix_match,
    name_length_ratio,
    compute_pair_features,
)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for pairwise similarity feature functions."""

    def test_levenshtein_ratio(self):
        self.assertEqual(levenshtein_ratio("acme", "acme"), 1.0)
        self.assertAlmostEqual(levenshtein_ratio("acme", "acme corp"), 4/9, places=2)
        self.assertEqual(levenshtein_ratio("", "test"), 0.0)
        self.assertEqual(levenshtein_ratio(None, "test"), 0.0)
        self.assertEqual(levenshtein_ratio("test", None), 0.0)

    def test_jaro_winkler_similarity(self):
        self.assertEqual(jaro_winkler_similarity("acme", "acme"), 1.0)
        self.assertGreater(jaro_winkler_similarity("acme corp", "acme corporation"), 0.8)
        self.assertEqual(jaro_winkler_similarity("", "test"), 0.0)

    def test_token_set_ratio(self):
        self.assertEqual(token_set_ratio("acme corporation", "corporation acme"), 1.0)
        self.assertGreater(token_set_ratio("acme corp", "acme corporation"), 0.7)
        self.assertEqual(token_set_ratio("", "test"), 0.0)

    def test_token_sort_ratio(self):
        self.assertEqual(token_sort_ratio("acme corporation", "corporation acme"), 1.0)
        self.assertEqual(token_sort_ratio("", "test"), 0.0)

    def test_partial_ratio(self):
        self.assertEqual(partial_ratio("acme", "acme corporation"), 1.0)
        self.assertEqual(partial_ratio("", "test"), 0.0)

    def test_exact_match(self):
        self.assertEqual(exact_match("acme", "acme"), 1)
        self.assertEqual(exact_match("Acme", "acme"), 1)
        self.assertEqual(exact_match("acme", "acme corp"), 0)
        self.assertEqual(exact_match("", "test"), 0)
        self.assertEqual(exact_match(None, "test"), 0)

    def test_prefix_match(self):
        self.assertEqual(prefix_match("acme corporation ltd", "acme corporation inc", 2), 1)
        self.assertEqual(prefix_match("acme corporation ltd", "acme inc", 2), 0)
        self.assertEqual(prefix_match("acme", "acme corp", 1), 1)
        self.assertEqual(prefix_match("", "test"), 0)

    def test_common_token_count(self):
        self.assertEqual(common_token_count("acme corporation", "corporation acme"), 2)
        self.assertEqual(common_token_count("acme corp", "acme inc"), 1)
        self.assertEqual(common_token_count("", "test"), 0)

    def test_jaccard_token_similarity(self):
        self.assertEqual(jaccard_token_similarity("acme corporation", "corporation acme"), 1.0)
        self.assertEqual(jaccard_token_similarity("acme corp", "acme inc"), 1/3)
        self.assertEqual(jaccard_token_similarity("", "test"), 0.0)

    def test_cosine_token_similarity(self):
        self.assertAlmostEqual(cosine_token_similarity("acme corporation", "corporation acme"), 1.0, places=10)
        self.assertGreater(cosine_token_similarity("acme corp", "acme inc"), 0.0)
        self.assertEqual(cosine_token_similarity("", "test"), 0.0)

    def test_address_postal_match(self):
        self.assertEqual(address_postal_match("123 main st 10001", "456 oak ave 10001"), 1)
        self.assertEqual(address_postal_match("123 main st 10001", "456 oak ave 90210"), 0)
        self.assertEqual(address_postal_match("", "10001"), 0)

    def test_address_street_number_match(self):
        self.assertEqual(address_street_number_match("123 main st", "123 oak ave"), 1)
        self.assertEqual(address_street_number_match("plot 42", "plot 42"), 1)
        self.assertEqual(address_street_number_match("123 main st", "456 oak ave"), 0)

    def test_address_landmark_overlap(self):
        self.assertGreater(
            address_landmark_overlap("near metro station", "opp metro station"),
            0.0
        )
        self.assertEqual(address_landmark_overlap("near park", "near mall"), 0.0)
        self.assertEqual(address_landmark_overlap("", "near station"), 0.0)

    def test_country_match(self):
        self.assertEqual(country_match("US", "US"), 1)
        self.assertEqual(country_match("us", "US"), 1)
        self.assertEqual(country_match("IN", "INDIA"), 0)
        self.assertEqual(country_match("", "US"), 0)

    def test_is_bare_domain_flag(self):
        self.assertEqual(is_bare_domain_flag("example.com"), 1)
        self.assertEqual(is_bare_domain_flag("my-business.co.in"), 1)
        self.assertEqual(is_bare_domain_flag("https://www.store.org"), 1)
        self.assertEqual(is_bare_domain_flag("Acme Corporation"), 0)
        self.assertEqual(is_bare_domain_flag("Amazon Web Services LLC"), 0)
        self.assertEqual(is_bare_domain_flag(""), 0)

    def test_legal_suffix_match(self):
        self.assertEqual(legal_suffix_match("acme limited", "xyz limited"), 1)
        self.assertEqual(legal_suffix_match("acme llc", "xyz llc"), 1)
        self.assertEqual(legal_suffix_match("acme limited", "xyz incorporated"), 0)
        self.assertEqual(legal_suffix_match("acme", "xyz limited"), 0)
        self.assertEqual(legal_suffix_match("", "xyz"), 0)

    def test_name_length_ratio(self):
        self.assertEqual(name_length_ratio("acme", "acme"), 1.0)
        self.assertAlmostEqual(name_length_ratio("acme", "acme corporation"), 4/16, places=10)
        self.assertEqual(name_length_ratio("", "test"), 0.0)

    def test_compute_pair_features(self):
        row1 = pd.Series({
            "cleaned_name": "acme corporation",
            "cleaned_address": "123 main street 10001",
            "cleaned_country": "US",
        })
        row2 = pd.Series({
            "cleaned_name": "acme corp",
            "cleaned_address": "123 main st 10001",
            "cleaned_country": "US",
        })

        feats = compute_pair_features(row1, row2)

        self.assertIn("name_levenshtein", feats)
        self.assertIn("name_jaro_winkler", feats)
        self.assertIn("name_token_set_ratio", feats)
        self.assertIn("name_exact_match", feats)
        self.assertIn("country_match", feats)
        self.assertIn("addr_postal_match", feats)
        self.assertIn("embedding_cosine", feats)

        # Country should match
        self.assertEqual(feats["country_match"], 1)
        # Postal code should match
        self.assertEqual(feats["addr_postal_match"], 1)
        # Name should have high similarity but not exact
        self.assertGreater(feats["name_token_set_ratio"], 0.5)
        self.assertEqual(feats["name_exact_match"], 0)

    def test_compute_pair_features_with_embedding(self):
        row1 = pd.Series({
            "cleaned_name": "acme",
            "cleaned_address": "123 main st",
            "cleaned_country": "US",
        })
        row2 = pd.Series({
            "cleaned_name": "acme",
            "cleaned_address": "123 main st",
            "cleaned_country": "US",
        })

        feats = compute_pair_features(row1, row2, embedding_sim=0.95)

        self.assertEqual(feats["embedding_cosine"], 0.95)
        self.assertEqual(feats["name_exact_match"], 1)
        self.assertEqual(feats["addr_exact_match"], 1)
        self.assertEqual(feats["country_match"], 1)


if __name__ == "__main__":
    unittest.main()