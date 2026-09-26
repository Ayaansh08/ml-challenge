"""Unit tests for Phase 2 data cleaning and normalization functions.

Covers:
- Missing address handling (None, NaN, empty string, ~3.3% missing rate in S2/S3)
- Literal "NaN" / "nan" / "null" string normalization
- Leading-space and whitespace-padded entity_id values
- Non-ASCII / Devanagari multilingual script preservation
- ALL-CAPS address casefolding and abbreviation expansion
- Bare domain name detection as boolean flag
- Prefix-position and suffix-position legal entity expansions
- Ampersand '&' <-> 'and' normalization
- Non-destructive record cleaning preserving original raw values
- Vectorized clean_dataframe parity with clean_record
"""

import math
import unittest
import pandas as pd

from src.pipeline.cleaning import (
    clean_address,
    clean_dataframe,
    clean_name,
    clean_record,
    is_bare_domain,
    normalize_missing,
    normalize_whitespace,
)


class TestDataCleaning(unittest.TestCase):
    """Test suite for pure data cleaning and normalization functions."""

    # 1. Missing Address Handling
    def test_missing_address(self):
        """Ensure missing address in various representations does not raise and returns None."""
        for missing_val in [None, "", "   ", float("nan"), "NaN", "nan", "None", "null", "N/A"]:
            res = clean_address(missing_val)
            self.assertIsInstance(res, dict)
            self.assertIsNone(res["cleaned_address"])
            self.assertIsNone(res["landmark"])
            self.assertIsNone(res["postal_code"])
            self.assertIsNone(res["street_number"])

    # 2. Literal "NaN" / "nan" String
    def test_literal_nan_string(self):
        """Ensure literal 'NaN', 'nan', 'none', and case variants normalize to None."""
        self.assertIsNone(normalize_missing("NaN"))
        self.assertIsNone(normalize_missing("nan"))
        self.assertIsNone(normalize_missing("NAN"))
        self.assertIsNone(normalize_missing("None"))
        self.assertIsNone(normalize_missing("null"))
        self.assertIsNone(normalize_missing("n/a"))
        self.assertIsNone(normalize_missing(float("nan")))
        self.assertIsNone(normalize_missing(""))
        self.assertEqual(normalize_missing("Valid Business"), "Valid Business")

    # 3. Leading-Space on entity_id
    def test_leading_space_entity_id(self):
        """Ensure leading and trailing whitespace on entity_id is cleanly stripped."""
        self.assertEqual(normalize_whitespace("  S1-965667"), "S1-965667")
        self.assertEqual(normalize_whitespace(" S2-681193310  "), "S2-681193310")
        self.assertEqual(normalize_whitespace("S3-775321672\t"), "S3-775321672")

        record = {
            "entity_id": "   S1-00421   ",
            "business_name": "Test Co",
            "business_address": "123 Main St",
            "country": "US",
        }
        cleaned = clean_record(record)
        self.assertEqual(cleaned["entity_id"], "   S1-00421   ")  # Raw preserved
        self.assertEqual(cleaned["cleaned_entity_id"], "S1-00421")

    # 4. Devanagari / Non-ASCII Script Name Preservation
    def test_devanagari_name_preservation(self):
        """Ensure Hindi/Devanagari scripts survive intact and are not stripped."""
        hindi_name = "रिलायंस इंडस्ट्रीज Pvt. Ltd."
        cleaned = clean_name(hindi_name)
        # Devanagari characters must be preserved; legal suffix expanded to 'private limited'
        self.assertIn("रिलायंस इंडस्ट्रीज", cleaned)
        self.assertIn("private limited", cleaned)

        pure_devanagari = "टाटा कंसल्टेंसी सर्विसेज"
        self.assertEqual(clean_name(pure_devanagari), "टाटा कंसल्टेंसी सर्विसेज")

    # 5. ALL-CAPS Address Normalization
    def test_all_caps_address_normalization(self):
        """Ensure ALL-CAPS addresses are casefolded and abbreviations are expanded."""
        caps_addr = "123 MAIN ST, SUITE 400, NEW YORK, NY 10001"
        res = clean_address(caps_addr)
        self.assertIsNotNone(res["cleaned_address"])
        # Should expand ST -> street and SUITE -> suite
        self.assertIn("street", res["cleaned_address"])
        self.assertIn("suite", res["cleaned_address"])
        self.assertEqual(res["postal_code"], "10001")
        self.assertEqual(res["street_number"], "123")

    # 6. Bare Domain as Business Name Flagging
    def test_domain_as_name_flag(self):
        """Ensure standalone domain names are flagged with is_bare_domain=True."""
        self.assertTrue(is_bare_domain("example.com"))
        self.assertTrue(is_bare_domain("my-business.co.in"))
        self.assertTrue(is_bare_domain("https://www.store.org"))
        self.assertTrue(is_bare_domain("techstart.ai"))

        # Non-bare domain business names should return False
        self.assertFalse(is_bare_domain("Acme Corporation"))
        self.assertFalse(is_bare_domain("Amazon Web Services LLC"))
        self.assertFalse(is_bare_domain("example.com solutions"))

    # 7. Prefix-Position Legal Suffix Expansion
    def test_prefix_position_legal_suffix(self):
        """Ensure legal entity identifiers in prefix position are properly expanded."""
        prefix_llc = "LLC Moncada Enterprises"
        cleaned_llc = clean_name(prefix_llc)
        self.assertEqual(cleaned_llc, "limited liability company moncada enterprises")

        prefix_pvt = "Pvt Ltd Alpha Solutions"
        cleaned_pvt = clean_name(prefix_pvt)
        self.assertEqual(cleaned_pvt, "private limited alpha solutions")

        # Multi-position legal terms
        multi_legal = "Pvt. EFS Logistics Ltd."
        cleaned_multi = clean_name(multi_legal)
        self.assertEqual(cleaned_multi, "private efs logistics limited")

    # 8. Ampersand '&' Normalization
    def test_ampersand_normalization(self):
        """Ensure '&' and fullwidth '＆' are consistently normalized to 'and'."""
        self.assertEqual(clean_name("AT&T"), "at and t")
        self.assertEqual(clean_name("Johnson & Johnson"), "johnson and johnson")
        self.assertEqual(clean_name("Barnes & Noble Inc."), "barnes and noble incorporated")
        self.assertEqual(clean_name("A＆B Trading"), "a and b trading")

    # 9. Comprehensive clean_record Integration
    def test_clean_record_preserves_originals(self):
        """Ensure clean_record preserves original fields while creating clean/flag fields."""
        raw_row = {
            "entity_id": "  S2-998811  ",
            "business_name": "  --Acme & Co. Ltd.  ",
            "business_address": "PLOT 42, NEAR METRO STATION, MG RD, 560001",
            "country": " in ",
            "custom_metadata": "meta_123",
        }
        cleaned = clean_record(raw_row)

        # Original raw fields preserved unchanged
        self.assertEqual(cleaned["entity_id"], "  S2-998811  ")
        self.assertEqual(cleaned["business_name"], "  --Acme & Co. Ltd.  ")
        self.assertEqual(cleaned["business_address"], "PLOT 42, NEAR METRO STATION, MG RD, 560001")
        self.assertEqual(cleaned["country"], " in ")
        self.assertEqual(cleaned["custom_metadata"], "meta_123")

        # Cleaned and derived fields
        self.assertEqual(cleaned["cleaned_entity_id"], "S2-998811")
        self.assertEqual(cleaned["cleaned_name"], "acme and company limited")
        self.assertFalse(cleaned["is_bare_domain"])
        self.assertEqual(cleaned["cleaned_country"], "IN")
        self.assertEqual(cleaned["postal_code"], "560001")
        self.assertEqual(cleaned["street_number"], "42")
        self.assertIsNotNone(cleaned["landmark"])
        self.assertIn("near metro station", cleaned["landmark"])
        self.assertIn("road", cleaned["cleaned_address"])

    # 10. Vectorized clean_dataframe Parity with clean_record
    def test_clean_dataframe_parity(self):
        """Ensure clean_dataframe matches clean_record output across rows."""
        df = pd.DataFrame([
            {
                "entity_id": "  S1-001  ",
                "business_name": "Acme & Sons LLC",
                "business_address": "100 MAIN ST, 90210",
                "country": "us",
            },
            {
                "entity_id": "S2-002",
                "business_name": "example.com",
                "business_address": None,
                "country": "IN",
            },
        ])
        cleaned_df = clean_dataframe(df)
        self.assertEqual(len(cleaned_df), 2)
        self.assertEqual(cleaned_df.loc[0, "cleaned_entity_id"], "S1-001")
        self.assertEqual(cleaned_df.loc[0, "cleaned_name"], "acme and sons limited liability company")
        self.assertFalse(cleaned_df.loc[0, "is_bare_domain"])
        self.assertEqual(cleaned_df.loc[1, "cleaned_name"], "example.com")
        self.assertTrue(cleaned_df.loc[1, "is_bare_domain"])
        self.assertTrue(pd.isna(cleaned_df.loc[1, "cleaned_address"]))


if __name__ == "__main__":
    unittest.main()
