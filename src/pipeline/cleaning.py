"""Data cleaning, normalization, and validation functions for business entities.

Pure functions designed for unit testing with zero side effects.
Handles messy real-world entity resolution patterns including:
- Leading/trailing whitespace on entity IDs and text fields
- Missing/NaN representation variants (None, NaN, "", "NaN", "nan", "null")
- Multi-script preservation (NFKC Unicode normalization, Devanagari/non-Latin scripts)
- Legal suffix/prefix expansion across leading and trailing positions
- Bare domain detection
- Address parsing (casefolding, abbreviation expansion, landmark extraction, postal code/street number parsing)
- Vectorized chunk processing for high-throughput, memory-bounded DataFrame cleaning
"""

import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & Synonym Tables
# ---------------------------------------------------------------------------

# Missing value string representations (case-insensitive)
MISSING_STRINGS = {
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "<na>",
    "undefined",
    ".",
    "-",
    "?",
}

# Legal entity abbreviations and canonical expansions
LEGAL_SYNONYMS: List[Tuple[str, str]] = [
    # Multi-word patterns first to prevent premature single-word matching
    (r"\bpvt\.?\s*ltd\.?\b", "private limited"),
    (r"\bpte\.?\s*ltd\.?\b", "private limited"),
    (r"\bpty\.?\s*ltd\.?\b", "proprietary limited"),
    (r"\bprivate\s+limited\b", "private limited"),
    (r"\bpublic\s+limited\s+company\b", "public limited company"),
    (r"\blimited\s+liability\s+company\b", "limited liability company"),
    (r"\blimited\s+liability\s+partnership\b", "limited liability partnership"),
    (r"\bl\.?l\.?c\.?\b", "limited liability company"),
    (r"\bl\.?l\.?p\.?\b", "limited liability partnership"),
    (r"\bp\.?l\.?c\.?\b", "public limited company"),
    (r"\bltd\.?\b", "limited"),
    (r"\blimited\b", "limited"),
    (r"\binc\.?\b", "incorporated"),
    (r"\bincorporated\b", "incorporated"),
    (r"\bcorp\.?\b", "corporation"),
    (r"\bcorporation\b", "corporation"),
    (r"\bco\.?\b", "company"),
    (r"\bcompany\b", "company"),
    (r"\bg\.?m\.?b\.?h\.?\b", "gmbh"),
    (r"\bs\.?a\.?r\.?l\.?\b", "sarl"),
    (r"\bs\.?a\.?\b", "sa"),
    (r"\bs\.?r\.?l\.?\b", "srl"),
    (r"\bs\.?p\.?a\.?\b", "spa"),
    (r"\bb\.?v\.?\b", "bv"),
    (r"\bn\.?v\.?\b", "nv"),
    (r"\bpvt\.?\b", "private"),
    (r"\bpte\.?\b", "private"),
    (r"\bpty\.?\b", "proprietary"),
]

# Bare domain regex (detects domain names without surrounding business name tokens)
DOMAIN_PATTERN = re.compile(
    r"^(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.)+(com|net|org|co\.in|in|io|ai|biz|info|edu|gov|me|app|dev|tech|store|online|co|uk|us|ca|de|fr|au|jp|cn|sg|hk|ae|sa|eu|xyz|site)(?:\/[^\s]*)?$",
    re.IGNORECASE,
)

# Address abbreviation synonym mappings
ADDRESS_ABBREVIATIONS: List[Tuple[str, str]] = [
    (r"\brd\.?\b", "road"),
    (r"\bst\.?\b", "street"),
    (r"\bave\.?\b|\bav\.?\b", "avenue"),
    (r"\bblvd\.?\b", "boulevard"),
    (r"\bdr\.?\b", "drive"),
    (r"\bln\.?\b", "lane"),
    (r"\bct\.?\b", "court"),
    (r"\bpl\.?\b", "place"),
    (r"\bhwy\.?\b", "highway"),
    (r"\bfwy\.?\b", "freeway"),
    (r"\bpkwy\.?\b|\bpk\.?\b", "parkway"),
    (r"\bsq\.?\b", "square"),
    (r"\bste\.?\b|\bste#\b", "suite"),
    (r"\bapt\.?\b|\bapt#\b", "apartment"),
    (r"\bflr\.?\b|\bfl\.?\b", "floor"),
    (r"\bbldg\.?\b", "building"),
    (r"\bdept\.?\b", "department"),
    (r"\bext\.?\b", "extension"),
    (r"\bno\.?\b", "number"),
    (r"\bopp\.?\b|\bopposite\b", "opposite"),
    (r"\bnr\.?\b", "near"),
    (r"\badj\.?\b|\badjacent\b", "adjacent"),
]

# Landmark phrase regex: matches "near X", "opp. X", "behind X", etc.
LANDMARK_REGEX = re.compile(
    r"\b(near|opp\.?|opposite|behind|next to|adjacent to|in front of|beside)\s+([^,;]+)",
    re.IGNORECASE,
)

# Postal code patterns (6-digit PIN codes, 5-digit / 5+4 US ZIP, UK, Canadian codes)
POSTAL_CODE_REGEX = re.compile(
    r"\b(?:([1-9][0-9]{5})|([0-9]{5}(?:-[0-9]{4})?)|([A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2})|([A-Z][0-9][A-Z]\s?[0-9][A-Z][0-9]))\b",
    re.IGNORECASE,
)

# Street / unit number pattern (e.g. "#123", "Plot 45", "Flat 3B", "12/A", "45-B")
STREET_NUMBER_REGEX = re.compile(
    r"\b(?:(?:no\.?|plot|house|flat|unit|shop|#|suite|apt)\s*)?([0-9]+[a-zA-Z]?(?:[/-][0-9]+[a-zA-Z]?)?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core Cleaning Functions
# ---------------------------------------------------------------------------


def normalize_whitespace(s: Optional[Any]) -> Optional[str]:
    """Strip leading/trailing whitespace and collapse internal repeated whitespace.

    Also handles non-breaking spaces, tabs, and newlines.
    Returns None if input is None.
    """
    if s is None:
        return None
    val = str(s)
    # Replace non-breaking space with standard space
    val = val.replace("\u00a0", " ")
    # Collapse all whitespace characters (spaces, tabs, newlines) into single space
    cleaned = re.sub(r"\s+", " ", val).strip()
    return cleaned


def normalize_missing(s: Optional[Any]) -> Optional[str]:
    """Normalize missing values consistently to None.

    Treats None, NaN, empty strings, whitespace-only strings, and literal
    strings like 'NaN', 'nan', 'None', 'null', 'n/a' as missing -> returns None.
    """
    if s is None:
        return None

    # Handle pandas / numpy NaN/NA types
    if isinstance(s, float):
        if math.isnan(s):
            return None

    val = str(s).strip()
    if not val:
        return None

    if val.lower() in MISSING_STRINGS:
        return None

    return val


def is_bare_domain(name: Optional[Any]) -> bool:
    """Flag whether a business name string appears to be a bare domain name.

    Returns True if the string matches a standalone domain pattern (e.g. 'amazon.com',
    'my-business.co.in') with no other descriptive business tokens.
    """
    val = normalize_missing(name)
    if val is None:
        return False
    clean = normalize_whitespace(val)
    if clean is None:
        return False
    return bool(DOMAIN_PATTERN.match(clean))


def clean_name(name: Optional[Any]) -> Optional[str]:
    """Clean and normalize business entity name for comparison.

    Steps applied:
    1. Check for missing values (returns None if missing).
    2. Unicode NFKC normalization (preserves non-ASCII / Devanagari / multilingual scripts).
    3. Lowercase conversion for standardized matching.
    4. Strip leading junk tokens (e.g. leading dashes '--', stray punctuation, asterisks).
    5. Normalize '&' <-> 'and' (converts '&' to 'and' with proper token spacing).
    6. Expand legal suffix/prefix abbreviations via synonym table (both leading and trailing).
    7. Normalize and collapse residual whitespace.

    Note: Non-ASCII characters (e.g. Hindi / Devanagari) survive unchanged.
    """
    val = normalize_missing(name)
    if val is None:
        return None

    # 1. Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", val)

    # 2. Lowercase for comparison copy
    normalized = normalized.lower()

    # 3. Strip leading junk tokens (leading dashes, bullets, quotes, stray punctuation runs)
    normalized = re.sub(r"^[\s\-_.,:;*#~!?/\\+|=]+", "", normalized)
    normalized = re.sub(r"[\s\-_.,:;*#~!?/\\+|=]+$", "", normalized)

    # 4. Normalize '&' to 'and'
    # Handles standard '&' and fullwidth '＆'
    normalized = re.sub(r"\s*[&＆]\s*", " and ", normalized)

    # 5. Expand legal prefix/suffix abbreviations (both leading and trailing positions)
    for pattern, expansion in LEGAL_SYNONYMS:
        # Check and expand across whole string via word boundary regex
        normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)

    # 6. Collapse internal whitespace and strip
    cleaned = normalize_whitespace(normalized)
    return cleaned if cleaned else None


def clean_address(address: Optional[Any]) -> Dict[str, Optional[str]]:
    """Clean and parse address into normalized components.

    Accepts missing/None/empty addresses without raising.
    Returns a dictionary containing:
    - 'cleaned_address': casefolded and abbreviation-expanded address string.
    - 'landmark': extracted landmark phrase (e.g., 'near metro station') or None.
    - 'postal_code': extracted postal/ZIP code string or None.
    - 'street_number': extracted street/unit/plot number or None.
    """
    val = normalize_missing(address)
    if val is None:
        return {
            "cleaned_address": None,
            "landmark": None,
            "postal_code": None,
            "street_number": None,
        }

    # Unicode NFKC normalization and casefold (handles ALL-CAPS sources)
    addr_str = unicodedata.normalize("NFKC", val).casefold()
    addr_str = normalize_whitespace(addr_str) or ""

    # 1. Extract Landmark Phrases ("near X", "opp. X", "behind X", etc.)
    landmark: Optional[str] = None
    landmark_match = LANDMARK_REGEX.search(addr_str)
    if landmark_match:
        landmark = normalize_whitespace(landmark_match.group(0))

    # 2. Extract Postal Code
    postal_code: Optional[str] = None
    postal_match = POSTAL_CODE_REGEX.search(addr_str)
    if postal_match:
        postal_code = normalize_whitespace(postal_match.group(0))

    # 3. Extract Street / Plot / Unit Number
    street_number: Optional[str] = None
    street_match = STREET_NUMBER_REGEX.search(addr_str)
    if street_match:
        street_number = normalize_whitespace(street_match.group(1))

    # 4. Expand address abbreviations
    expanded_addr = addr_str
    for pattern, expansion in ADDRESS_ABBREVIATIONS:
        expanded_addr = re.sub(pattern, expansion, expanded_addr, flags=re.IGNORECASE)

    # Clean up redundant commas, punctuation runs, and whitespace
    expanded_addr = re.sub(r"[,;]+", ", ", expanded_addr)
    expanded_addr = re.sub(r"\s+", " ", expanded_addr)
    expanded_addr = re.sub(r"^[\s,.-]+|[\s,.-]+$", "", expanded_addr).strip()

    cleaned_address = expanded_addr if expanded_addr else None

    return {
        "cleaned_address": cleaned_address,
        "landmark": landmark,
        "postal_code": postal_code,
        "street_number": street_number,
    }


def clean_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single entity record row.

    Preserves all original input fields unchanged and appends cleaned / flag fields.

    Returned fields include:
    - Original fields: row['entity_id'], row['business_name'], row['business_address'], row['country'], etc.
    - 'cleaned_entity_id': whitespace-normalized entity ID (strips leading/trailing spaces).
    - 'cleaned_name': NFKC normalized, lowercased, legal-expanded business name.
    - 'is_bare_domain': boolean flag indicating if business name is a bare domain.
    - 'cleaned_address': casefolded, abbreviation-expanded normalized address.
    - 'landmark': extracted landmark description (or None).
    - 'postal_code': extracted postal / ZIP code (or None).
    - 'street_number': extracted street / unit number (or None).
    - 'cleaned_country': trimmed, uppercase country code / name (or None).
    """
    # Shallow copy input row to preserve original fields untouched
    record = dict(row)

    # 1. Clean entity_id (handle leading/trailing spaces in real data)
    raw_entity_id = row.get("entity_id")
    cleaned_id = normalize_whitespace(raw_entity_id)
    record["cleaned_entity_id"] = normalize_missing(cleaned_id)

    # 2. Clean business_name & detect bare domain
    raw_name = row.get("business_name")
    record["cleaned_name"] = clean_name(raw_name)
    record["is_bare_domain"] = is_bare_domain(raw_name)

    # 3. Clean business_address & parse subcomponents
    raw_address = row.get("business_address")
    addr_components = clean_address(raw_address)
    record["cleaned_address"] = addr_components["cleaned_address"]
    record["landmark"] = addr_components["landmark"]
    record["postal_code"] = addr_components["postal_code"]
    record["street_number"] = addr_components["street_number"]

    # 4. Clean country
    raw_country = row.get("country")
    norm_country = normalize_whitespace(normalize_missing(raw_country))
    record["cleaned_country"] = norm_country.upper() if norm_country else None

    return record


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized and column-wise cleaning for a DataFrame or DataFrame chunk.

    Preserves all original input columns and appends cleaned / flag columns
    without creating expensive row-wise dictionaries.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame chunk containing source entity records.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with original columns preserved and new cleaned/flag columns.
    """
    out_df = df.copy()

    # 1. Clean entity_id (strip whitespace and normalize missing)
    if "entity_id" in df.columns:
        out_df["cleaned_entity_id"] = (
            df["entity_id"]
            .astype(object)
            .map(lambda x: normalize_missing(normalize_whitespace(x)))
        )
    else:
        out_df["cleaned_entity_id"] = None

    # 2. Clean business_name & detect bare domain
    if "business_name" in df.columns:
        out_df["cleaned_name"] = df["business_name"].astype(object).map(clean_name)
        out_df["is_bare_domain"] = df["business_name"].astype(object).map(is_bare_domain)
    else:
        out_df["cleaned_name"] = None
        out_df["is_bare_domain"] = False

    # 3. Clean business_address & extract subcomponents
    if "business_address" in df.columns:
        addr_results = df["business_address"].astype(object).map(clean_address)
        out_df["cleaned_address"] = [res["cleaned_address"] for res in addr_results]
        out_df["landmark"] = [res["landmark"] for res in addr_results]
        out_df["postal_code"] = [res["postal_code"] for res in addr_results]
        out_df["street_number"] = [res["street_number"] for res in addr_results]
    else:
        out_df["cleaned_address"] = None
        out_df["landmark"] = None
        out_df["postal_code"] = None
        out_df["street_number"] = None

    # 4. Clean country
    if "country" in df.columns:
        out_df["cleaned_country"] = (
            df["country"]
            .astype(object)
            .map(lambda x: (normalize_whitespace(normalize_missing(x)) or "").upper() or None)
        )
    else:
        out_df["cleaned_country"] = None

    return out_df
