"""Data cleaning, normalization, and validation functions for business entities.

Pure functions designed for unit testing with zero side effects.
Handles messy real-world entity resolution patterns including:
- Leading/trailing whitespace on entity IDs and text fields
- Missing/NaN representation variants (None, NaN, "", "NaN", "nan", "null")
- Multi-script preservation (NFKC Unicode normalization, Devanagari/non-Latin scripts)
- Cross-script transliteration (Indic scripts → Latin for matching)
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

# Optional import for Indic script transliteration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False
    sanscript = None

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
# Use lookahead (?=\s|$) instead of trailing \b to handle periods correctly
LEGAL_SYNONYMS: List[Tuple[str, str]] = [
    # Multi-word patterns first to prevent premature single-word matching
    (r"\bpvt\.?\s*ltd\.?(?=\s|$)", "private limited"),
    (r"\bpte\.?\s*ltd\.?(?=\s|$)", "private limited"),
    (r"\bpty\.?\s*ltd\.?(?=\s|$)", "proprietary limited"),
    (r"\bprivate\s+limited\b", "private limited"),
    (r"\bpublic\s+limited\s+company\b", "public limited company"),
    (r"\blimited\s+liability\s+company\b", "limited liability company"),
    (r"\blimited\s+liability\s+partnership\b", "limited liability partnership"),
    (r"\bl\.?l\.?c\.?(?=\s|$)", "limited liability company"),
    (r"\bl\.?l\.?p\.?(?=\s|$)", "limited liability partnership"),
    (r"\bp\.?l\.?c\.?(?=\s|$)", "public limited company"),
    (r"\bltd\.?(?=\s|$)", "limited"),
    (r"\blimited\b", "limited"),
    (r"\binc\.?(?=\s|$)", "incorporated"),
    (r"\bincorporated\b", "incorporated"),
    (r"\bcorp\.?(?=\s|$)", "corporation"),
    (r"\bcorporation\b", "corporation"),
    (r"\bco\.?(?=\s|$)", "company"),
    (r"\bcompany\b", "company"),
    (r"\bg\.?m\.?b\.?h\.?(?=\s|$)", "gmbh"),
    (r"\bs\.?a\.?r\.?l\.?(?=\s|$)", "sarl"),
    (r"\bs\.?a\.?(?=\s|$)", "sa"),
    (r"\bs\.?r\.?l\.?(?=\s|$)", "srl"),
    (r"\bs\.?p\.?a\.?(?=\s|$)", "spa"),
    (r"\bb\.?v\.?(?=\s|$)", "bv"),
    (r"\bn\.?v\.?(?=\s|$)", "nv"),
    (r"\bpvt\.?(?=\s|$)", "private"),
    (r"\bpte\.?(?=\s|$)", "private"),
    (r"\bpty\.?(?=\s|$)", "proprietary"),
# Indic legal suffixes (Devanagari, transliterated forms)
    # Use (?<!\S)/(?!\S) instead of \b for Unicode script compatibility
    (r"(?<!\S)प्राइवेट\s+लिमिटेड(?!\S)", "private limited"),
    (r"(?<!\S)प्रा\.\s*लि\.(?!\S)", "private limited"),
    (r"(?<!\S)लिमिटेड(?!\S)", "limited"),
    (r"(?<!\S)एलएलपी(?!\S)", "llp"),
    (r"(?<!\S)प्राइवेट(?!\S)", "private"),
    (r"(?<!\S)पब्लिक\s+लिमिटेड(?!\S)", "public limited"),
    (r"(?<!\S)सार्वजनिक\s+कंपनी(?!\S)", "public company"),
    # Bengali legal suffixes
    (r"(?<!\S)প্রাইভেট\s+লিমিটেড(?!\S)", "private limited"),
    (r"(?<!\S)লিমিটেড(?!\S)", "limited"),
    (r"(?<!\S)এলএলপি(?!\S)", "llp"),
    # Gurmukhi legal suffixes
    (r"(?<!\S)ਪ੍ਰਾਈਵੇਟ\s+ਲਿਮਟਿਡ(?!\S)", "private limited"),
    (r"(?<!\S)ਲਿਮਟਿਡ(?!\S)", "limited"),
    (r"(?<!\S)ਐਲਐਲਪੀ(?!\S)", "llp"),
    # Gujarati legal suffixes
    (r"(?<!\S)પ્રાઇવેટ\s+લિમિટેડ(?!\S)", "private limited"),
    (r"(?<!\S)લિમિટેડ(?!\S)", "limited"),
    (r"(?<!\S)એલએલપી(?!\S)", "llp"),
    # Tamil legal suffixes
    (r"(?<!\S)பிரைவேட்\s+லிமிடெட்(?!\S)", "private limited"),
    (r"(?<!\S)லிமிடெட்(?!\S)", "limited"),
    (r"(?<!\S)எல்எல்பி(?!\S)", "llp"),
    # Telugu legal suffixes
    (r"(?<!\S)ప్రైవెట్\s+లిమిటెడ్(?!\S)", "private limited"),
    (r"(?<!\S)లిమిటెడ్(?!\S)", "limited"),
    (r"(?<!\S)ఎల్ఎల్పి(?!\S)", "llp"),
    # Kannada legal suffixes
    (r"(?<!\S)ಪ್ರೈವೇಟ್\s+ಲಿಮಿಟೆಡ್(?!\S)", "private limited"),
    (r"(?<!\S)ಲಿಮಿಟೆಡ್(?!\S)", "limited"),
    (r"(?<!\S)ಎಲ್ಎಲ್ಪಿ(?!\S)", "llp"),
    # Malayalam legal suffixes
    (r"(?<!\S)പ്രൈവറ്റ്\s+ലിമിറ്റഡ്(?!\S)", "private limited"),
(r"(?<!\S)ലിമിറ്റഡ്(?!\S)", "limited"),
    (r"(?<!\S)എല്‍എല്‍പി(?!\S)", "llp"),
    # IAST transliterated Indic legal suffixes (from indic-transliteration)
    # Devanagari
    (r"\bpr[āa]ive[ṭṭ]a\s+l[īi]mi[ṭṭ]e[ḍḍ]a\b", "private limited"),
    (r"\bpr[āa]\.\s*l[īi]\.\b", "private limited"),
    (r"\bl[īi]mi[ṭṭ]e[ḍḍ]a\b", "limited"),
    (r"\bel[ell]p[īi]\b", "llp"),
    (r"\bpr[āa]ive[ṭṭ]a\b", "private"),
    (r"\bpabl[īi]k\s+l[īi]mi[ṭṭ]e[ḍḍ]a\b", "public limited"),
    (r"\bs[āa]rvajan[īi]k\s+k[āa]mpan[īi]\b", "public company"),
    # Bengali
    (r"\bpr[āa]ibhe[ṭṭ]e\s+l[īi]mi[ṭṭ]e[ḍḍ]e\b", "private limited"),
    (r"\bl[īi]mi[ṭṭ]e[ḍḍ]e\b", "limited"),
    (r"\bel[ell]p[īi]\b", "llp"),
    # Gurmukhi
    (r"\bpr[āa]iv[ēe]ṭe\s+l[īi]maṭiḍ\b", "private limited"),
    (r"\bl[īi]maṭiḍ\b", "limited"),
    (r"\baillp[īi]\b", "llp"),
    # Gujarati
    (r"\bpr[āa]iv[ēe]ṭ\s+l[īi]miṭeḍ\b", "private limited"),
    (r"\bl[īi]miṭeḍ\b", "limited"),
    (r"\bel[ell]p[īi]\b", "llp"),
    # Tamil
    (r"\bpiraiv[ēe]ṭ\s+limiḍeḍ\b", "private limited"),
    (r"\blimiḍeḍ\b", "limited"),
    (r"\bel[l]lbh[īi]\b", "llp"),
    # Telugu
    (r"\bpr[āa]iv[ēe]ṭ\s+limiṭeḍ\b", "private limited"),
    (r"\blimiṭeḍ\b", "limited"),
    (r"\bel[l]lp[īi]\b", "llp"),
    # Kannada
    (r"\bpraive[ṭṭ]\s+l[īi]miṭeḍ\b", "private limited"),
    (r"\bl[īi]miṭeḍ\b", "limited"),
    (r"\bel[l]lp[īi]\b", "llp"),
    # Malayalam
    (r"\bpraiv[ēe]ṭ\s+limiṭṭ\b", "private limited"),
    (r"\blimiṭṭ\b", "limited"),
    (r"\bel[l]lp[īi]\b", "llp"),
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
# Indic Script Detection & Transliteration
# ---------------------------------------------------------------------------

# Unicode script ranges for Indic scripts
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "oriya": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}

# Map script names to sanscript scheme codes
SANSCRIPT_SCHEMES = {
    "devanagari": sanscript.DEVANAGARI if INDIC_TRANSLITERATION_AVAILABLE else None,
    "bengali": sanscript.BENGALI if INDIC_TRANSLITERATION_AVAILABLE else None,
    "gurmukhi": sanscript.GURMUKHI if INDIC_TRANSLITERATION_AVAILABLE else None,
    "gujarati": sanscript.GUJARATI if INDIC_TRANSLITERATION_AVAILABLE else None,
    "oriya": sanscript.ORIYA if INDIC_TRANSLITERATION_AVAILABLE else None,
    "tamil": sanscript.TAMIL if INDIC_TRANSLITERATION_AVAILABLE else None,
    "telugu": sanscript.TELUGU if INDIC_TRANSLITERATION_AVAILABLE else None,
    "kannada": sanscript.KANNADA if INDIC_TRANSLITERATION_AVAILABLE else None,
    "malayalam": sanscript.MALAYALAM if INDIC_TRANSLITERATION_AVAILABLE else None,
}

# Target transliteration scheme (IAST for romanization)
TARGET_SCHEME = sanscript.IAST if INDIC_TRANSLITERATION_AVAILABLE else None


def detect_script(text: str) -> Optional[str]:
    """Detect the primary Indic script in a text string.
    
    Returns the script name (e.g., 'devanagari', 'tamil') or None if no Indic script detected.
    If multiple scripts present, returns the one with the most characters.
    """
    if not text:
        return None
    
    script_counts: Dict[str, int] = {script: 0 for script in SCRIPT_RANGES}
    
    for char in text:
        cp = ord(char)
        for script, (start, end) in SCRIPT_RANGES.items():
            if start <= cp <= end:
                script_counts[script] += 1
                break
    
    if not any(script_counts.values()):
        return None
    
    return max(script_counts, key=script_counts.get)


def transliterate_to_latin(text: Optional[str]) -> Optional[str]:
    """Transliterate Indic script text to Latin (IAST romanization).
    
    If indic-transliteration is not available or no Indic script detected,
    returns the original text unchanged.
    
    Parameters
    ----------
    text : Optional[str]
        Input text that may contain Indic scripts.
        
    Returns
    -------
    Optional[str]
        Romanized text, or original text if no transliteration needed/possible.
    """
    if not text or not INDIC_TRANSLITERATION_AVAILABLE:
        return text
    
    script = detect_script(text)
    if script is None or script not in SANSCRIPT_SCHEMES:
        return text
    
    source_scheme = SANSCRIPT_SCHEMES[script]
    if source_scheme is None:
        return text
    
    try:
        return transliterate(text, source_scheme, TARGET_SCHEME)
    except Exception:
        # Fallback: return original text if transliteration fails
        return text


def clean_name(name: Optional[Any]) -> Optional[str]:
    """Clean and normalize business entity name for comparison.

    Steps applied:
    1. Check for missing values (returns None if missing).
    2. Unicode NFKC normalization (preserves non-ASCII / Devanagari / multilingual scripts).
    3. Expand native Indic legal suffixes BEFORE transliteration (so native patterns match).
    4. Transliterate Indic scripts to Latin (IAST) for cross-script matching.
    5. Lowercase conversion for standardized matching.
    6. Strip leading junk tokens (e.g. leading dashes '--', stray punctuation, asterisks).
    7. Normalize '&' <-> 'and' (converts '&' to 'and' with proper token spacing).
    8. Expand legal suffix/prefix abbreviations via synonym table (both leading and trailing).
    9. Normalize and collapse residual whitespace.

    Note: Non-ASCII characters (e.g. Hindi / Devanagari) are transliterated to Latin
    for matching compatibility, while preserving original in source data.
    """
    val = normalize_missing(name)
    if val is None:
        return None

    # 1. Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", val)

    # 2. Expand native Indic legal suffixes BEFORE transliteration
    # This allows native script patterns in LEGAL_SYNONYMS to match
    for pattern, expansion in LEGAL_SYNONYMS:
        normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)

    # 3. Transliterate Indic scripts to Latin for cross-script matching
    normalized = transliterate_to_latin(normalized)

    # 4. Lowercase for comparison copy
    normalized = normalized.lower()

    # 5. Strip leading junk tokens (leading dashes, bullets, quotes, stray punctuation runs)
    normalized = re.sub(r"^[\s\-_.,:;*#~!?/\\+|=]+", "", normalized)
    normalized = re.sub(r"[\s\-_.,:;*#~!?/\\+|=]+$", "", normalized)

    # 6. Normalize '&' to 'and'
    # Handles standard '&' and fullwidth '＆'
    normalized = re.sub(r"\s*[&＆]\s*", " and ", normalized)

    # 7. Expand legal prefix/suffix abbreviations (both leading and trailing positions)
    # This catches Latin patterns and transliterated forms
    for pattern, expansion in LEGAL_SYNONYMS:
        normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)

    # 8. Collapse internal whitespace and strip
    cleaned = normalize_whitespace(normalized)
    return cleaned if cleaned else None


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
    # Transliterate Indic scripts to Latin for cross-script matching
    addr_str = transliterate_to_latin(addr_str)
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
