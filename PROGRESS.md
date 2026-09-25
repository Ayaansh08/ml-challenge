# Project Progress & Implementation Log

## Phase 1 - Project Scaffolding & Ingestion Setup
*Date: 2026-09-25*
- Established clean modular directory layout (`/data`, `/src/pipeline`, `/src/utils`, `/notebooks`, `/outputs`, `/tests`).
- Pinned stable dependencies in `requirements.txt` (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `sentence-transformers`, `rapidfuzz`, `pyarrow`).
- Created `src/utils/io.py` with `load_source(path)` providing strict TSV string loading (`dtype=str`) and schema validation on required columns (`entity_id`, `business_name`, `business_address`, `country`).
- Created scratch inspection script `src/inspect_data.py`.

---

## Phase 2 - Data Cleaning
*Date: 2026-09-25*

### What Was Implemented
- **Pure Function Module (`src/pipeline/cleaning.py`)**:
  - `normalize_whitespace(s)`: Strips leading/trailing whitespace, non-breaking spaces (`\u00a0`), tabs, and collapses repeated whitespace runs.
  - `normalize_missing(s)`: Unifies missing indicators (`None`, `NaN`, `""`, whitespace, `"NaN"`, `"nan"`, `"null"`, `"n/a"`) to return `None` consistently.
  - `clean_name(name)`: Applies Unicode NFKC normalization, lowercase conversion, strips leading/trailing junk punctuation/bullets, normalizes `&`/`＆` to `and`, and expands legal suffixes/prefixes.
  - `is_bare_domain(name)`: Detects and flags bare domain names as business names using regular expression matching.
  - `clean_address(address)`: Casefolds address text, safely handles missing values, expands street/unit abbreviations, extracts landmark phrases (`near`, `opp.`, `behind`, etc.), and parses postal/ZIP codes and street numbers.
  - `clean_record(row)`: Processes an entire row non-destructively, preserving all raw input fields while generating `cleaned_entity_id`, `cleaned_name`, `is_bare_domain`, `cleaned_address`, `landmark`, `postal_code`, `street_number`, and `cleaned_country`.
  - `clean_dataframe(df)`: Vectorized and column-wise DataFrame cleaner that operates on Series arrays instead of allocating millions of row dictionaries.
- **CLI Batch Processing Script (`src/pipeline/run_cleaning.py`)**:
  - Memory-bounded streaming architecture using chunked ingestion (`pd.read_csv(chunksize=...)`) and `pyarrow.parquet.ParquetWriter` incremental writes.
  - File-level resume capability to skip previously completed files.
  - Inter-file and inter-chunk garbage collection (`gc.collect()`).
- **Unit Test Suite (`tests/test_cleaning.py`)**:
  - Comprehensive unit tests covering missing address variations, literal `"NaN"` strings, padded `entity_id` strings, Devanagari script preservation, ALL-CAPS address casefolding, bare domain detection, prefix-position legal suffixes, ampersand normalization, and DataFrame/record parity.

### Noise Patterns Handled from Real Data
1. **Leading and Trailing Spaces on `entity_id`**: Real data sources contain leading spaces on ID fields (e.g. `"  S1-965667"`). `normalize_whitespace` and `clean_record` strip all ID padding.
2. **Missing and Pseudo-Missing Addresses (~3.3% in S2/S3)**: Handles missing address values across float `NaN`, empty strings `""`, whitespace, and literal `"NaN"`/`"nan"` without raising exceptions or breaking downstream tokenizers.
3. **Multilingual Script Support (Devanagari / Non-ASCII)**: Uses Unicode NFKC normalization without stripping non-ASCII characters, allowing Hindi/Devanagari and other native scripts to survive unchanged.
4. **ALL-CAPS Address Formatting**: Casefolds uppercase address records from legacy systems and expands abbreviations (e.g., `ST` -> `street`, `SUITE` -> `suite`, `RD` -> `road`).
5. **Bare Domain Names as Entity Names**: Detects and flags records where the entity name is a web domain (e.g. `example.com`, `mybusiness.co.in`) via `is_bare_domain=True`.
6. **Prefix and Suffix Legal Identifiers**: Handles legal forms appearing at both the start and end of names (e.g., `"LLC Moncada Enterprises"`, `"Pvt. EFS Logistics Ltd."`, `"Pvt Ltd Alpha Services"`).
7. **Ampersand Inconsistencies**: Normalizes `&` and fullwidth `＆` surrounded by arbitrary whitespace into `"and"`.
8. **Embedded Landmark Phrases in Addresses**: Extracts landmark phrases (`"near X"`, `"opp. X"`, `"behind X"`, `"next to X"`) into a distinct `landmark` field.
9. **Postal Codes and Street Numbers**: Extracts 6-digit PINs, 5/9-digit ZIPs, UK/Canadian postcodes, and plot/building numbers into separate structured fields.

### OOM Fix & Memory Optimization Log
*Date: 2026-09-25*
- **Problem**: Full-dataset loading of `train_source3.tsv` (5,285,603 rows) and `test_source3.tsv` (5,082,316 rows) triggered:
  `"Unable to allocate 444. MiB for an array with shape (11, 5285603) and data type object"`
- **Root Cause**:
  1. Full-table in-memory loading combined with row-wise dictionary generation (`to_dict(orient="records")`) created over 5.28M intermediate Python dict objects and references simultaneously in RAM.
  2. Construction of a single massive 11-column 2D numpy object array for the entire DataFrame exceeded contiguous heap allocation limits.
  3. Memory retained across sequentially processed files in the same process caused subsequent large files to fail even if earlier files succeeded.
- **Fix Implemented**:
  1. **Chunked Streaming Ingestion**: Configured `pd.read_csv(..., chunksize=chunksize, low_memory=False)` to process rows in configurable batches (default: 200,000 rows).
  2. **Streaming Parquet Writing**: Utilized `pyarrow.parquet.ParquetWriter` to stream each cleaned chunk table directly to disk incrementally.
  3. **Vectorized Column Operations**: Introduced `clean_dataframe` utilizing pandas Series operations and direct column unpacking, avoiding millions of Python dictionary allocations.
  4. **Active Memory Reclamation**: Explicit `del` on chunk data structures and `gc.collect()` between chunks and files to ensure peak RAM is strictly bounded to the size of a single chunk.
  5. **File-Level Resume**: Checks if destination `*_cleaned.parquet` already exists and is non-empty; skips re-processing already finished files (overridable with `--force`).

### Open Decisions Left to a Human
1. **Non-Latin Transliteration / Romanization**:
   - Currently, non-Latin script characters (such as Devanagari) are preserved in their native Unicode form.
   - *Decision needed*: If the ground truth matches Devanagari names with English/Latin phonetic spellings (e.g. `"रिलायंस"` vs `"Reliance"`), transliteration via libraries like `indic-transliteration` or multilingual embedding models will be required in Phase 3/4.
2. **Legal Suffix / Prefix Coverage & Jurisdiction-Specific Gaps**:
   - The current dictionary covers major US, UK, Indian, German, French, and Dutch corporate identifiers (`LLC`, `Pvt Ltd`, `Inc`, `Corp`, `GmbH`, `SA`, `SRL`, `SPA`, `BV`, `NV`, `Pty Ltd`).
   - *Decision needed*: Review whether specific regional designations (e.g., Chinese `Co., Ltd.`, Japanese `K.K.`, Nordic `AB`/`AS`, Middle Eastern `FZ-LLC`) need dedicated expansion rules.

### How to run

```bash
# Run unit tests
python -m unittest tests/test_cleaning.py

# Or with pytest (if installed)
pytest tests/test_cleaning.py -v

# Run chunked data cleaning (default chunk size: 200,000 rows, resumes automatically)
python src/pipeline/run_cleaning.py --input-dir data --output-dir outputs/cleaned

# Tune chunk size for memory-constrained environments
python src/pipeline/run_cleaning.py --input-dir data --output-dir outputs/cleaned --chunksize 100000

# Force re-processing of all files, ignoring existing non-empty output Parquet files
python src/pipeline/run_cleaning.py --input-dir data --output-dir outputs/cleaned --force
```
