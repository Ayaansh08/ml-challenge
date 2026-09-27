# Business Entity Resolution Methodology

*This is a living document representing the methodologies, architecture decisions, and known gaps for the Amazon ML Challenge 2026. It is updated incrementally as each phase completes.*

## 1. Problem statement & constraints
- **Task:** Business entity resolution across `Source 1` (cleaner reference source) versus `Source 2` and `Source 3` (noisy sources).
- **Metric:** Macro-averaged F0.5 per S1 entity.
- **Constraints:** 
  - Open-set country field (e.g., France is unseen in training).
  - No external lookups or APIs permitted.
  - Final model parameters must be <= 8B.
  - All code/dependencies must be under MIT or Apache-2.0 licenses.
  - Limit of 5 submissions per day.

## 2. Data cleaning
- **Normalization applied:**
  - Standardized whitespace and NFKC Unicode normalization.
  - Legal-suffix expansion and bare-domain detection.
  - Postal code and street number extraction.
  - Standardized ALL-CAPS text and missing-address handling.
- **Validation & Bug Fixes:** 
  - Found and fixed a trailing-period regression bug during suffix expansion by switching to a regex lookahead logic `(?=\s|$)`.
  - Found and fixed a None/NaN parity bug between row-wise iteration and vectorized paths (standardized exclusively on `NaN` checks via `pd.isna()`).
  - *These fixes were verified via unit tests (`tests/test_cleaning.py`), ensuring deterministic downstream behavior.*

## 3. Candidate generation / blocking
- **Implementation Strategy:** This pipeline currently uses a **hand-rolled inverted-index blocking approach** (comprising character n-grams, word n-grams, country codes, legal-suffix, postal codes, and composite keys).
  - *Note on Architecture Shift:* This is **NOT** the originally-scoped FAISS/embedding-based ANN design. The shift was a deliberate simplification to get a working baseline fast without requiring a GPU dependency, given the 72-hour window constraints.
- **Trade-offs:** Trading away a semantic/embedding-based recall channel means typo- or translation-style near-duplicates that do not share exact n-grams or tokens may be missed. 
  - *Metric Gap:* Blocking recall has **NOT** been measured against held-out validation yet (PENDING).
- **Oversized-Block Handling:** To prevent memory/combinatorial explosion on hyper-dense blocks (e.g., generic tokens like 'limited' generating >20M pairs), the pipeline implements a RapidFuzz top-K truncation. It scores candidates using `fuzz.token_sort_ratio` against the S1 record and truncates to the `max_pairs` limit. 
  - *Gap:* This works to prevent OOMs but is a massive wall-clock bottleneck in Python for extremely dense keys.

## 4. Matching model
- **Implementation Strategy**: Extracted 30+ pairwise features mapping topological similarities across `cleaned_name`, `cleaned_address`, and `country` (e.g. `token_sort_ratio`, `Jaro-Winkler`, `Levenshtein`). 
- **Classifier**: Used `LightGBM` binary classifier to train on ground-truth subsets.
- **Handling Imbalance**: Parameter `scale_pos_weight` accounts for the large volume of negative pairs relative to positive match pairs during training.

## 5. Aggregation & Thresholding
- **Threshold Tuning**: Employs configurable cutoff boundaries on LightGBM output predictions.
- **Aggregation Strategy**: Evaluates matched pairs as an undirected graph, utilizing `networkx.connected_components` to extract distinct equivalence classes (clusters). Singletons are cleanly extracted in `O(N)`.
- **Output Validation**: Formats the final submission precisely to match test entity counts in `submission.tsv`.

## 6. Fair-play compliance
- **External lookups:** No external API or data lookups are used at any stage in this pipeline.
- **Third-Party Components & Licenses:**
  - `pandas` (BSD 3-Clause)
  - `pyarrow` / `pyarrow.parquet` (Apache-2.0)
  - `rapidfuzz` (MIT)
  - `networkx` (BSD 3-Clause)
  - `LightGBM` (MIT)
- **Model Size:** The pipeline logic + matching tree ensemble operates safely within the competition's 8B parameter memory limits.

## 7. Known limitations
- **Country-bucket handling for France:** The country field is open-set and France is explicitly unseen in training. Evaluated via proxy cross-validation on French localized terms.
- **Candidate-set size vs. recall trade-off:** Using max-pairs limits OOMs but requires balancing threshold tuning for recall.

## 8. Results & Pipeline Status
- **Status**: The pipeline components (Cleaning, Blocking, Features, Matching, Aggregation) are 100% complete. End-to-end integration is in progress via `run_pipeline.py`.
- **Validation Score**: *PENDING* (Final integrated metric evaluation is pending on the validation splits).
