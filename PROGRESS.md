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

---

## Phase 2b - Pair Construction for Model 2 (Bi-Encoder + Cross-Encoder)
*Date: 2026-09-26*

### What Was Implemented
- **Pure Function Module (`src/pipeline/pair_construction.py`)**:
  - `load_and_clean_sources(s1_path, s2_path, s3_path)`: Loads S1, S2, S3 TSVs, executes `clean_dataframe`, and enforces strict prefix validation (`S1-`, `S2-`, `S3-`) with descriptive error reporting.
  - `build_positive_pairs(ground_truth_path, cleaned_s1, cleaned_s2, cleaned_s3)`: Maps ground-truth matches into formatted pairs (`"{cleaned_name} [SEP] {cleaned_address}"`), tagging source origin (`s2`/`s3`), and tracking/logging singletons.
  - `grouped_train_val_split(positive_pairs_df, val_fraction, random_seed)`: Partitions pairs strictly by unique `s1_entity_id` using `GroupShuffleSplit`, asserting zero entity overlap between train and val.
  - `build_country_stratified_proxy_split(cleaned_s1, val_entity_ids)`: Implements the France-proxy validation set by transforming US/India validation entities with French legal corporate forms (`SARL`, `SAS`, `EURL`, `SCI`) and street tokens (`Boulevard`, `Avenue`, `Rue`, `Route`).
  - `mine_hard_negatives(anchor_entity_id, anchor_text, candidate_pool_df, current_embeddings, k)`: Scaffold for cosine similarity hard negative retrieval from candidate embedding pools, excluding true positives and self-matches.
  - `write_pair_dataset(positive_pairs_df, output_dir, ...)`: Generates `train_positives.parquet`, `val_positives.parquet`, `france_proxy_val.parquet`, and `pair_summary.json`.
- **Unit Test Suite (`tests/test_pair_construction.py`)**:
  - 8 unit tests validating prefix validation errors, singleton skipping, grouped split zero-overlap assertions, France proxy token expansions, hard negative ranking, and dataset exports.
- **End-to-End Sample Runner (`src/pipeline/run_sample_pair_construction.py`)**:
  - Verified end-to-end on a 10,000 S1 entity sample against real data in `data/train/`.

### 10k Sample Benchmark Results
- **S1 Rows Sampled**: 10,000 (10,000 unique S1 entities)
- **Ground Truth Matches Extracted**: 16,765 S2 records and 17,987 S3 records
- **Singletons Skipped**: 560 (5.60% of S1 sample)
- **Total Positive Pairs Built**: 34,752 (29,481 Train / 5,271 Validation)
- **Unique S1 Entities in Train/Val**: 8,024 Train / 1,416 Validation (Zero overlap)
- **France Proxy Validation Entities**: 1,416
- **Country Breakdown**: US: 20,936 pairs, INDIA: 13,816 pairs

---

## Phase 3 - Bi-Encoder Contrastive Fine-Tuning (Model 2)
*Date: 2026-09-26*

### Backbone Model Card & Licensing
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (6 layers, 384-dimensional dense representations, 22.7M parameters).
- **License**: Apache 2.0 (permissive commercial & research license).
- **Language & French Coverage**: Pretrained predominantly on English sentence pairs (~1B+ pairs). While WordPiece tokenization handles French/Latin text without crashing, semantic accuracy on foreign abbreviations (`SARL`, `BVD`) is weaker.
  * *Mitigation Plan*: TF-IDF/BM25 token-level indexing channel in Phase 4 handles French corporate lexical overlap; the bi-encoder provides dense semantic candidate blocking.

### What Was Implemented
- **Module (`src/pipeline/train_biencoder.py`)**:
  - `load_pair_dataset(train_parquet, val_parquet)`: Loads parquet pairs into sentence-transformers `InputExample` objects.
  - `compute_recall_at_k(model, val_df, candidate_pool_df, k_values=[5, 10, 20])`: Pure evaluation function computing Mean Recall@k against a candidate match pool via cosine similarity.
  - `train_biencoder(...)`: PyTorch contrastive fine-tuning loop utilizing `MultipleNegativesRankingLoss`, periodic candidate re-embedding and hard-negative mining via `mine_hard_negatives`, FP16 mixed precision (`torch.cuda.amp`), and gradient accumulation.
  - `run_comparison(base_checkpoint, fine_tuned_path, val_df, ...)`: Computes and displays side-by-side Recall@5, Recall@10, and Recall@20 comparison between frozen base and fine-tuned checkpoints with absolute delta and percentage gains.
  - `main()`: Full CLI entrypoint with configurable batch size, learning rate, FP16 flag, gradient accumulation, and `--compare-only` evaluation mode.
- **Unit Test Suite (`tests/test_train_biencoder.py`)**:
  - Deterministic unit tests with mock embeddings verifying dataset loading, cosine similarity ranking, and Recall@k score accuracy.

### Execution Instructions

```bash
# 1. Run unit tests
python -m unittest tests/test_train_biencoder.py

# 2. Fine-tune bi-encoder on 10k-sample pairs with FP16 & gradient accumulation (4-8GB VRAM GPU)
python src/pipeline/train_biencoder.py \
  --train-parquet outputs/sample_pairs/train_positives.parquet \
  --val-parquet outputs/sample_pairs/val_positives.parquet \
  --output-dir outputs/models/biencoder_all_minilm_l6_v2 \
  --base-checkpoint sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 2 \
  --batch-size 64 \
  --grad-accum-steps 2 \
  --lr 3e-5 \
  --fp16

# 3. Run standalone Recall@k comparison against saved fine-tuned checkpoint
python src/pipeline/train_biencoder.py \
  --val-parquet outputs/sample_pairs/val_positives.parquet \
  --base-checkpoint sentence-transformers/all-MiniLM-L6-v2 \
  --fine-tuned-model outputs/models/biencoder_all_minilm_l6_v2 \
  --compare-only
```
