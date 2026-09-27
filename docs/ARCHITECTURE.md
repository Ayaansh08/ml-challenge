# Architecture

This document describes both the *Current Implementation* (what actually exists in code) and the *Target Model 2 Architecture* (the documented goal).

## A. CURRENT IMPLEMENTATION

1. **Cleaning (`cleaning.py`):** Standardizes entity names, extracts legal suffixes, normalizes addresses, and maps nulls.
2. **Inverted-Index Blocking (`blocking.py`):** Hand-rolled exact-match inverted index using character/word n-grams, legal suffixes, and composites. Streams via PyArrow/DuckDB disk sharding to avoid memory crashes. Oversized blocks truncated via RapidFuzz string scoring.
3. **Candidate Pairs:** Cross-matches S1 against S2/S3. Never matches S2-S3. Emits raw deduplicated ID pairs.
4. **Downstream Stages:** Currently scaffolded / missing. Bi-encoder training scripts exist but are not yet wired into inference.

## B. TARGET MODEL 2 ARCHITECTURE

1. **Cleaning:** Base normalization.
2. **Bi-Encoder Retrieval:** Dense embedding extraction (using `sentence-transformers/all-MiniLM-L6-v2`) fine-tuned via `MultipleNegativesRankingLoss`. Uses hard-negative mining. FP16 enabled.
3. **Lexical Safety Channel:** TF-IDF / BM25 fallback to catch language-agnostic dense retrieval failures (especially for unseen countries like France).
4. **Union / Cap:** Merge dense and lexical candidate streams, applying a hard maximum ceiling per S1 entity.
5. **Pre-Rank:** Cheap localized scoring (e.g., token overlap) to further prune the unioned set before heavy inference.
6. **Cross-Encoder:** Heavy pair-wise sequence classification to determine exact matching probability.
7. **Calibration:** Threshold tuning to maximize F0.5.
8. **Output:** Transitive closure and formatted predictions.
