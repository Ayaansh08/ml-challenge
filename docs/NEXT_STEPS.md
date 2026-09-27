# Next Steps

## NOW
- Validate candidate statistics from the full `blocking.py` run on the ~9M row dataset. (Pipeline is actively running).
- Verify S2S3 combinations are 0 and no duplicates exist in `candidate_pairs.parquet`.

## NEXT
- Connect teammate's Bi-Encoder (`train_biencoder.py` / `pair_construction.py`).
- Generate embeddings and run dense retrieval.
- Establish the first measurable validation statistic (Recall@K).

## AFTER THAT
- Implement cheap Pre-rank.
- Implement Cross-Encoder (`matching.py`).
- Implement Threshold calibration.
- Implement final Aggregation / submission formatting (`aggregation.py`).

## BLOCKED
- None. (Combinatorial explosion block removed by omitting generic country keys. Waiting for blocking run to finish).

## DONE
- 2026-09-26: Project scaffolding and ingestion setup.
- 2026-09-26: Data cleaning module implemented (`cleaning.py`) and functionally verified. Fixed NULL parity bugs.
- 2026-09-26: Phase 3 Blocking fully fixed (memory capped, duckdb sharding, exact max-pair limits, no cross-contamination, no S2-S3). Tested successfully at Level 2 & 3.
