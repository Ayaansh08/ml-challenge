# Implementation Status

*Last updated: 2026-09-26*

## Module Status

| Module | Status | What works | Known gaps | Next action |
|---|---|---|---|---|
| `cleaning.py` | IMPLEMENTED | Null/NaN parity, regex lookaheads, glob discovery | N/A | Monitor on full dataset |
| `run_cleaning.py` | IMPLEMENTED | CLI execution, chunking | N/A | None |
| `blocking.py` | IMPLEMENTED | DuckDB sharding/dedup, memory limits, no S2-S3 pairs, oversized limits | Waiting for full scale run verification | Verify full scale output |
| `pair_construction.py` | PARTIALLY IMPLEMENTED | Scaffold pushed to main by teammate | Requires integration with pipeline | Hook up and test |
| `train_biencoder.py` | IMPLEMENTED | Pushed to main, uses MNRL and MiniLM | Not yet trained/validated on local full data | Run training locally |
| `features.py` | SCAFFOLD | Empty / placeholder | Missing dense/lexical feature extraction | Implement Bi-Encoder inference |
| `matching.py` | SCAFFOLD | Empty / placeholder | Missing Cross-Encoder implementation | Implement Cross-Encoder logic |
| `aggregation.py` | SCAFFOLD | Empty / placeholder | Missing final grouping / closure logic | Implement output formatting |
| `methodology.md` | PARTIALLY IMPLEMENTED | Initial Phase 1-3 documented | Missing Phase 4-6 documentation | Update as phases complete |

## Current Critical Path

1. Verify cleaning output on full dataset.
2. Fix/validate `blocking.py` on full dataset. *(In Progress)*
3. Produce `candidate_pairs.parquet`. *(In Progress)*
4. Validate candidate statistics.
5. Connect/use Bi-Encoder (`train_biencoder.py` / `pair_construction.py`).
6. Get first measurable validation statistic.
7. Add remaining matching stages (Cross-Encoder & Aggregation).
