# Test Plan

Follow this staged testing ladder for all changes. Do not jump to Level 4/5 without passing the lower levels.

## LEVEL 0: Syntax / Import Validation
Ensure the module compiles and imports cleanly.
*Commands:*
- `python -m py_compile <file>`
- `python -c "import src.pipeline.<module>"`

## LEVEL 1: Unit Tests
Run existing pytest fixtures.

## LEVEL 2: Synthetic Test
Use a tiny, completely controlled mock dataset (e.g., 5-10 records).
*Verify:*
- IDs are correctly formatted.
- Source-pair rules are strictly obeyed.
- Expected candidate matches are generated.
- NO S2-S3 pairs exist.
- Deduplication works (0 duplicate final pairs).

## LEVEL 3: Small Real-Data Test
Execute the pipeline on a 50k-100k row sample (e.g., `outputs/phase0_cleaned`).
*Record & Report:*
- Total runtime
- Peak RSS memory
- Input rows / Output rows
- Raw candidate count
- Exceptions or errors

## LEVEL 4: Full Training-Scale Test
*Only launch after Levels 0-3 pass.* Run against the full ~9M row dataset. Monitor system memory closely.

## LEVEL 5: Validation / Model Test
Measure the actual machine learning pipeline metrics.
*Record & Report:*
- Recall@5, Recall@10, Recall@20
- Total Candidate Count
- Precision
- Recall
- Final F0.5 Score
- Total Runtime / Peak Memory

### Blocking-Specific PASS/FAIL Criteria
When modifying `blocking.py`, the following explicit criteria MUST be met and derivable from empirical statistics:
- S1-S2 Count: `> 0`
- S1-S3 Count: `> 0`
- S2-S3 Count: `= 0` (MUST BE ZERO)
- Duplicate Pair Count: `= 0` (MUST BE ZERO)
- Final Unique Candidate Count: `< Total possible combinations`
- Oversized Block Count: Recorded and explicitly handled.
