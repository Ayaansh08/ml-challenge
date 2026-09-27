# AI Coding Guide

This document outlines standard operating procedures and safety rules for any AI agent interacting with this repository.

## Operational Workflow

1. Read `PROJECT_CONTEXT.md` to understand constraints and metrics.
2. Read `IMPLEMENTATION_STATUS.md` and `NEXT_STEPS.md` for current progress and immediate priorities.
3. Inspect current code and open tasks (`hive/tasks.json`, `hive/board.md`).
4. Formulate a plan and explicitly state it before writing code.
5. Implement changes in small, deterministic functions.
6. Test using the testing ladder defined in `TEST_PLAN.md`.
7. Analyze runtime, memory, and output artifacts.
8. Report results concisely.
9. Update `IMPLEMENTATION_STATUS.md` and `NEXT_STEPS.md` if status changed.

## Safety Rules

- **Never** silently delete local work.
- **Never** reset/checkout destructive Git operations without explicit human approval.
- **Never** overwrite teammate work blindly.
- **Never** rewrite a working module merely because a "cleaner" design exists. (Fix bugs instead of rewriting if it works).
- **Never** introduce a dependency without checking requirements/license constraints (must be MIT/Apache 2.0 compatible).
- **Never** launch expensive full-scale training/testing without human approval.
- **Never** claim something was tested if it was not.
- **Never** invent or hallucinate metrics, test results, or candidate counts.
- **Never** mix train and test data accidentally. Check file prefixes.
- **Never** use external entity-resolution APIs/lookups.

## Coding Style Preferences

**Prefer:**
- Small functions with single responsibilities.
- Explicit inputs/outputs (type hints).
- Deterministic behavior.
- Resumable processing (sharding, checkpointing for expensive stages).
- Bounded memory limits (stream data, don't accumulate).
- Informative logging (info, stats, warnings).
- Atomic writes where appropriate.

**Avoid:**
- Unnecessary abstractions (e.g. over-engineered OOP).
- Global state mutation.
- Giant DataFrames loaded into memory at once.
- Silent exception swallowing.
- Hidden internet downloads.
- Unnecessary refactors.
