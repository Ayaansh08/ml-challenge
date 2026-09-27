# Decisions Log

* 2026-09-26 | **Retain Hand-Rolled Blocking** | Rather than rewriting the initial blocking to use FAISS + MinHash LSH (as originally spec'd in v4 architecture), we explicitly decided to keep the hand-rolled inverted index (n-grams, country, etc.) to get a working baseline fast without GPU dependency.
* 2026-09-26 | **DuckDB for Blocking Sharding** | Implemented `duckdb` for out-of-core streaming and deduplication in `blocking.py` to prevent OOM errors on the 9M-record dataset without massive memory overhead.
* 2026-09-26 | **Remove Generic Country Keys** | To solve RapidFuzz stalling on combinatorial explosions, we dropped the isolated `ctry_{country}` blocking key, preserving composite keys (e.g. `ctrychar_`).
* 2026-09-26 | **Strict Train/Test Isolation** | Ensured the blocking phase enforces absolute file-prefix isolation (`train` vs `test`) to prevent cross-contamination in candidate generation.
* 2026-09-26 | **No External APIs** | Re-confirmed strict adherence to the challenge rules: no external lookup tables or web APIs will be used for resolution.
