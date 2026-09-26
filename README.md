# Business Entity Resolution Pipeline

A modular machine learning pipeline for business entity resolution across disparate data sources for the Amazon ML Challenge 2026.

## Repo Structure

```text
├── data/
│   ├── train/                 # Training raw datasets
│   └── test/                  # Test raw datasets
├── notebooks/                 # Exploratory notebooks and prototyping
├── outputs/                   # Processed parquets, submission files, evaluation reports
├── src/
│   ├── inspect_data.py        # Inspection script for raw datasets
│   ├── pipeline/
│   │   ├── cleaning.py        # Text normalisation, missing value imputation
│   │   ├── run_cleaning.py    # Runner for the cleaning stage
│   │   ├── blocking.py        # Candidate pair generation (inverted-index & RapidFuzz scoring)
│   │   ├── features.py        # (PENDING) Feature extraction 
│   │   ├── matching.py        # (PENDING) Classifier model & decision logic
│   │   └── aggregation.py     # (PENDING) Connected components & submission formatting
│   └── utils/
│       ├── config.py          # Base paths and configurations
│       └── io.py              # Data ingestion wrappers
├── tests/                     # Unit tests
├── .gitignore
├── methodology.md             # Living document of architectural decisions and gaps
├── README.md
└── requirements.txt
```

## Setup & Installation

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> **Note on Encoding (Windows Users):**
> When reading the raw TSV files, you **must** explicitly pass `encoding="utf-8"` (e.g. to `pd.read_csv`). Windows defaults to `charmap`/`cp1252` encoding, which will crash or severely corrupt Devanagari characters and other Unicode text present in the dataset. This has already been patched in our data loaders (`src/utils/io.py`), but keep this in mind if writing custom scripts.

## Data Layout Expected

The pipeline expects raw TSV files to be strictly located in the following directories, with exact filenames:

*   **Training Data**: 
    *   `data/train/train_ground_truth.tsv`
    *   `data/train/train_source1.tsv`
    *   `data/train/train_source2.tsv`
    *   `data/train/train_source3.tsv`
*   **Test Data**:
    *   `data/test/test_source1.tsv`
    *   `data/test/test_source2.tsv`
    *   `data/test/test_source3.tsv`

## How to Run the Pipeline

### 1. Data Cleaning
The data cleaning module normalizes text, standardizes missing values, and exports compressed `.parquet` files for downstream memory efficiency.

```bash
python src/pipeline/run_cleaning.py
```
*(Optionally pass `--input-dir` and `--output-dir` to point to different dataset samples).*

### 2. Blocking (Candidate Generation)
The blocking module builds a memory-efficient disk-sharded inverted index to generate candidate pairs between `Source 1` and the noisy sources (`Source 2`/`Source 3`). 

```bash
python src/pipeline/blocking.py --chunksize 200000 --max-pairs 100000 --shards 32
```
*   `--chunksize`: Controls the PyArrow Parquet streaming batch size. Keep at `200000` to prevent memory blowouts.
*   `--max-pairs`: The ceiling threshold for combinatorial explosion. Oversized blocks exceeding this will be strictly truncated using a `RapidFuzz` Top-K token similarity score to preserve recall.
*   `--shards`: The number of intermediate disk shards to map blocking keys into.

### 3. Feature Engineering & Matching
*PENDING - Pipeline execution details will be updated once Phase 4 and 5 are implemented.*
