# Business Entity Resolution Pipeline

A modular machine learning pipeline for business entity resolution across disparate data sources.

## Project Structure

```text
├── data/                  # Raw TSV dataset files (source_1.tsv, source_2.tsv, source_3.tsv, ground_truth.tsv)
├── notebooks/             # Exploratory notebooks and prototyping (not part of production pipeline)
├── outputs/               # Submission files, evaluation reports, and match artifacts
├── src/
│   ├── inspect_data.py    # Scratch inspection script for raw datasets & ground truth
│   ├── pipeline/          # Pipeline stage modules (stubs ready for implementation)
│   │   ├── cleaning.py    # Text normalisation, missing value imputation, entity standardization
│   │   ├── blocking.py    # Candidate pair generation (n-gram, token index, country blocking)
│   │   ├── features.py    # Feature extraction (Levenshtein, Jaro-Winkler, cosine embeddings)
│   │   ├── matching.py    # Classifier model & decision logic (LightGBM, thresholds)
│   │   └── aggregation.py # Connected components, transitive closure & submission formatting
│   └── utils/             # Shared helpers
│       ├── config.py      # Base paths, schemas, and pipeline configurations
│       └── io.py          # Data ingestion with strict string dtype & column validation
├── .gitignore
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

## Dataset Specifications

Place raw TSV files into `/data`.

Each source TSV is loaded with `sep="\t"` and `dtype=str` for all columns to prevent automatic type coercion on IDs.

### Expected Schema
Source files are expected to contain the following columns:
- `entity_id`: Unique identifier for the record
- `business_name`: Name of the business entity
- `business_address`: Full or partial address string
- `country`: Country identifier/code

## Quick Start / Data Inspection

To inspect raw files in the `data/` directory:

```bash
python src/inspect_data.py
```

Options:
- `--data-dir PATH`: Directory containing TSVs (default: `data/`)
- `--source-files PATH [PATH ...]`: Specify exact source files to inspect
- `--ground-truth PATH`: Specify exact ground truth file path

The inspection script outputs:
- Total row and column counts
- Column data types
- Missing/empty value counts for `business_name` and `business_address`
- First 5 rows preview
