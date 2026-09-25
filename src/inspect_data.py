"""Scratch script to inspect raw datasets and ground truth.

Loads source files + ground truth, displays row counts, column dtypes,
5-row preview, and null/empty string diagnostics for key columns.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as standalone script from project root or src dir
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils.config import DATA_DIR
from src.utils.io import load_ground_truth, load_source


def count_missing_or_empty(series: pd.Series) -> int:
    """Count nulls, NaN, and whitespace-only/empty strings in a Series."""
    if series.empty:
        return 0
    is_null = series.isna()
    is_empty_str = series.fillna("").astype(str).str.strip() == ""
    return int((is_null | is_empty_str).sum())


def inspect_dataframe(name: str, df: pd.DataFrame, is_source: bool = True) -> None:
    """Print formatted inspection details for a DataFrame."""
    print("=" * 80)
    print(f"Dataset: {name}")
    print("=" * 80)
    print(f"Row count: {len(df):,}")
    print(f"Column count: {len(df.columns)}")
    print("\nColumn Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    if is_source:
        print("\nMissing / Empty Value Analysis (business_name, business_address):")
        for col in ["business_name", "business_address"]:
            if col in df.columns:
                missing_cnt = count_missing_or_empty(df[col])
                pct = (missing_cnt / len(df) * 100) if len(df) > 0 else 0.0
                print(f"  - {col}: {missing_cnt:,} / {len(df):,} missing or empty ({pct:.2f}%)")
            else:
                print(f"  - {col}: [Column not found]")

    print("\nSample (First 5 rows):")
    if len(df) > 0:
        print(df.head(5).to_string(index=False))
    else:
        print("  [Dataset is empty]")
    print("\n")


def discover_files(data_dir: Path) -> Dict[str, Optional[Path]]:
    """Automatically detect source files and ground truth in data directory."""
    all_tsvs = sorted(list(data_dir.glob("*.tsv")) + list(data_dir.glob("*.csv")))
    # Exclude gitkeep or hidden files
    all_tsvs = [f for f in all_tsvs if not f.name.startswith(".")]

    sources: Dict[str, Optional[Path]] = {}
    gt_file: Optional[Path] = None

    for f in all_tsvs:
        fname = f.stem.lower()
        if "ground_truth" in fname or "gt" in fname or "truth" in fname:
            gt_file = f
        else:
            sources[f.name] = f

    return {"sources": list(sources.values()), "ground_truth": gt_file}  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw entity resolution datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to the directory containing raw TSV files.",
    )
    parser.add_argument(
        "--source-files",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit paths to source TSV files.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Explicit path to ground truth file.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir

    source_paths: List[Path] = []
    gt_path: Optional[Path] = args.ground_truth

    if args.source_files:
        source_paths = args.source_files
    else:
        discovered = discover_files(data_dir)
        source_paths = discovered.get("sources", [])  # type: ignore
        if gt_path is None:
            gt_path = discovered.get("ground_truth")  # type: ignore

    if not source_paths and gt_path is None:
        print(f"[INFO] No data files found in '{data_dir.resolve()}'.")
        print("Please place your raw TSV source files and ground truth file into the /data folder.")
        print("Expected source schema: entity_id, business_name, business_address, country")
        return

    # Inspect sources
    for sp in source_paths:
        try:
            df_src = load_source(sp)
            inspect_dataframe(name=sp.name, df=df_src, is_source=True)
        except Exception as e:
            print(f"[ERROR] Failed to load source file '{sp}': {e}", file=sys.stderr)

    # Inspect ground truth
    if gt_path is not None and gt_path.exists():
        try:
            df_gt = load_ground_truth(gt_path)
            inspect_dataframe(name=gt_path.name, df=df_gt, is_source=False)
        except Exception as e:
            print(f"[ERROR] Failed to load ground truth file '{gt_path}': {e}", file=sys.stderr)
    elif gt_path is not None:
        print(f"[WARN] Ground truth file not found at: {gt_path}")


if __name__ == "__main__":
    main()
