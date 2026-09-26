"""CLI entry point to execute Phase 2 data cleaning and normalization on source TSVs.

Uses memory-bounded chunked streaming via PyArrow ParquetWriter to support
multi-million row datasets (e.g. 5M+ rows in source3) without OOM.

Features:
- Streaming chunked ingestion (pd.read_csv with chunksize)
- Incremental Parquet writing (pyarrow.parquet.ParquetWriter)
- Vectorized column-wise cleaning (avoids row-wise dict allocation)
- Explicit garbage collection (gc.collect()) between chunks and files
- File-level resume capability (skips already-cleaned non-empty files)
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline.cleaning import clean_dataframe
from src.utils.config import DATA_DIR, OUTPUTS_DIR


def process_file_in_chunks(
    file_path: Path,
    output_file: Path,
    chunksize: int = 200_000,
) -> int:
    """Process a large TSV file in memory-bounded chunks and stream to Parquet.

    Parameters
    ----------
    file_path : Path
        Path to the raw source TSV file.
    output_file : Path
        Target destination path for the cleaned Parquet file.
    chunksize : int, default 200_000
        Number of rows to read and transform per chunk.

    Returns
    -------
    int
        Total number of records processed and written.
    """
    start_time = time.time()
    total_rows = 0
    writer: Optional[pq.ParquetWriter] = None

    try:
        # Stream TSV in chunks with string dtype to avoid type inference on IDs
        chunk_iter = pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            chunksize=chunksize,
            low_memory=False,
        )

        for chunk_idx, raw_chunk in enumerate(chunk_iter, start=1):
            chunk_start = time.time()
            rows_in_chunk = len(raw_chunk)

            # Apply column-wise vectorized cleaning on chunk
            cleaned_chunk = clean_dataframe(raw_chunk)

            # Convert to PyArrow Table
            table = pa.Table.from_pandas(cleaned_chunk, preserve_index=False)

            # Initialize streaming ParquetWriter on first chunk's schema
            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_file),
                    table.schema,
                    compression="snappy",
                )

            # Write chunk table incrementally to disk
            writer.write_table(table)

            total_rows += rows_in_chunk
            chunk_elapsed = time.time() - chunk_start
            total_elapsed = time.time() - start_time
            print(
                f"  [{file_path.name}] Chunk {chunk_idx:3d} | "
                f"+{rows_in_chunk:,} rows | Total: {total_rows:,} rows | "
                f"Chunk time: {chunk_elapsed:.2f}s | Elapsed: {total_elapsed:.1f}s"
            )

            # Explicitly release chunk memory
            del raw_chunk, cleaned_chunk, table

    finally:
        # Guarantee writer closure to flush buffers and finalize file footer
        if writer is not None:
            writer.close()
            del writer

    total_time = time.time() - start_time
    file_size_mb = (
        output_file.stat().st_size / (1024 * 1024) if output_file.exists() else 0.0
    )
    print(
        f"[SUCCESS] Wrote {total_rows:,} cleaned records for '{file_path.name}' "
        f"to '{output_file.name}' ({file_size_mb:.2f} MB in {total_time:.1f}s)\n"
    )
    return total_rows


def run_cleaning(
    input_dir: Path,
    output_dir: Path,
    source_files: Optional[List[str]] = None,
    chunksize: int = 200_000,
    force: bool = False,
) -> None:
    """Run memory-efficient chunked data cleaning across all source files.

    Parameters
    ----------
    input_dir : Path
        Directory containing raw TSV files.
    output_dir : Path
        Target directory to write cleaned Parquet files.
    source_files : Optional[List[str]], default None
        Optional explicit list of filenames to clean.
    chunksize : int, default 200_000
        Chunk batch size for reading and writing.
    force : bool, default False
        If True, re-process files even if output Parquet already exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_files:
        files_to_process = [input_dir / f for f in source_files]
    else:
        # Auto-discover source TSVs excluding ground truth and hidden files
        all_tsvs = sorted(input_dir.rglob("*.tsv"))
        files_to_process = [
            f
            for f in all_tsvs
            if not f.name.startswith(".")
            and "ground_truth" not in f.name.lower()
            and "gt" not in f.name.lower()
        ]

    if not files_to_process:
        print(f"[WARN] No source TSV files found in: {input_dir.resolve()}")
        return

    print(f"[INFO] Found {len(files_to_process)} source file(s) to process.")
    print(
        f"[INFO] Chunk size: {chunksize:,} rows per batch | "
        f"Streaming writer: PyArrow ParquetWriter | Force overwrite: {force}"
    )

    for file_path in files_to_process:
        if not file_path.exists():
            print(f"[ERROR] Source file not found: {file_path}", file=sys.stderr)
            continue

        output_file = output_dir / f"{file_path.stem}_cleaned.parquet"

        # Check for file-level resume
        if not force and output_file.exists() and output_file.stat().st_size > 0:
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(
                f"[INFO] Skipping '{file_path.name}' — output '{output_file.name}' already "
                f"exists and is non-empty ({size_mb:.2f} MB). Pass --force to re-process."
            )
            continue

        print(f"\n[INFO] Starting processing for '{file_path.name}' -> '{output_file.name}'...")
        try:
            process_file_in_chunks(
                file_path=file_path,
                output_file=output_file,
                chunksize=chunksize,
            )
        except Exception as e:
            print(f"[ERROR] Failed cleaning '{file_path.name}': {e}", file=sys.stderr)
            # Remove incomplete corrupt Parquet file if processing failed midway
            if output_file.exists():
                try:
                    output_file.unlink()
                    print(f"[INFO] Cleaned up incomplete output file: '{output_file.name}'")
                except Exception:
                    pass
        finally:
            # Force explicit garbage collection between files to release memory
            gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 chunked data cleaning on raw entity resolution TSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing raw source TSVs (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "cleaned",
        help="Directory to save cleaned Parquet datasets (default: outputs/cleaned/)",
    )
    parser.add_argument(
        "--source-files",
        nargs="*",
        default=None,
        help="Optional explicit list of source filenames to clean.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Batch chunk size in number of rows (default: 200,000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if non-empty output Parquet already exists.",
    )
    args = parser.parse_args()

    run_cleaning(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        source_files=args.source_files,
        chunksize=args.chunksize,
        force=args.force,
    )


if __name__ == "__main__":
    main()
