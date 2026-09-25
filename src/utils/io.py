"""Input/output helper functions for loading and saving dataset files."""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from src.utils.config import EXPECTED_COLUMNS


def load_source(
    path: Union[str, Path],
    expected_columns: Optional[List[str]] = None,
    validate_columns: bool = True,
) -> pd.DataFrame:
    """Load a TSV source file with string dtype for all columns and validate schema.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the TSV file.
    expected_columns : Optional[List[str]], default None
        List of required column names. Defaults to EXPECTED_COLUMNS
        ('entity_id', 'business_name', 'business_address', 'country').
    validate_columns : bool, default True
        Whether to validate that expected_columns exist in the loaded DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with all columns loaded as string (object/string dtype).

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If any of the expected columns are missing from the TSV file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Source file not found at: {file_path.resolve()}")

    # Read TSV with dtype=str for all columns to prevent pandas type inference on IDs
    df = pd.read_csv(file_path, sep="\t", dtype=str)

    if validate_columns:
        required = expected_columns if expected_columns is not None else EXPECTED_COLUMNS
        missing_columns = [col for col in required if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing expected column(s) in '{file_path.name}': {missing_columns}. "
                f"Found columns: {list(df.columns)}. Expected columns: {required}."
            )

    return df


def load_ground_truth(path: Union[str, Path]) -> pd.DataFrame:
    """Load a ground truth TSV/CSV file with all columns as string dtype.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the ground truth file.

    Returns
    -------
    pd.DataFrame
        DataFrame with all columns loaded as string.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found at: {file_path.resolve()}")

    sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(file_path, sep=sep, dtype=str)
