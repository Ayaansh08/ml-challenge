"""Configuration constants and path definitions."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Schema Expectations
EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Default Source File Names (if placed directly in /data)
DEFAULT_SOURCE_FILES = {
    "source_1": DATA_DIR / "source_1.tsv",
    "source_2": DATA_DIR / "source_2.tsv",
    "source_3": DATA_DIR / "source_3.tsv",
    "ground_truth": DATA_DIR / "ground_truth.tsv",
}
