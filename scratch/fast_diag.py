import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from src.pipeline.features import compute_features_batch

cleaned_dir = Path("outputs/final_submission_downstream/cleaned/test")
pairs_path = "outputs/final_submission_downstream/blocked/test/candidate_pairs.parquet"

s1_files = list(cleaned_dir.glob("*source1*_cleaned.parquet"))
s2_files = list(cleaned_dir.glob("*source2*_cleaned.parquet"))
s3_files = list(cleaned_dir.glob("*source3*_cleaned.parquet"))

cols = ["cleaned_entity_id", "cleaned_name", "cleaned_address", "cleaned_country"]
s1_df = pd.read_parquet(s1_files[0], columns=cols).head(5)
s2_df = pd.read_parquet(s2_files[0], columns=cols).head(5)
s3_df = pd.read_parquet(s3_files[0], columns=cols).head(5)

print(f"S1 sample ids: {s1_df['cleaned_entity_id'].tolist()}")
print(f"S2 sample ids: {s2_df['cleaned_entity_id'].tolist()}")
print(f"S3 sample ids: {s3_df['cleaned_entity_id'].tolist()}")

pf = pq.ParquetFile(pairs_path)
pairs_chunk = next(pf.iter_batches(batch_size=5)).to_pandas()
print(f"Pairs chunk:\n{pairs_chunk}")
