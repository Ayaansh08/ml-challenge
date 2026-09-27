import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

cleaned_dir = Path("outputs/final_submission_downstream/cleaned/test")
pairs_path = "outputs/final_submission_downstream/blocked/test/candidate_pairs.parquet"

s1_files = list(cleaned_dir.glob("*source1*_cleaned.parquet"))
s1_df = pd.concat([pd.read_parquet(f, columns=["cleaned_entity_id"]) for f in s1_files], ignore_index=True)
s1_set = set(s1_df["cleaned_entity_id"])

pf = pq.ParquetFile(pairs_path)
pairs_chunk = next(pf.iter_batches(batch_size=10000)).to_pandas()
id1s = set(pairs_chunk["entity_id_1"])

intersect = id1s.intersection(s1_set)
print(f"Number of entity_id_1 matching in test_source1: {len(intersect)} out of {len(id1s)}")
