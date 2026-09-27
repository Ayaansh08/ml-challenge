import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from src.pipeline.features import compute_features_batch, compute_pair_features

cleaned_dir = Path("outputs/final_submission_downstream/cleaned/test")
pairs_path = "outputs/final_submission_downstream/blocked/test/candidate_pairs.parquet"

cols = ["cleaned_entity_id", "cleaned_name", "cleaned_address", "cleaned_country"]
s1_files = list(cleaned_dir.glob("*source1*_cleaned.parquet"))
s2_files = list(cleaned_dir.glob("*source2*_cleaned.parquet"))
s3_files = list(cleaned_dir.glob("*source3*_cleaned.parquet"))

s1_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s1_files], ignore_index=True)
s2_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s2_files], ignore_index=True)
s3_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s3_files], ignore_index=True)

combined_lookup = pd.concat([s1_df, s2_df, s3_df], ignore_index=True).set_index("cleaned_entity_id")
print("Combined lookup size:", len(combined_lookup))

pf = pq.ParquetFile(pairs_path)
pairs_chunk = next(pf.iter_batches(batch_size=5)).to_pandas()
print("Pairs to compute:")
print(pairs_chunk)

print("\nDirect lookup in combined_lookup:")
id1 = pairs_chunk.iloc[0]["entity_id_1"]
id2 = pairs_chunk.iloc[0]["entity_id_2"]
print(f"Is {id1} in lookup?", id1 in combined_lookup.index)
print(f"Is {id2} in lookup?", id2 in combined_lookup.index)

print(combined_lookup.loc[id1])
print(combined_lookup.loc[id2])

feats = compute_features_batch(pairs_chunk, combined_lookup, None)
print(f"\nReturned shape: {feats.shape}")
print(feats)
