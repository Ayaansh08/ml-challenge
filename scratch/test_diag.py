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
s1_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s1_files], ignore_index=True)
s2_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s2_files], ignore_index=True)
s3_df = pd.concat([pd.read_parquet(f, columns=cols) for f in s3_files], ignore_index=True)

print(f"S1 unique ids: {s1_df['cleaned_entity_id'].nunique()}, sample: {s1_df['cleaned_entity_id'].head(3).tolist()}")
print(f"S2 unique ids: {s2_df['cleaned_entity_id'].nunique()}, sample: {s2_df['cleaned_entity_id'].head(3).tolist()}")
print(f"S3 unique ids: {s3_df['cleaned_entity_id'].nunique()}, sample: {s3_df['cleaned_entity_id'].head(3).tolist()}")

combined_lookup = pd.concat([s1_df, s2_df, s3_df], ignore_index=True).set_index("cleaned_entity_id")

pf = pq.ParquetFile(pairs_path)
for batch in pf.iter_batches(batch_size=10000):
    pairs_chunk = batch.to_pandas()
    print(f"\nInput candidate row count: {len(pairs_chunk)}")
    print(f"Sample pairs: {pairs_chunk.head(3).to_dict('records')}")
    
    # number of rows retrieved from S1, S2/S3
    id1s = pairs_chunk["entity_id_1"]
    id2s = pairs_chunk["entity_id_2"]
    
    in_s1_1 = id1s.isin(s1_df["cleaned_entity_id"]).sum()
    in_s2_1 = id1s.isin(s2_df["cleaned_entity_id"]).sum()
    in_s3_1 = id1s.isin(s3_df["cleaned_entity_id"]).sum()
    
    in_s1_2 = id2s.isin(s1_df["cleaned_entity_id"]).sum()
    in_s2_2 = id2s.isin(s2_df["cleaned_entity_id"]).sum()
    in_s3_2 = id2s.isin(s3_df["cleaned_entity_id"]).sum()
    
    print(f"ID1s in S1: {in_s1_1}, S2: {in_s2_1}, S3: {in_s3_1}")
    print(f"ID2s in S1: {in_s1_2}, S2: {in_s2_2}, S3: {in_s3_2}")
    
    missing1 = id1s[~id1s.isin(combined_lookup.index)]
    missing2 = id2s[~id2s.isin(combined_lookup.index)]
    print(f"Missing ID1 lookups: {len(missing1)}")
    print(f"Missing ID2 lookups: {len(missing2)}")
    
    if len(missing1) > 0:
        print(f"Sample missing ID1: {missing1.head(3).tolist()}")
    if len(missing2) > 0:
        print(f"Sample missing ID2: {missing2.head(3).tolist()}")

    feats = compute_features_batch(pairs_chunk, combined_lookup, None)
    print(f"\nReturned shape: {feats.shape}")
    print(f"Returned empty: {feats.empty}")
    print(f"Returned columns: {feats.columns.tolist() if not feats.empty else []}")
    break
