import sys
import psutil
import pandas as pd
import numpy as np

from src.pipeline.features import compute_features_batch

def mem():
    return psutil.Process().memory_info().rss / (1024*1024)

print(f"[MEM] RSS before lookup initialization: {mem():.2f} MB")

print("Generating FULL mock datasets (S1: 2.2M, S2: 5M, S3: 5.3M rows)...")

s1_ids = [f"S1-{i}" for i in range(2_200_000)]
s1_df = pd.DataFrame({
    "cleaned_entity_id": s1_ids,
    "cleaned_name": ["acme corporation limited"] * 2_200_000,
    "cleaned_address": ["123 main street, new york, ny 10001"] * 2_200_000,
    "cleaned_country": ["US"] * 2_200_000
})

s2_ids = [f"S2-{i}" for i in range(5_000_000)]
s2_df = pd.DataFrame({
    "cleaned_entity_id": s2_ids,
    "cleaned_name": ["acme corp"] * 5_000_000,
    "cleaned_address": ["123 main st, ny"] * 5_000_000,
    "cleaned_country": ["US"] * 5_000_000
})

s3_ids = [f"S3-{i}" for i in range(5_300_000)]
s3_df = pd.DataFrame({
    "cleaned_entity_id": s3_ids,
    "cleaned_name": ["acme corp llc"] * 5_300_000,
    "cleaned_address": ["main st ny"] * 5_300_000,
    "cleaned_country": ["US"] * 5_300_000
})

print(f"[MEM] RSS before concat: {mem():.2f} MB")

combined_lookup = pd.concat([s1_df, s2_df, s3_df], ignore_index=True).set_index("cleaned_entity_id")

print(f"[MEM] RSS after lookup initialization: {mem():.2f} MB")

# Free individual dfs
del s1_df, s2_df, s3_df
import gc
gc.collect()

print("Generating 100k candidate pairs...")
# Just randomly sample IDs
pairs_df = pd.DataFrame({
    "entity_id_1": [f"S1-{i}" for i in range(100_000)],
    "entity_id_2": [f"S3-{i}" for i in range(100_000)]
})

print(f"[MEM] RSS before compute_features_batch: {mem():.2f} MB")

features_df = compute_features_batch(pairs_df, combined_lookup, None)

print(f"[MEM] RSS after compute_features_batch: {mem():.2f} MB")
print(f"[MEM] Peak RSS (approx): {mem():.2f} MB")

sys.exit(0)
