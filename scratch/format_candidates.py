import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict

print("Reading test_source1.tsv...")
s1_df = pd.read_csv("data/test/test_source1.tsv", sep="\t", usecols=["entity_id"])
s1_ids = list(s1_df["entity_id"].unique())

print("Iterating candidate_pairs.parquet (500k rows)...")
pf = pq.ParquetFile("outputs/final_submission_downstream/blocked/test/candidate_pairs.parquet")
cand_dict = defaultdict(set)

for batch in pf.iter_batches(batch_size=500_000):
    df = batch.to_pandas()
    
    mask1 = df["entity_id_1"].str.startswith("S1-")
    mask2 = df["entity_id_2"].str.startswith("S1-")
    
    valid_e1_s1 = df[mask1 & ~mask2]
    for s1, cands in valid_e1_s1.groupby("entity_id_1")["entity_id_2"].apply(set).items():
        cand_dict[s1].update(cands)
        
    valid_e2_s1 = df[mask2 & ~mask1]
    for s1, cands in valid_e2_s1.groupby("entity_id_2")["entity_id_1"].apply(set).items():
        cand_dict[s1].update(cands)
    break # Only process the first 500,000 candidate pairs (the ones fed to features.py)

print("Writing candidate_pairs.tsv...")
Path("outputs/final_submission_downstream/submission").mkdir(parents=True, exist_ok=True)
with open("outputs/final_submission_downstream/submission/candidate_pairs.tsv", "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1 in s1_ids:
        cand_str = " ".join(sorted(list(cand_dict.get(s1, set()))))
        f.write(f"{s1}\t{cand_str}\n")

print("Done formatting candidate_pairs.tsv!")
