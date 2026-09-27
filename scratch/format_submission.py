import pandas as pd
from pathlib import Path

print("Reading test_source1.tsv...")
s1_df = pd.read_csv("data/test/test_source1.tsv", sep="\t", usecols=["entity_id"])
s1_ids = s1_df["entity_id"].unique()
print(f"Loaded {len(s1_ids)} S1 IDs")

print("Reading test_predictions.parquet...")
preds_df = pd.read_parquet("outputs/final_submission_downstream/matching/test_predictions.parquet")
matches = preds_df[preds_df["prediction"] == 1]
print(f"Loaded {len(preds_df)} predictions, {len(matches)} matches")

mask1 = matches["entity_id_1"].str.startswith("S1-")
e1_s1 = matches[mask1]
e1_s1_pairs = e1_s1[["entity_id_1", "entity_id_2"]].rename(columns={"entity_id_1": "S1", "entity_id_2": "cand"})

mask2 = matches["entity_id_2"].str.startswith("S1-")
e2_s1 = matches[mask2]
e2_s1_pairs = e2_s1[["entity_id_2", "entity_id_1"]].rename(columns={"entity_id_2": "S1", "entity_id_1": "cand"})

all_pairs = pd.concat([e1_s1_pairs, e2_s1_pairs])
all_pairs = all_pairs[~all_pairs["cand"].str.startswith("S1-")]

grouped = all_pairs.groupby("S1")["cand"].apply(lambda x: " ".join(sorted(set(x)))).reset_index()

s1_df = pd.DataFrame({"source1_entity_id": s1_ids})
merged = s1_df.merge(grouped, left_on="source1_entity_id", right_on="S1", how="left")
merged["matched_entity_ids"] = merged["cand"].fillna("")
merged = merged[["source1_entity_id", "matched_entity_ids"]]

Path("outputs/final_submission_downstream/submission").mkdir(parents=True, exist_ok=True)
merged.to_csv("outputs/final_submission_downstream/submission/matching_results.tsv", sep="\t", index=False)
print("Done formatting matching_results.tsv!")
