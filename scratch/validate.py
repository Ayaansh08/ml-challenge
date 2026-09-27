import pandas as pd

print("Loading test_source1.tsv...")
s1_df = pd.read_csv("data/test/test_source1.tsv", sep="\t", usecols=["entity_id"])
s1_ids = set(s1_df["entity_id"].unique())

for file_name, col_name in [
    ("matching_results.tsv", "matched_entity_ids"),
    ("candidate_pairs.tsv", "candidate_entity_ids")
]:
    print(f"Validating {file_name}...")
    try:
        results = pd.read_csv(f"outputs/final_submission_downstream/submission/{file_name}", sep="\t", keep_default_na=False)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        exit(1)

    if list(results.columns) != ["source1_entity_id", col_name]:
        print(f"FAIL: Invalid columns {list(results.columns)} in {file_name}")
        exit(1)

    result_s1 = set(results["source1_entity_id"].unique())
    if len(results) != len(s1_ids):
        print(f"FAIL: Expected {len(s1_ids)} rows, got {len(results)} in {file_name}")
        exit(1)
    if result_s1 != s1_ids:
        print(f"FAIL: Missing or extra S1 IDs in {file_name}")
        exit(1)

    for i, row in results.iterrows():
        matches = str(row[col_name]).split() if str(row[col_name]).strip() != "" else []
        for m in matches:
            if m.startswith("S1-"):
                print(f"FAIL: S1 ID {m} found in {col_name} list")
                exit(1)
            if not m.startswith("S"):
                print(f"FAIL: Invalid ID {m} in {col_name}")
                exit(1)
        if len(matches) != len(set(matches)):
            print(f"FAIL: Duplicate IDs found in {col_name} for {row['source1_entity_id']}")
            exit(1)
            
print("PASS")
