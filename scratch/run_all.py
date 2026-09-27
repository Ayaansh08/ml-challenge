import subprocess
import os

print("Waiting for predictions and candidates to finish...")
# They are running in background tasks, but we can just wait for the output files.
import time
while not os.path.exists("outputs/final_submission_downstream/matching/test_predictions.parquet"):
    time.sleep(1)

while not os.path.exists("outputs/final_submission_downstream/submission/candidate_pairs.tsv"):
    time.sleep(1)

print("Running format_submission.py...")
subprocess.run(["python", "scratch/format_submission.py"], check=True)

print("Running validate.py...")
res = subprocess.run(["python", "scratch/validate.py"], capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
    exit(1)

print("ALL DONE!")
