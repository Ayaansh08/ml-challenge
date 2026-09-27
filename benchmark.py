import time
import pandas as pd
from src.pipeline.features import compute_pair_features

r1 = {"cleaned_name": "acme corporation limited", "cleaned_address": "123 main street, new york, ny 10001", "cleaned_country": "US"}
r2 = {"cleaned_name": "acme corp llc", "cleaned_address": "main st ny", "cleaned_country": "US"}

t0 = time.time()
for _ in range(100_000):
    compute_pair_features(r1, r2)
t1 = time.time()
print(f"100k pairs took {t1-t0:.2f}s")
