import duckdb
import time

def generate_cross_source_pairs_fast(s1_records, s2_records, max_pairs, stats, blocking_key=""):
    if not s1_records or not s2_records:
        return []
    total_possible = len(s1_records) * len(s2_records)
    if total_possible <= max_pairs:
        return [(r1[0], r2[0]) for r1 in s1_records for r2 in s2_records]
    else:
        stats["oversized_blocks"] += 1
        if len(s1_records) > max_pairs:
            s1_records = s1_records[:max_pairs]
        k_per_s1 = max_pairs // len(s1_records)
        remainder = max_pairs % len(s1_records)
        pairs = []
        emitted = 0
        n_s2 = len(s2_records)
        for i, r1 in enumerate(s1_records):
            add_count = k_per_s1 + (1 if i < remainder else 0)
            for j in range(add_count):
                if emitted >= max_pairs:
                    break
                idx = (j * n_s2) // add_count if add_count > 0 else 0
                idx = min(idx, n_s2 - 1)
                pairs.append((r1[0], s2_records[idx][0]))
                emitted += 1
        return pairs

con = duckdb.connect()
t0 = time.time()
cur = con.execute("""
SELECT blocking_key, list(entity_id), list(source), list(name) 
FROM read_parquet('outputs/pipeline_50k/blocked/train/shards/shard_0/*.parquet') 
GROUP BY blocking_key
""")
rows = cur.fetchall()

stats = {"s1s2_pairs": 0, "s1s3_pairs": 0, "blocks_processed": len(rows), "oversized_blocks": 0}
all_pairs = []
t1 = time.time()
for key, eids, srcs, names in rows:
    if len(eids) < 2:
        continue
    # Prefix separation (e.g. 'train' vs 'test')
    # Sources have format 'train_source1', 'test_source1', etc.
    # Group by prefix
    prefix_map = {}
    for eid, src, name in zip(eids, srcs, names):
        pfx = src.split("_")[0]
        if pfx not in prefix_map:
            prefix_map[pfx] = {"s1": [], "s2": [], "s3": []}
        if "source1" in src or "source_1" in src:
            prefix_map[pfx]["s1"].append((eid, name))
        elif "source2" in src or "source_2" in src:
            prefix_map[pfx]["s2"].append((eid, name))
        elif "source3" in src or "source_3" in src:
            prefix_map[pfx]["s3"].append((eid, name))
            
    for pfx, grp in prefix_map.items():
        s1 = grp["s1"]
        if not s1:
            continue
        s2 = grp["s2"]
        s3 = grp["s3"]
        if s2:
            p = generate_cross_source_pairs_fast(s1, s2, 500, stats, key)
            stats["s1s2_pairs"] += len(p)
            all_pairs.extend(p)
        if s3:
            p = generate_cross_source_pairs_fast(s1, s3, 500, stats, key)
            stats["s1s3_pairs"] += len(p)
            all_pairs.extend(p)

t2 = time.time()
print(f"Generated {len(all_pairs):,} pairs in {t2-t1:.2f}s!")
print(f"Stats: {stats}")
