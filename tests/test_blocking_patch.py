import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.pipeline.blocking import get_blocking_keys, generate_cross_source_pairs, process_shard

def test_no_ctry_key():
    keys = get_blocking_keys("acme corp", "france", "123 rue")
    for k in keys:
        assert not k.startswith("ctry_")

def test_generate_cross_source_pairs_oversized():
    # 200 S1 x 200 S2 = 40,000 potential pairs > max_pairs (50)
    s1_data = [{"entity_id": f"s1_{i}", "name": f"name_{i}"} for i in range(200)]
    s2_data = [{"entity_id": f"s2_{i}", "name": f"name_{i}"} for i in range(200)]
    
    s1_df = pd.DataFrame(s1_data)
    s2_df = pd.DataFrame(s2_data)
    
    stats = {"oversized_blocks": 0}
    max_pairs = 50
    
    pairs = generate_cross_source_pairs(s1_df, s2_df, max_pairs, stats, "test_key")
    
    assert len(pairs) == 50
    assert stats["oversized_blocks"] == 1

def test_s2_s3_zeros():
    # Verify process_shard logic skips S2xS3
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy shard parquet file with S2 and S3 only
        df = pd.DataFrame({
            "blocking_key": ["test_key"] * 4,
            "entity_id": ["s2_1", "s2_2", "s3_1", "s3_2"],
            "source": ["train_source2", "train_source2", "train_source3", "train_source3"],
            "name": ["n1", "n2", "n3", "n4"]
        })
        
        tmpdir_path = Path(tmpdir)
        shard_dir = tmpdir_path / "shard_0"
        shard_dir.mkdir()
        
        df.to_parquet(shard_dir / "test.parquet")
        
        out_dir = tmpdir_path / "out"
        out_dir.mkdir()
        
        stats = {
            "s1s2_pairs": 0,
            "s1s3_pairs": 0,
            "s2s3_pairs": 0,
            "blocks_processed": 0,
            "oversized_blocks": 0
        }
        
        process_shard(shard_dir, out_dir, 100, stats)
        
        assert stats["s1s2_pairs"] == 0
        assert stats["s1s3_pairs"] == 0
        assert stats["s2s3_pairs"] == 0
