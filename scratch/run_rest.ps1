$env:PYTHONPATH="."
python src/pipeline/matching.py --features outputs/final_submission_downstream/features/test_features.parquet --output-dir outputs/final_submission_downstream/matching
python src/pipeline/aggregation.py --predictions outputs/final_submission_downstream/matching/test_predictions.parquet --output-dir outputs/final_submission_downstream/submission
