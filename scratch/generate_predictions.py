import lightgbm as lgb
import pandas as pd
from src.pipeline.matching import predict_matches

print("Loading test features...")
features_df = pd.read_parquet("outputs/final_submission_downstream/features/test_features.parquet")
print(f"Loaded {len(features_df)} features.")

print("Loading model...")
model = lgb.Booster(model_file="outputs/final_submission_downstream/matching/matcher_model.txt")

print("Predicting matches...")
preds = predict_matches(model, features_df)

print("Writing predictions...")
preds.to_parquet("outputs/final_submission_downstream/matching/test_predictions.parquet")
print("Done!")
