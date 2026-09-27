"""Model 2 (Bi-Encoder) Contrastive Fine-Tuning Module for Business Entity Resolution.

Architecture:
- Backbone: sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0 license).
  * Note: Pretrained predominantly on English data; subword tokenization allows
    multilingual encoding, but dedicated French blocking is complemented via
    TF-IDF/BM25 in Phase 4.
- Loss: MultipleNegativesRankingLoss with in-batch negatives and periodic
  hard negative buffer refreshes.
- Validation: Metric tracking via Recall@k (k=5, 10, 20) against candidate pools.
- Hardware Optimization: FP16 mixed precision and gradient accumulation for
  VRAM-constrained laptop GPUs (RTX 3050 / GTX 1650 class).
"""

import argparse
import contextlib
import importlib
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import InputExample, SentenceTransformer

# Sentence Transformers losses compatibility import across v2 and v3.
try:
    st_losses = importlib.import_module("sentence_transformers.sentence_transformer.losses")
except ImportError:
    st_losses = importlib.import_module("sentence_transformers.losses")

from src.pipeline.pair_construction import mine_hard_negatives
from src.utils.config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset & Ingestion Helpers
# ---------------------------------------------------------------------------


class EntityPairDataset(Dataset):
    """PyTorch Dataset yielding InputExample objects for SentenceTransformer training."""

    def __init__(self, examples: List[InputExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self.examples[idx]


def make_collate_fn(model: SentenceTransformer) -> Callable:
    """Create a robust collate function that tokenizes InputExample batches into PyTorch tensor features."""
    def collate_fn(batch: List[InputExample]) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor]:
        if not batch:
            return [], torch.empty(0, dtype=torch.long)
        num_slots = len(batch[0].texts or [])
        slot_texts = [[(ex.texts or [])[i] for ex in batch] for i in range(num_slots)]

        features: List[Dict[str, Any]] = []
        for texts in slot_texts:
            if hasattr(model, "tokenize"):
                tokenized = model.tokenize(texts)
            elif hasattr(model, "preprocess"):
                tokenized = model.preprocess(texts)
            elif hasattr(model, "tokenizer"):
                tokenized = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            else:
                tokenizer = getattr(model[0], "tokenizer", None)
                if not callable(tokenizer):
                    raise TypeError("The model tokenizer is not callable")
                tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            features.append(cast(Dict[str, Any], tokenized))

        labels = torch.zeros(len(batch), dtype=torch.long)
        return features, labels

    return collate_fn


def load_pair_dataset(
    train_parquet: Union[str, Path, pd.DataFrame],
    val_parquet: Union[str, Path, pd.DataFrame],
) -> Tuple[List[InputExample], List[InputExample]]:
    """Load positive pair Parquet files into sentence-transformers InputExample pairs.

    Parameters
    ----------
    train_parquet : Union[str, Path, pd.DataFrame]
        Path to train_positives.parquet or loaded DataFrame.
    val_parquet : Union[str, Path, pd.DataFrame]
        Path to val_positives.parquet or loaded DataFrame.

    Returns
    -------
    Tuple[List[InputExample], List[InputExample]]
        (train_examples, val_examples) containing InputExample(texts=[s1_text, match_text]).
    """
    train_df = pd.read_parquet(train_parquet) if isinstance(train_parquet, (str, Path)) else train_parquet.copy()
    val_df = pd.read_parquet(val_parquet) if isinstance(val_parquet, (str, Path)) else val_parquet.copy()

    train_examples: List[InputExample] = []
    for idx, row in train_df.iterrows():
        s1_t = str(row.get("s1_text", "")).strip()
        m_t = str(row.get("match_text", "")).strip()
        if s1_t and m_t:
            train_examples.append(InputExample(guid=str(idx), texts=[s1_t, m_t]))

    val_examples: List[InputExample] = []
    for idx, row in val_df.iterrows():
        s1_t = str(row.get("s1_text", "")).strip()
        m_t = str(row.get("match_text", "")).strip()
        if s1_t and m_t:
            val_examples.append(InputExample(guid=str(idx), texts=[s1_t, m_t]))

    logger.info(f"Loaded pair datasets: {len(train_examples):,} train pairs, {len(val_examples):,} val pairs.")
    return train_examples, val_examples


# ---------------------------------------------------------------------------
# Metric Evaluation: Recall@k
# ---------------------------------------------------------------------------


def compute_recall_at_k(
    model: SentenceTransformer,
    val_df: pd.DataFrame,
    candidate_pool_df: Optional[pd.DataFrame] = None,
    k_values: List[int] = [5, 10, 20],
    batch_size: int = 128,
) -> Dict[int, float]:
    """Pure function evaluating Recall@k across validation anchors against a candidate pool.

    Parameters
    ----------
    model : SentenceTransformer
        Bi-encoder model to generate embeddings.
    val_df : pd.DataFrame
        Validation pairs DataFrame with columns [s1_entity_id, match_entity_id, s1_text, match_text].
    candidate_pool_df : Optional[pd.DataFrame], default None
        Candidate pool DataFrame with [match_entity_id, match_text]. If None, built from val_df matches.
    k_values : List[int], default [5, 10, 20]
        Cutoff ranks for Recall@k evaluation.
    batch_size : int, default 128
        Batch size for embedding inference.

    Returns
    -------
    Dict[int, float]
        Dictionary mapping k -> Recall@k score in [0.0, 1.0].
    """
    if val_df.empty:
        return {k: 0.0 for k in k_values}

    # 1. Build unique anchors and mapping of ground-truth matches per anchor
    anchor_df = val_df[["s1_entity_id", "s1_text"]].drop_duplicates(subset=["s1_entity_id"]).reset_index(drop=True)
    gt_map: Dict[str, Set[str]] = cast(
        Dict[str, Set[str]],
        val_df.groupby("s1_entity_id")["match_entity_id"]
        .apply(lambda s: set(s.dropna().unique()))
        .to_dict(),
    )

    # 2. Build candidate pool
    if candidate_pool_df is None or candidate_pool_df.empty:
        cand_pool = (
            val_df[["match_entity_id", "match_text"]]
            .drop_duplicates(subset=["match_entity_id"])
            .reset_index(drop=True)
        )
    else:
        cand_pool = candidate_pool_df.drop_duplicates(subset=["match_entity_id"]).reset_index(drop=True)

    anchor_texts = anchor_df["s1_text"].tolist()
    cand_texts = cand_pool["match_text"].tolist()
    cand_ids = cand_pool["match_entity_id"].tolist()

    # 3. Compute normalized L2 embeddings
    anchor_embeddings = model.encode(
        anchor_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    cand_embeddings = model.encode(
        cand_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # 4. Compute cosine similarity matrix: [num_anchors, num_candidates]
    sim_matrix = np.dot(anchor_embeddings, cand_embeddings.T)

    max_k = max(k_values)
    cand_ids_arr = np.array(cand_ids)

    recall_totals = {k: 0.0 for k in k_values}
    num_anchors = len(anchor_df)

    for i, anchor_row in anchor_df.iterrows():
        anchor_id = anchor_row["s1_entity_id"]
        true_matches = gt_map.get(anchor_id, set())
        if not true_matches:
            continue

        sims = sim_matrix[i]
        top_k_indices = np.argpartition(-sims, min(max_k, len(sims) - 1))[:max_k]
        top_k_sorted = top_k_indices[np.argsort(-sims[top_k_indices])]
        top_k_cand_ids = cand_ids_arr[top_k_sorted]

        for k in k_values:
            retrieved_k = set(top_k_cand_ids[:k])
            hits = len(retrieved_k.intersection(true_matches))
            recall_totals[k] += hits / len(true_matches)

    recalls = {k: (recall_totals[k] / max(num_anchors, 1)) for k in k_values}
    return recalls


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


def train_biencoder(
    train_df: Union[str, Path, pd.DataFrame],
    val_df: Union[str, Path, pd.DataFrame],
    base_checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: Union[str, Path] = OUTPUTS_DIR / "models" / "biencoder_all_minilm_l6_v2",
    epochs: int = 2,
    batch_size: int = 128,
    lr: float = 3e-5,
    hard_negative_refresh_steps: int = 1500,
    hard_negatives_per_positive: int = 6,
    fp16: bool = True,
    grad_accum_steps: int = 1,
    candidate_pool_df: Optional[pd.DataFrame] = None,
) -> str:
    """Fine-tune sentence-transformer bi-encoder with MultipleNegativesRankingLoss and hard negative refreshes.

    Parameters
    ----------
    train_df : Union[str, Path, pd.DataFrame]
        Train positives dataset.
    val_df : Union[str, Path, pd.DataFrame]
        Validation positives dataset.
    base_checkpoint : str, default "sentence-transformers/all-MiniLM-L6-v2"
        Base pretrained huggingface model checkpoint.
    output_dir : Union[str, Path]
        Directory path where the fine-tuned model weights and tokenizer will be saved.
    epochs : int, default 2
        Number of training epochs.
    batch_size : int, default 128
        Batch size per training step.
    lr : float, default 3e-5
        Learning rate with AdamW.
    hard_negative_refresh_steps : int, default 1500
        Number of optimizer steps between candidate pool re-embedding and hard negative mining.
    hard_negatives_per_positive : int, default 6
        Number of top hard negatives mined per anchor.
    fp16 : bool, default True
        Whether to enable PyTorch mixed precision (AMP) for VRAM efficiency.
    grad_accum_steps : int, default 1
        Gradient accumulation steps to simulate larger batch sizes on constrained GPUs.
    candidate_pool_df : Optional[pd.DataFrame], default None
        Pool of candidate entities for hard negative mining and validation evaluation.

    Returns
    -------
    str
        Absolute path to the saved fine-tuned model directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tr_df = pd.read_parquet(train_df) if isinstance(train_df, (str, Path)) else train_df.copy()
    vl_df = pd.read_parquet(val_df) if isinstance(val_df, (str, Path)) else val_df.copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = (device == "cuda")
    use_amp = (fp16 and is_cuda)

    logger.info(f"Initializing base bi-encoder checkpoint '{base_checkpoint}' on device: {device}...")
    model = SentenceTransformer(base_checkpoint, device=device)

    # Initial Baseline Recall Evaluation
    logger.info("Computing initial zero-shot validation Recall@k...")
    init_recalls = compute_recall_at_k(model, vl_df, candidate_pool_df=candidate_pool_df)
    for k, score in init_recalls.items():
        logger.info(f"  [Baseline] Recall@{k:2d}: {score * 100:.2f}%")

    # Build Candidate Pool for Hard Negative Mining
    if candidate_pool_df is None or candidate_pool_df.empty:
        cand_pool = (
            tr_df[["match_entity_id", "match_text"]]
            .drop_duplicates(subset=["match_entity_id"])
            .reset_index(drop=True)
        )
    else:
        cand_pool = candidate_pool_df.drop_duplicates(subset=["match_entity_id"]).reset_index(drop=True)

    train_gt_map: Dict[str, Set[str]] = {
        str(entity_id): {str(match_id) for match_id in match_ids}
        for entity_id, match_ids in tr_df.groupby("s1_entity_id")["match_entity_id"]
        .apply(lambda s: set(s.dropna().unique()))
        .items()
    }

    # Prepare PyTorch Dataloader with custom collate function
    train_examples, _ = load_pair_dataset(tr_df, vl_df)
    collate_fn = make_collate_fn(model)
    train_dataloader = DataLoader(
        EntityPairDataset(train_examples),
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    # Loss function: MultipleNegativesRankingLoss supports in-batch negatives + triplets
    train_loss = st_losses.MultipleNegativesRankingLoss(model=model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_dataloader) // max(grad_accum_steps, 1)) * epochs

    # Use PyTorch AMP scaler for fp16 mixed precision
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def get_autocast_context():
        if use_amp:
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda")
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    logger.info(
        f"Starting Training: Epochs={epochs} | BatchSize={batch_size} | LR={lr} | "
        f"FP16={use_amp} | GradAccum={grad_accum_steps} | TotalSteps={total_steps}"
    )

    global_step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for step, batch in enumerate(train_dataloader):
            features, labels = batch
            # Move only tensor entries in features to target device
            features = [
                {k: v.to(device) for k, v in t.items() if hasattr(v, "to")}
                for t in features
            ]
            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.to(device)

            with get_autocast_context():
                loss_val = train_loss(features, labels)
                if grad_accum_steps > 1:
                    loss_val = loss_val / grad_accum_steps

            if use_amp:
                scaler.scale(loss_val).backward()
            else:
                loss_val.backward()

            epoch_loss += loss_val.item() * (grad_accum_steps if grad_accum_steps > 1 else 1.0)
            batch_count += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()
                global_step += 1

                # Periodic Hard Negative Buffer Refresh
                if global_step > 0 and global_step % hard_negative_refresh_steps == 0:
                    logger.info(f"Step {global_step}: Re-embedding candidate pool for hard negative refresh...")
                    model.eval()
                    cand_embeddings = model.encode(
                        cand_pool["match_text"].tolist(),
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    # Sample anchors to mine hard negatives
                    sample_anchors = tr_df.drop_duplicates(subset=["s1_entity_id"]).head(100)
                    mined_count = 0
                    sample_negatives = []

                    for _, anchor_row in sample_anchors.iterrows():
                        a_id = anchor_row["s1_entity_id"]
                        a_txt = anchor_row["s1_text"]
                        a_emb = model.encode(a_txt, show_progress_bar=False, normalize_embeddings=True)
                        known_pos = train_gt_map.get(a_id, set())

                        negs = mine_hard_negatives(
                            anchor_entity_id=a_id,
                            anchor_text=a_txt,
                            candidate_pool_df=cand_pool,
                            current_embeddings=cand_embeddings,
                            anchor_embedding=a_emb,
                            known_positive_ids=known_pos,
                            k=hard_negatives_per_positive,
                        )
                        mined_count += len(negs)
                        if len(sample_negatives) < 3 and negs:
                            sample_negatives.append(negs[0])

                    logger.info(f"Mined {mined_count:,} hard negatives across sample anchors.")
                    if sample_negatives:
                        logger.info("Sample Mined Hard Negatives (Sanity Inspection):")
                        for s_neg in sample_negatives:
                            logger.info(
                                f"  - Anchor: '{s_neg['anchor_text'][:40]}...' -> "
                                f"Neg: '{s_neg['negative_text'][:40]}...' (Score: {s_neg['similarity_score']:.4f})"
                            )
                    model.train()

        avg_loss = epoch_loss / max(batch_count, 1)
        # End of Epoch Recall@k Evaluation
        model.eval()
        val_recalls = compute_recall_at_k(model, vl_df, candidate_pool_df=candidate_pool_df)
        recall_str = " | ".join([f"Recall@{k}: {score * 100:.2f}%" for k, score in val_recalls.items()])
        logger.info(f"Epoch {epoch}/{epochs} Complete | Avg Loss: {avg_loss:.4f} | Validation: {recall_str}")

    # Save final model
    logger.info(f"Saving fine-tuned bi-encoder model to: '{output_path.resolve()}'...")
    model.save(str(output_path))
    total_elapsed = time.time() - start_time
    logger.info(f"Bi-encoder fine-tuning successfully finished in {total_elapsed:.1f}s.")

    return str(output_path.resolve())


# ---------------------------------------------------------------------------
# Model Comparison Table
# ---------------------------------------------------------------------------


def run_comparison(
    base_checkpoint: str,
    fine_tuned_path: str,
    val_df: pd.DataFrame,
    candidate_pool_df: Optional[pd.DataFrame] = None,
    k_values: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """Produce side-by-side Recall@k comparison table between baseline and fine-tuned models.

    Parameters
    ----------
    base_checkpoint : str
        Base pretrained model checkpoint name or path.
    fine_tuned_path : str
        Directory path of the fine-tuned model checkpoint.
    val_df : pd.DataFrame
        Validation pairs DataFrame.
    candidate_pool_df : Optional[pd.DataFrame], default None
        Candidate pool DataFrame.
    k_values : List[int], default [5, 10, 20]
        Recall@k cutoff ranks.

    Returns
    -------
    pd.DataFrame
        Comparison summary table with columns [k, base_recall, fine_tuned_recall, delta, relative_gain_pct].
    """
    logger.info(f"Loading base checkpoint: {base_checkpoint}...")
    base_model = SentenceTransformer(base_checkpoint)
    base_recalls = compute_recall_at_k(base_model, val_df, candidate_pool_df, k_values=k_values)

    logger.info(f"Loading fine-tuned model: {fine_tuned_path}...")
    ft_model = SentenceTransformer(fine_tuned_path)
    ft_recalls = compute_recall_at_k(ft_model, val_df, candidate_pool_df, k_values=k_values)

    records = []
    for k in k_values:
        b_score = base_recalls.get(k, 0.0)
        f_score = ft_recalls.get(k, 0.0)
        delta = f_score - b_score
        rel_gain = (delta / b_score * 100) if b_score > 0 else 0.0

        records.append({
            "k": k,
            "base_checkpoint_recall": round(b_score, 4),
            "fine_tuned_recall": round(f_score, 4),
            "delta": round(delta, 4),
            "relative_gain_pct": f"{rel_gain:+.2f}%",
        })

    comparison_df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("BI-ENCODER VALIDATION RECALL COMPARISON (Model 2 Benchmark)")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80 + "\n")
    return comparison_df


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark Model 2 sentence-transformer bi-encoder.")
    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=OUTPUTS_DIR / "sample_pairs" / "train_positives.parquet",
        help="Path to training positive pairs Parquet file.",
    )
    parser.add_argument(
        "--val-parquet",
        type=Path,
        default=OUTPUTS_DIR / "sample_pairs" / "val_positives.parquet",
        help="Path to validation positive pairs Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "models" / "biencoder_all_minilm_l6_v2",
        help="Destination directory for fine-tuned model weights.",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Pretrained sentence-transformers backbone checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per optimizer step.")
    parser.add_argument("--lr", type=float, default=3e-5, help="AdamW learning rate.")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable mixed precision AMP training.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable mixed precision.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument(
        "--hard-negative-refresh-steps",
        type=int,
        default=1500,
        help="Steps interval between hard negative refreshes.",
    )
    parser.add_argument(
        "--hard-negatives-per-positive",
        type=int,
        default=6,
        help="Number of hard negatives mined per anchor.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip training and run Recall@k comparison between base checkpoint and fine-tuned model.",
    )
    parser.add_argument(
        "--fine-tuned-model",
        type=str,
        default=None,
        help="Path to fine-tuned model (required when --compare-only is used).",
    )
    args = parser.parse_args()

    val_df = pd.read_parquet(args.val_parquet)

    if args.compare_only:
        ft_path = args.fine_tuned_model or str(args.output_dir)
        run_comparison(
            base_checkpoint=args.base_checkpoint,
            fine_tuned_path=ft_path,
            val_df=val_df,
        )
        return

    # Run Fine-Tuning
    saved_model_path = train_biencoder(
        train_df=args.train_parquet,
        val_df=args.val_parquet,
        base_checkpoint=args.base_checkpoint,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hard_negative_refresh_steps=args.hard_negative_refresh_steps,
        hard_negatives_per_positive=args.hard_negatives_per_positive,
        fp16=args.fp16,
        grad_accum_steps=args.grad_accum_steps,
    )

    # Run Comparison Benchmark Post-Training
    run_comparison(
        base_checkpoint=args.base_checkpoint,
        fine_tuned_path=saved_model_path,
        val_df=val_df,
    )


if __name__ == "__main__":
    main()
