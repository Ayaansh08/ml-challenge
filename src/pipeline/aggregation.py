"""Aggregation module (Phase 6): match graph, transitive closure, and submission.

Takes pairwise match predictions from Phase 5 (`matching.py:predict_matches`),
forms entity clusters via transitive closure (connected components), assigns
deterministic cluster IDs, and writes the final submission TSV.

Functions
--------
- `build_match_graph`: Filter predicted matches and build an undirected graph.
- `find_clusters`: Connected components -> {cluster_id: set(entity_ids)}.
- `form_clusters`: Like `find_clusters` but also assigns singletons for
  entities that never appear in any predicted match.
- `generate_submission`: Write clusters as a sorted TSV (entity_id, cluster_id).
- `run_aggregation`: End-to-end CLI entry point.
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)

REQUIRED_PAIR_COLUMNS = ("entity_id_1", "entity_id_2")


def _select_match_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Boolean mask of pairs considered matches.

    The `prediction` column (output of Phase 5) takes precedence when present;
    otherwise pairs are kept when `score >= threshold`.
    """
    if "prediction" in df.columns:
        pred = pd.to_numeric(df["prediction"], errors="coerce")
        return pred.fillna(0) == 1
    if "score" in df.columns:
        score = pd.to_numeric(df["score"], errors="coerce")
        return score.fillna(float("-inf")) >= threshold
    raise ValueError(
        "Predictions DataFrame must contain a 'prediction' and/or 'score' column."
    )


def build_match_graph(
    predictions_df: pd.DataFrame,
    threshold: float = 0.5,
) -> nx.Graph:
    """Build an undirected graph of predicted entity matches.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of `matching.py:predict_matches` with columns
        [entity_id_1, entity_id_2, score, prediction].
    threshold : float, default 0.5
        Score threshold used only when no `prediction` column is present.

    Returns
    -------
    nx.Graph
        Undirected graph with entity IDs as nodes and predicted matches as
        edges. Edge attribute `score` holds the pair score when available.
    """
    missing = [c for c in REQUIRED_PAIR_COLUMNS if c not in predictions_df.columns]
    if missing:
        raise ValueError(
            f"Predictions DataFrame missing required columns: {missing}. "
            f"Found: {list(predictions_df.columns)}"
        )

    mask = _select_match_mask(predictions_df, threshold)
    matches = predictions_df[mask]

    graph = nx.Graph()
    has_score = "score" in matches.columns

    for row in matches.itertuples(index=False):
        e1 = getattr(row, "entity_id_1", None)
        e2 = getattr(row, "entity_id_2", None)
        if pd.isna(e1) or pd.isna(e2):
            continue
        e1, e2 = str(e1), str(e2)
        if e1 == e2:
            continue  # Self-loop carries no clustering signal.
        score = float(getattr(row, "score", float("nan"))) if has_score else None
        if pd.notna(score):
            graph.add_edge(e1, e2, score=score)
        else:
            graph.add_edge(e1, e2)

    logger.info(
        "Built match graph: %d nodes, %d edges (threshold=%.2f).",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        threshold,
    )
    return graph


def find_clusters(graph: nx.Graph) -> Dict[int, Set[str]]:
    """Find connected components and assign deterministic cluster IDs.

    Parameters
    ----------
    graph : nx.Graph
        Match graph from `build_match_graph`.

    Returns
    -------
    Dict[int, Set[str]]
        Mapping of cluster_id -> set of entity_ids. Components are sorted by
        their lexicographically sorted member tuples and numbered 1..N, so the
        same graph always yields the same IDs.
    """
    groups: List[Set[str]] = [set(component) for component in nx.connected_components(graph)]
    groups.sort(key=lambda members: tuple(sorted(members)))
    return {cluster_id: members for cluster_id, members in enumerate(groups, start=1)}


def form_clusters(
    graph: nx.Graph,
    entity_ids: Optional[Iterable[str]] = None,
) -> Dict[int, Set[str]]:
    """Cluster graph components plus singleton clusters for unmatched entities.

    Parameters
    ----------
    graph : nx.Graph
        Match graph from `build_match_graph`.
    entity_ids : Optional[Iterable[str]]
        Every entity that must appear in the output. Entities with no
        predicted match get their own singleton cluster.

    Returns
    -------
    Dict[int, Set[str]]
        Mapping of cluster_id -> set of entity_ids, deterministic (sorted by
        member tuple, numbered 1..N) across components and singletons alike.
    """
    clusters = find_clusters(graph)
    if entity_ids is None:
        return clusters

    covered: Set[str] = set()
    for members in clusters.values():
        covered |= members

    missing = sorted(set(str(e) for e in entity_ids) - covered)
    if not missing:
        return clusters

    groups: List[Set[str]] = list(clusters.values()) + [{eid} for eid in missing]
    groups.sort(key=lambda members: tuple(sorted(members)))
    return {cluster_id: members for cluster_id, members in enumerate(groups, start=1)}


def generate_submission(clusters: Dict[int, Set[str]], output_path: Union[str, Path]) -> Path:
    """Write clusters to a TSV with columns (entity_id, cluster_id).

    Parameters
    ----------
    clusters : Dict[int, Set[str]]
        Mapping of cluster_id -> set of entity_ids (from `form_clusters`).
    output_path : Union[str, Path]
        Destination TSV path. Parent directories are created as needed.

    Returns
    -------
    Path
        Path to the written submission file.

    Raises
    ------
    ValueError
        If any entity appears in more than one cluster.
    """
    rows = []
    seen: Dict[str, int] = {}
    for cluster_id, members in clusters.items():
        for entity_id in members:
            eid = str(entity_id)
            if eid in seen:
                raise ValueError(
                    f"Entity '{eid}' appears in both cluster {seen[eid]} and "
                    f"cluster {cluster_id}; each entity must appear exactly once."
                )
            seen[eid] = cluster_id
            rows.append((eid, cluster_id))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df = pd.DataFrame(rows, columns=["entity_id", "cluster_id"])
    if not submission_df.empty:
        submission_df = submission_df.sort_values("entity_id", kind="mergesort").reset_index(drop=True)
    submission_df.to_csv(out_path, sep="\t", index=False)

    logger.info("Wrote submission: %s (%d rows).", out_path, len(submission_df))
    return out_path


def load_all_entity_ids(entities_dir: Union[str, Path]) -> List[str]:
    """Load every entity ID from the 3 source files in a directory.

    Test sources are preferred (final submission); otherwise all cleaned
    parquet/source TSV files in the directory are used. Deduplicated and
    sorted for deterministic output.

    Parameters
    ----------
    entities_dir : Union[str, Path]
        Directory holding the source files (raw TSVs or *_cleaned.parquet).

    Returns
    -------
    List[str]
        Sorted, deduplicated entity IDs.

    Raises
    ------
    FileNotFoundError
        If the directory contains no readable source files.
    """
    directory = Path(entities_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Entities directory not found: {directory.resolve()}")

    def _pick(patterns: List[str]) -> List[Path]:
        for pattern in patterns:
            found = sorted(directory.glob(pattern))
            if found:
                return found
        return []

    files = _pick(["test_source*_cleaned.parquet", "test_source*.tsv"])
    if not files:
        files = _pick(["*_cleaned.parquet"])
    if not files:
        files = sorted(
            p for p in directory.glob("*.tsv")
            if "source" in p.name and "ground_truth" not in p.name
        )
    if not files:
        raise FileNotFoundError(
            f"No source files (parquet/TSV) found in: {directory.resolve()}"
        )

    ids: Set[str] = set()
    for file_path in files:
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, sep="\t", dtype=str, encoding="utf-8")
        col = "entity_id" if "entity_id" in df.columns else "cleaned_entity_id"
        if col not in df.columns:
            raise ValueError(f"No entity ID column in {file_path.name}: {list(df.columns)}")
        ids.update(str(v) for v in df[col].dropna().unique())

    logger.info("Loaded %d unique entity IDs from %d file(s) in %s.", len(ids), len(files), directory)
    return sorted(ids)


def run_aggregation(
    predictions_path: Union[str, Path],
    entities_dir: Union[str, Path],
    output_path: Union[str, Path],
    threshold: float = 0.5,
) -> Path:
    """End-to-end aggregation: predictions -> clusters -> submission TSV.

    Parameters
    ----------
    predictions_path : Union[str, Path]
        Parquet file of match predictions (Phase 5 output).
    entities_dir : Union[str, Path]
        Directory containing the 3 source files to collect all entity IDs from.
    output_path : Union[str, Path]
        Destination submission TSV.
    threshold : float, default 0.5
        Score threshold used when predictions lack a `prediction` column.

    Returns
    -------
    Path
        Path to the written submission file.
    """
    predictions_df = pd.read_parquet(predictions_path)
    entity_ids = load_all_entity_ids(entities_dir)

    graph = build_match_graph(predictions_df, threshold=threshold)
    clusters = form_clusters(graph, entity_ids=entity_ids)

    n_singletons = sum(1 for members in clusters.values() if len(members) == 1)
    logger.info(
        "Formed %d clusters (%d singletons) covering %d entities.",
        len(clusters),
        n_singletons,
        sum(len(m) for m in clusters.values()),
    )
    return generate_submission(clusters, output_path)


if __name__ == "__main__":
    import argparse

    from src.utils.config import DATA_DIR, OUTPUTS_DIR

    parser = argparse.ArgumentParser(description="Phase 6: Aggregation & Submission")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=OUTPUTS_DIR / "matches" / "predictions.parquet",
        help="Parquet file of match predictions from Phase 5.",
    )
    parser.add_argument(
        "--entities-dir",
        type=Path,
        default=DATA_DIR / "test",
        help="Directory containing the 3 test source files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "submission.tsv",
        help="Output TSV path for the submission file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold (used only when predictions lack a 'prediction' column).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_aggregation(
        predictions_path=args.predictions,
        entities_dir=args.entities_dir,
        output_path=args.output,
        threshold=args.threshold,
    )
