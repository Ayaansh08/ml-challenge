"""Unit tests for Phase 6 aggregation (clustering & submission)."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.aggregation import (
    build_match_graph,
    find_clusters,
    form_clusters,
    generate_submission,
    load_all_entity_ids,
    run_aggregation,
)


def make_predictions(rows, include_prediction=True):
    """Build a predictions DataFrame with the Phase 5 schema."""
    columns = ["entity_id_1", "entity_id_2", "score"]
    if include_prediction:
        columns.append("prediction")
        data = rows
    else:
        data = [row[:3] for row in rows]
    return pd.DataFrame(data, columns=columns)


class TestBuildMatchGraph(unittest.TestCase):
    """Tests for match graph construction."""

    def test_filters_on_prediction_column(self):
        df = make_predictions([
            ("S1-1", "S2-1", 0.9, 1),
            ("S1-2", "S2-2", 0.4, 0),
            ("S1-3", "S3-3", 0.95, 1),
        ])
        graph = build_match_graph(df)
        self.assertEqual(graph.number_of_nodes(), 4)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertFalse(graph.has_edge("S1-2", "S2-2"))
        # Score edge attribute preserved
        self.assertAlmostEqual(graph["S1-1"]["S2-1"]["score"], 0.9)

    def test_threshold_used_when_prediction_missing(self):
        df = make_predictions([
            ("S1-1", "S2-1", 0.9, 1),
            ("S1-2", "S2-2", 0.3, 0),
        ], include_prediction=False)
        graph = build_match_graph(df, threshold=0.5)
        self.assertTrue(graph.has_edge("S1-1", "S2-1"))
        self.assertFalse(graph.has_edge("S1-2", "S2-2"))

    def test_missing_required_columns_raises(self):
        with self.assertRaises(ValueError):
            build_match_graph(pd.DataFrame({"entity_id_1": ["S1-1"]}))


class TestFindClusters(unittest.TestCase):
    """Tests for connected-components clustering."""

    def test_triangle_transitive_closure(self):
        """A-B and B-C matches must merge A, B, C into one cluster."""
        df = make_predictions([
            ("S1-1", "S2-1", 0.9, 1),
            ("S2-1", "S3-1", 0.8, 1),
        ])
        graph = build_match_graph(df)
        clusters = find_clusters(graph)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(next(iter(clusters.values())), {"S1-1", "S2-1", "S3-1"})

    def test_disconnected_pairs_form_separate_clusters(self):
        df = make_predictions([
            ("S1-1", "S2-1", 0.9, 1),
            ("S1-2", "S3-2", 0.9, 1),
        ])
        clusters = find_clusters(build_match_graph(df))
        self.assertEqual(len(clusters), 2)

    def test_deterministic_cluster_ids(self):
        """Same graph (any row order) yields identical, sequential cluster IDs."""
        rows = [
            ("S1-1", "S2-1", 0.9, 1),
            ("S1-2", "S3-2", 0.8, 1),
            ("S1-3", "S2-3", 0.7, 1),
        ]
        clusters_a = find_clusters(build_match_graph(make_predictions(rows)))
        clusters_b = find_clusters(build_match_graph(make_predictions(list(reversed(rows)))))

        self.assertEqual(clusters_a, clusters_b)
        self.assertEqual(sorted(clusters_a.keys()), list(range(1, len(clusters_a) + 1)))
        # IDs are assigned by sorted member tuple, independent of row order
        first_members = [sorted(clusters_a[k]) for k in sorted(clusters_a)]
        self.assertEqual(first_members, sorted(first_members))


class TestSingletons(unittest.TestCase):
    """Tests for singleton cluster assignment."""

    def test_unmatched_entity_gets_own_singleton_cluster(self):
        df = make_predictions([("S1-1", "S2-1", 0.9, 1)])
        graph = build_match_graph(df)
        clusters = form_clusters(graph, entity_ids=["S1-1", "S2-1", "S1-9"])

        all_members = [eid for members in clusters.values() for eid in members]
        self.assertEqual(sorted(all_members), ["S1-1", "S1-9", "S2-1"])

        singleton = [members for members in clusters.values() if members == {"S1-9"}]
        self.assertEqual(len(singleton), 1)
        # Matched pair stays merged
        pair = [members for members in clusters.values() if "S1-1" in members]
        self.assertEqual(pair[0], {"S1-1", "S2-1"})


class TestGenerateSubmission(unittest.TestCase):
    """Tests for submission file generation."""

    def test_submission_format(self):
        clusters = {1: {"S1-1", "S2-1"}, 2: {"S1-2"}}
        with tempfile.TemporaryDirectory() as tmp:
            out = generate_submission(clusters, Path(tmp) / "sub" / "submission.tsv")
            self.assertTrue(out.exists())

            raw = out.read_text(encoding="utf-8")
            lines = raw.strip().split("\n")
            self.assertEqual(lines[0], "entity_id\tcluster_id")

            df = pd.read_csv(out, sep="\t", dtype={"entity_id": str})
            self.assertEqual(list(df.columns), ["entity_id", "cluster_id"])
            # Sorted by entity_id
            self.assertEqual(list(df["entity_id"]), sorted(df["entity_id"]))
            self.assertEqual(set(df["cluster_id"]), {1, 2})

    def test_duplicate_entity_across_clusters_raises(self):
        clusters = {1: {"S1-1"}, 2: {"S1-1"}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                generate_submission(clusters, Path(tmp) / "submission.tsv")


class TestRunAggregation(unittest.TestCase):
    """End-to-end tests for run_aggregation."""

    def _write_sources(self, directory: Path):
        for idx, ids in enumerate(
            [["S1-1", "S1-2"], ["S2-1", "S2-2"], ["S3-1"]], start=1
        ):
            df = pd.DataFrame({
                "entity_id": ids,
                "business_name": [f"name-{e}" for e in ids],
                "business_address": [f"addr-{e}" for e in ids],
                "country": ["US"] * len(ids),
            })
            df.to_csv(directory / f"test_source{idx}.tsv", sep="\t", index=False)
        return [e for ids in [["S1-1", "S1-2"], ["S2-1", "S2-2"], ["S3-1"]] for e in ids]

    def test_all_entities_present_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            expected_ids = self._write_sources(tmp_path)

            preds = make_predictions([
                ("S1-1", "S2-1", 0.95, 1),
                ("S1-2", "S2-2", 0.2, 0),
            ])
            preds_path = tmp_path / "predictions.parquet"
            preds.to_parquet(preds_path, index=False)

            out = run_aggregation(
                predictions_path=preds_path,
                entities_dir=tmp_path,
                output_path=tmp_path / "submission.tsv",
                threshold=0.5,
            )

            df = pd.read_csv(out, sep="\t", dtype={"entity_id": str})
            self.assertEqual(len(df), len(expected_ids))
            self.assertEqual(sorted(df["entity_id"]), sorted(expected_ids))
            self.assertEqual(df["entity_id"].is_unique, True)

            # Matched pair merged; predicted non-match and unmatched stay singletons
            cluster_of = dict(zip(df["entity_id"], df["cluster_id"]))
            self.assertEqual(cluster_of["S1-1"], cluster_of["S2-1"])
            self.assertNotEqual(cluster_of["S1-2"], cluster_of["S2-2"])
            self.assertNotEqual(cluster_of["S1-1"], cluster_of["S1-2"])

            cluster_sizes = df["cluster_id"].value_counts()
            self.assertEqual(int(cluster_sizes[cluster_of["S1-2"]]), 1)
            self.assertEqual(int(cluster_sizes[cluster_of["S2-2"]]), 1)
            self.assertEqual(int(cluster_sizes[cluster_of["S3-1"]]), 1)
            self.assertEqual(int(cluster_sizes[cluster_of["S1-1"]]), 2)

    def test_load_all_entity_ids_prefers_test_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            expected_ids = self._write_sources(tmp_path)
            ids = load_all_entity_ids(tmp_path)
            self.assertEqual(sorted(ids), sorted(expected_ids))


if __name__ == "__main__":
    unittest.main()
