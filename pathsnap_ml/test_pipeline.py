import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from .data import FEATURES, SCHEMA, align, candidates
from .inference import load, matrix, predict
from .review1 import METRICS, PROTOCOL, SEED, compare, save
from .validation import validate_metrics


class PipelineTest(unittest.TestCase):
    def test_comparison_marks_missing_and_rejects_mismatched_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(compare(folder)["forest"], {"status": "pending"})
            manifest = dict(seed=SEED, schema=SCHEMA, protocol=PROTOCOL, metrics=METRICS,
                            archive_sha256="a"*64,
                            validation_trips=[[str(index) for index in range(start, 80, 3)]
                                              for start in range(3)])
            save(folder / "manifest.json", manifest)
            save(folder / "logistic.json", {"model": "logistic", "manifest": {"seed": 99}})
            with self.assertRaisesRegex(ValueError, "Incompatible result"):
                compare(folder)

    def test_rejects_impossible_top1(self):
        metrics = dict(observations=1, confident=1, coverage=1, accuracy=1, precision=0,
                       recall=0, macro_f1=.5, weighted_f1=1, top1=1, candidate_recall=0,
                       miss_rate=1, brier=0, log_loss=0, confusion=[[8, 0], [0, 0]])
        with self.assertRaisesRegex(ValueError, "Top-1"):
            validate_metrics(metrics)

    def test_alignment_repeats_skips_and_ambiguity(self):
        starts = np.array([[0, 0], [100, 0], [200, 0], [0, 0]])
        ends = starts + [20, 0]
        points = np.array([[2, 1], [10, 1], [210, 1], [8, 1]])
        np.testing.assert_equal(align(points, starts, ends, [0, 1, 2, 0]), [0, 0, 2, 0])
        np.testing.assert_equal(align(points[:1], starts, ends, [0, 3]), [-1])
        np.testing.assert_equal(align([[50, 500]], starts, ends, [0]), [-1])

    def test_reference_free_candidates(self):
        nodes = np.column_stack((np.arange(10) * .0001, np.zeros(10)))
        arcs = np.column_stack((np.arange(9), np.arange(1, 10)))
        track = np.column_stack((nodes[:3], np.arange(3)))
        features, ids, *_ = candidates(track, nodes, arcs)
        self.assertEqual(features.shape, (3, 8, len(FEATURES)))
        self.assertTrue(np.isnan(features[0, :, 1]).all())
        self.assertIn(0, ids[0])
        np.testing.assert_equal(candidates(track, nodes, arcs)[1], ids)

    def test_metrics_grouping_and_serialized_contract(self):
        from .train import assemble, evaluate, fit, GroupKFold

        values = np.tile(np.arange(8, dtype=float)[None, :, None], (3, 1, len(FEATURES)))
        values[:, :, 1] = np.nan
        arcs = np.tile(np.arange(8), (3, 1))
        data = assemble((str(trip), values, arcs, np.array([0, 99, -1]))
                        for trip in range(3))
        train, valid = next(GroupKFold(3).split(data[0], groups=data[2]))
        self.assertFalse(set(data[2][train]) & set(data[2][valid]))
        model, _ = fit("tree", {}, data, train)
        self.assertEqual(evaluate(model, data, valid)["candidate_recall"], .5)
        self.assertEqual(evaluate(model, data, valid)["coverage"], 2/3)
        row = dict(trip_id="trip", observation_index=2, timestamp=100, arc_id=3,
                   features={key: None if key == "heading_cos" else 0
                             for key in reversed(FEATURES)})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(dict(model=model, schema=SCHEMA, features=FEATURES, version="test"), path)
            artifact = load(path)
            result = predict(artifact, [row, row | {"arc_id": 4}])
            self.assertEqual([item["arc_id"] for item in result], [3, 4])
            self.assertTrue(all(0 <= item["score"] <= 1 for item in result))
            self.assertEqual(predict(artifact, []), [])
            with self.assertRaises(ValueError):
                predict(artifact, [row, row])
            with self.assertRaises(ValueError):
                matrix([{"route": 0}])


if __name__ == "__main__":
    unittest.main()
