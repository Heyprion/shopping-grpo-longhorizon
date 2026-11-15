"""Joint margins, grouped leakage protection, and reserve extension regression."""

import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from shopping_grpo.collection.stratification import (
    LEVELS,
    bounded_transport,
    fit_marginals,
    label_task,
    partition_pools,
    proportional_counts,
    select_tasks,
)


def task(i, domain="A", difficulty="simple", group=None, tag="train"):
    return {
        "task_id": i,
        "domain": domain,
        "difficulty": difficulty,
        "group_id": str(i) if group is None else group,
        "source_tag": tag,
    }


class StratificationTests(unittest.TestCase):
    def test_structural_boundaries_use_extracted_hard_requirements(self):
        for n, options, axes, expected in [
            (3, 5, 1, "simple"),
            (4, 5, 1, "medium"),
            (6, 19, 1, "medium"),
            (7, 6, 1, "hard"),
            (3, 20, 1, "medium"),
            (4, 2, 2, "hard"),
        ]:
            cleaned = {
                "task_id": 1,
                "asin": "x",
                "record_hash": "h",
                "budget": {"query_sha256": "q"},
                "requirements": [{"hard": True, "kind": "spec"}] * n
                + [{"hard": True, "kind": "category"}, {"hard": False, "kind": "spec"}],
            }
            product = {
                "tag": "train",
                "domain_zh": "A",
                "category": "a",
                "customization_options": {str(i): list(range(options)) for i in range(axes)},
            }
            with self.subTest(n=n, options=options, axes=axes):
                result = label_task(cleaned, product)
                self.assertEqual(result["difficulty"], expected)
                self.assertEqual(result["hard_requirement_count"], n)

    def test_transport_has_exact_margins_bounds_and_group_lower_limits(self):
        ideal = fit_marginals([[80, 10, 10], [10, 40, 50]], [50, 50], [30, 50, 20])
        matrix = bounded_transport(
            [50, 50],
            [30, 50, 20],
            ideal,
            capacities=[[80, 30, 10], [10, 40, 50]],
            lower=[[0, 0, 0], [2, 0, 0]],
        )
        self.assertEqual([sum(row) for row in matrix], [50, 50])
        self.assertEqual([sum(row[j] for row in matrix) for j in range(3)], [30, 50, 20])
        self.assertGreaterEqual(matrix[1][0], 2)
        self.assertLessEqual(matrix[0][2], 10)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            bounded_transport([2, 2], [2, 2], [[1, 1], [1, 1]], capacities=[[1, 0], [0, 1]])

    def test_selection_matches_natural_categories_and_reweighted_difficulty(self):
        rows = []
        for domain, counts in [("A", [80, 10, 10]), ("B", [10, 40, 50])]:
            for level, count in zip(LEVELS, counts):
                start = len(rows)
                rows.extend(task(start + i, domain, level) for i in range(count))
        selected = select_tasks(rows, 60, ratios=(30, 50, 20))
        self.assertEqual(Counter(r["domain"] for r in selected), {"A": 30, "B": 30})
        self.assertEqual(
            Counter(r["difficulty"] for r in selected), {"simple": 18, "medium": 30, "hard": 12}
        )
        self.assertEqual(selected, select_tasks(rows, 60, ratios=(30, 50, 20)))
        self.assertEqual(len({r["task_id"] for r in selected}), 60)

    def test_repeated_instructions_do_not_cross_pools_and_eval_duplicates_are_excluded(self):
        rows = [task(i, difficulty=LEVELS[i % 3]) for i in range(100)]
        rows[0]["group_id"] = rows[1]["group_id"] = "train-pair"
        rows.extend([task(100, tag="eval", group="cross"), task(101, group="cross")])
        sizes = {"sft_train": 40, "sft_validation": 10, "rl_train": 40, "rl_validation": 10}
        with patch("shopping_grpo.collection.stratification.POOL_SIZES", sizes):
            pools = partition_pools(rows)
        self.assertEqual([r["task_id"] for r in pools["quarantine"]], [101])
        self.assertEqual([r["task_id"] for r in pools["evaluation"]], [100])
        containing = [
            name for name, group in pools.items() if any(r["task_id"] == 0 for r in group)
        ]
        self.assertEqual(len(containing), 1)
        self.assertIn(1, [r["task_id"] for r in pools[containing[0]]])
        self.assertEqual({name: len(pools[name]) for name in sizes}, sizes)

    def test_hamilton_quota_is_within_one_of_natural_proportion(self):
        weights = {"A": 168, "B": 49, "C": 48}
        quotas = proportional_counts(weights, 50)
        self.assertEqual(sum(quotas.values()), 50)
        for key, value in weights.items():
            self.assertLess(abs(quotas[key] - value * 50 / sum(weights.values())), 1)

    def test_extension_is_read_only_in_preview_and_never_reuses_selected_tasks(self):
        from scripts.prepare_data_splits import extend, verify
        from shopping_grpo.environment.manifest import sha256_file

        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "data/splits/manifest.json"
        if not manifest_path.exists():
            self.skipTest("materialized population is not available")
        manifest = json.loads(manifest_path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for name in [*manifest["files"], "data/splits/manifest.json"]:
                destination = temporary / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, destination)
            before = sha256_file(temporary / "data/splits/manifest.json")
            preview = extend(temporary, "sft", 100, dry_run=True)
            self.assertEqual(before, sha256_file(temporary / "data/splits/manifest.json"))
            batch = extend(temporary, "sft", 100)
            first = {
                json.loads(x)["task_id"]
                for x in (temporary / batch["path"]).read_text().splitlines()
            }
            self.assertEqual(first, set(preview["task_ids"]))
            second = extend(temporary, "sft", 100)
            second_ids = {
                json.loads(x)["task_id"]
                for x in (temporary / second["path"]).read_text().splitlines()
            }
            self.assertFalse(first & second_ids)
            for name in ("sft_validation", "rl_validation", "evaluation"):
                path = manifest["initial"][name]["path"]
                self.assertEqual((root / path).read_bytes(), (temporary / path).read_bytes())
            self.assertEqual(verify(temporary)["rounds"], 2)


if __name__ == "__main__":
    unittest.main()
