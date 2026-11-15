"""The canonical test is a fresh stratified selection from original eval tags."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.environment.reward import REWARD_VERSION
from shopping_grpo.evaluation.artifacts import ArtifactError
from shopping_grpo.evaluation.blind_guard import guard_blind_final, validate_canonical_blind_asset

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line.strip()]


class EvaluationDatasetTests(unittest.TestCase):
    def test_strict_test_metadata_and_packaged_guard_match(self):
        path = ROOT / "data/evaluation/tasks.jsonl"
        metadata = json.loads((ROOT / "data/evaluation/metadata.json").read_text())
        guard, ids = validate_canonical_blind_asset()
        self.assertEqual(len(ids), 200)
        self.assertEqual(metadata["reward_version"], REWARD_VERSION)
        self.assertEqual(metadata["source_tag"], "eval")
        self.assertFalse(metadata["evaluated"])
        self.assertEqual(metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(guard["task_sha256"], metadata["sha256"])
        self.assertEqual(
            guard["metadata_sha256"],
            hashlib.sha256((ROOT / "data/evaluation/metadata.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(ids, {r["task_id"] for r in rows("data/evaluation/tasks.jsonl")})

    def test_every_selected_test_task_has_original_eval_tag_and_all_domains_are_present(self):
        labels = {r["task_id"]: r for r in rows("data/splits/labels.jsonl")}
        selected = [labels[r["task_id"]] for r in rows("data/evaluation/tasks.jsonl")]
        self.assertEqual({r["source_tag"] for r in selected}, {"eval"})
        self.assertEqual(len({r["domain"] for r in selected}), 9)
        self.assertEqual({r["difficulty"] for r in selected}, {"simple", "medium", "hard"})
        pool = {r["task_id"] for r in rows("data/splits/pools/evaluation.jsonl")}
        self.assertEqual(len(pool), 944)
        self.assertTrue({r["task_id"] for r in selected} <= pool)

    def test_guard_protects_unselected_eval_reserve_and_cross_tag_duplicates(self):
        reserve = rows("data/splits/reserves/evaluation.jsonl")[0]["task_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            for task_id in [reserve, 13358, 8413, 5423]:
                with self.subTest(task_id=task_id):
                    path.write_text(json.dumps({"task_id": task_id}) + "\n")
                    with self.assertRaises(ArtifactError):
                        guard_blind_final([path], allowed=False)
        guard_blind_final(
            [ROOT / "data/grpo/train.jsonl", ROOT / "data/grpo/validation.jsonl"], allowed=False
        )


if __name__ == "__main__":
    unittest.main()
