"""SFT conversations must match the newly frozen tasks and reward provenance."""

import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.collection.data_contract import validate_current_sft
from shopping_grpo.environment.manifest import sha256_file
from shopping_grpo.environment.reward import REWARD_VERSION


class SFTDataContractTests(unittest.TestCase):
    def test_metadata_cannot_hide_wrong_task_membership_or_old_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/sft").mkdir(parents=True)
            (root / "data/splits").mkdir()
            initial, metadata = {}, {"reward": REWARD_VERSION}
            for role, ids in (("train", [1, 2]), ("validation", [3])):
                task_path = root / f"data/sft/{role}_tasks.jsonl"
                data_path = root / f"data/sft/{role}.jsonl"
                task_path.write_text("".join(json.dumps({"task_id": i}) + "\n" for i in ids))
                data_path.write_text(
                    "".join(json.dumps({"task_id": i, "messages": []}) + "\n" for i in ids)
                )
                initial["sft_" + role] = {
                    "path": str(task_path.relative_to(root)),
                    "sha256": sha256_file(task_path),
                    "tasks": len(ids),
                }
                metadata[role] = {"sha256": sha256_file(data_path)}
            (root / "data/splits/manifest.json").write_text(json.dumps({"initial": initial}))
            meta_path = root / "data/sft/metadata.json"
            meta_path.write_text(json.dumps(metadata))
            self.assertEqual(validate_current_sft(root)["train"], 2)
            train = root / "data/sft/train.jsonl"
            train.write_text(
                json.dumps({"task_id": 3, "messages": []})
                + "\n"
                + json.dumps({"task_id": 2, "messages": []})
                + "\n"
            )
            metadata["train"]["sha256"] = sha256_file(train)
            meta_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "不一致"):
                validate_current_sft(root)
            metadata["reward"] = "shopsimulator-reward-v3"
            meta_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "旧数据"):
                validate_current_sft(root)


if __name__ == "__main__":
    unittest.main()
