"""Stateful category/difficulty sampling over the frozen RL training task IDs."""

from __future__ import annotations

import functools
import hashlib
import json
import random
from collections import Counter, defaultdict
from fractions import Fraction
from math import lcm
from pathlib import Path

from shopping_grpo.collection.stratification import (
    LEVELS,
    bounded_transport,
    fit_marginals,
    proportional_counts,
)
from shopping_grpo.environment.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[4]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class StratifiedTaskSampler:
    """Exact epoch marginals, shuffled cycling within each category/difficulty cell.

    The source dataset stays unchanged. Draws can repeat tasks across batches;
    within each candidate batch, duplicate instruction groups are excluded.
    Both this sampler and its iterator expose torchdata's Stateful protocol.
    """

    def __init__(self, task_ids, config, *, root=ROOT, diagnostics_path=None, batch_size=2):
        self.task_ids = [int(i) for i in task_ids]
        self.batch_size = int(batch_size)
        if self.batch_size not in (2, 4):
            raise ValueError("task sampling requires batches of 2 or 4")
        self.seed = int(config.get("seed", 42))
        self.weights = [float(v) for v in config["difficulty_weights"]]
        if self.seed != 42:
            raise ValueError("task sampler requires the fixed seed=42")
        if len(self.weights) != 3 or any(not 0 < v < float("inf") for v in self.weights):
            raise ValueError("difficulty_weights must contain three positive finite values")
        if not config.get("preserve_domain_marginal", True):
            raise ValueError("the selected recipe preserves natural category marginals")
        root = Path(root)
        manifest = json.loads((root / "data/splits/manifest.json").read_text())
        train_spec = manifest["initial"]["rl_train"]
        train_path = root / train_spec["path"]
        if sha256_file(train_path) != train_spec["sha256"]:
            raise ValueError("frozen RL training IDs do not match the split manifest")
        train_ids = {r["task_id"] for r in read_rows(train_path)}
        if len(self.task_ids) != len(train_ids) or set(self.task_ids) != train_ids:
            raise ValueError("task sampler requires the complete frozen RL training dataset")
        if len(self.task_ids) % self.batch_size:
            raise ValueError("task batch size must divide the candidate epoch size")
        labels_path = root / "data/splits/labels.jsonl"
        if sha256_file(labels_path) != manifest["files"]["data/splits/labels.jsonl"]:
            raise ValueError("frozen task labels do not match the split manifest")
        labels = {r["task_id"]: r for r in read_rows(labels_path)}
        protected = {r["group_id"] for r in labels.values() if r["source_tag"] == "eval"}
        self.rows = [labels[i] for i in self.task_ids]
        if any(r["source_tag"] != "train" or r["group_id"] in protected for r in self.rows):
            raise ValueError(
                "task sampler cannot draw held-out or cross-tag instruction duplicates"
            )
        self.domains = sorted({r["domain"] for r in self.rows})
        self.cells = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.cells[row["domain"], row["difficulty"]].append(index)
        self.domain_counts = Counter(r["domain"] for r in self.rows)
        fractions = [Fraction(str(v)) for v in self.weights]
        denominator = lcm(*(v.denominator for v in fractions))
        integer_weights = [int(v * denominator) for v in fractions]
        self.difficulty_counts = proportional_counts(
            dict(zip(LEVELS, integer_weights)), len(self.task_ids)
        )
        counts = [[len(self.cells[d, level]) for level in LEVELS] for d in self.domains]
        row_totals = [self.domain_counts[d] for d in self.domains]
        col_totals = [self.difficulty_counts[level] for level in LEVELS]
        ideals = fit_marginals(counts, row_totals, col_totals)
        capacities = [[len(self.task_ids) if n else 0 for n in row] for row in counts]
        self.quotas = bounded_transport(row_totals, col_totals, ideals, capacities=capacities)
        self.epoch = 0
        self.diagnostics_path = Path(diagnostics_path) if diagnostics_path else None
        fingerprint = {
            "task_ids_in_dataset_order": self.task_ids,
            "labels": [(r["group_id"], r["domain"], r["difficulty"]) for r in self.rows],
            "seed": self.seed,
            "weights": self.weights,
            "quotas": self.quotas,
        }
        if self.batch_size != 2:
            fingerprint["batch_size"] = self.batch_size
        self.fingerprint = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest()

    def __len__(self):
        return len(self.task_ids)

    def summary(self):
        return {
            "version": "shopping-stratified-task-sampler-v1",
            "seed": self.seed,
            "source_unique_tasks": len(self.task_ids),
            "candidate_draws_per_epoch": len(self),
            "candidate_batch_size": self.batch_size,
            "difficulty_order": list(LEVELS),
            "difficulty_weights": self.weights,
            "candidate_difficulty_counts_per_epoch": dict(self.difficulty_counts),
            "source_difficulty_counts": dict(Counter(r["difficulty"] for r in self.rows)),
            "candidate_domain_counts_per_epoch": dict(self.domain_counts),
            "domain_difficulty_counts": {
                d: dict(zip(LEVELS, self.quotas[i])) for i, d in enumerate(self.domains)
            },
            "source_membership_unchanged": True,
            "within_batch_instruction_groups_distinct": True,
            "fingerprint": self.fingerprint,
        }

    def schedule(self, epoch):
        seed = int(hashlib.sha256(f"{self.seed}:task-epoch:{epoch}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        draws = []
        for i, domain in enumerate(self.domains):
            for j, level in enumerate(LEVELS):
                remaining = self.quotas[i][j]
                while remaining:
                    cycle = list(self.cells[domain, level])
                    rng.shuffle(cycle)
                    take = min(remaining, len(cycle))
                    draws.extend(cycle[:take])
                    remaining -= take
        rng.shuffle(draws)
        if self.batch_size == 4:
            return self._distinct_four_task_batches(draws)
        if len(draws) % 2:
            raise ValueError("two-task batches require an even candidate epoch size")
        gids = [self.rows[index]["group_id"] for index in draws]
        if max(Counter(gids).values()) > len(draws) // 2:
            raise ValueError("instruction multiplicity prevents distinct two-task batches")
        for start in range(0, len(draws), 2):
            if gids[start] != gids[start + 1]:
                continue
            replacement = next(
                (
                    k
                    for k in range(len(draws))
                    if k not in (start, start + 1)
                    and gids[k] != gids[start]
                    and gids[k ^ 1] != gids[start]
                ),
                None,
            )
            if replacement is None:
                raise ValueError("could not form distinct instruction groups in a candidate batch")
            draws[start + 1], draws[replacement] = draws[replacement], draws[start + 1]
            gids[start + 1], gids[replacement] = gids[replacement], gids[start + 1]
        return draws

    def _distinct_four_task_batches(self, draws):
        """Repair the shuffled epoch by swaps, retaining all category/difficulty quotas."""
        gids = [self.rows[index]["group_id"] for index in draws]
        if max(Counter(gids).values()) > len(draws) // self.batch_size:
            raise ValueError("instruction multiplicity prevents distinct four-task batches")
        for start in range(0, len(draws), self.batch_size):
            seen = set()
            for position in range(start, start + self.batch_size):
                if gids[position] in seen:
                    replacement = next((
                        k for k in range(len(draws))
                        if not start <= k < start + self.batch_size
                        and gids[k] not in seen
                        and all(
                            j == k or gids[j] != gids[position]
                            for j in range(k // self.batch_size * self.batch_size,
                                           (k // self.batch_size + 1) * self.batch_size)
                        )
                    ), None)
                    if replacement is None:
                        raise ValueError("could not form distinct four-task candidate batches")
                    draws[position], draws[replacement] = draws[replacement], draws[position]
                    gids[position], gids[replacement] = gids[replacement], gids[position]
                seen.add(gids[position])
        return draws

    def __iter__(self):
        return _TaskSamplerIterator(self, self.epoch)

    def state_dict(self):
        return {"epoch": self.epoch, "fingerprint": self.fingerprint}

    def load_state_dict(self, state):
        if state["fingerprint"] != self.fingerprint or int(state["epoch"]) < 0:
            raise ValueError("sampler checkpoint belongs to different data or sampling weights")
        self.epoch = int(state["epoch"])


class _TaskSamplerIterator:
    def __init__(self, sampler, epoch):
        self.sampler, self.epoch = sampler, epoch
        self.draws = sampler.schedule(epoch)
        self.position = 0
        if sampler.diagnostics_path:
            sampler.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "event": "candidate_epoch_plan",
                "epoch": epoch,
                **sampler.summary(),
                "planned_task_ids": [sampler.task_ids[i] for i in self.draws],
            }
            with sampler.diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == len(self.draws):
            self.sampler.epoch = max(self.sampler.epoch, self.epoch + 1)
            raise StopIteration
        index = self.draws[self.position]
        self.position += 1
        return index

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "position": self.position,
            "fingerprint": self.sampler.fingerprint,
        }

    def load_state_dict(self, state):
        if state["fingerprint"] != self.sampler.fingerprint or not 0 <= int(
            state["position"]
        ) <= len(self.sampler):
            raise ValueError("invalid sampler iterator checkpoint")
        self.epoch, self.position = int(state["epoch"]), int(state["position"])
        if self.epoch < 0:
            raise ValueError("invalid sampler epoch")
        self.draws = self.sampler.schedule(self.epoch)
        self.sampler.epoch = max(self.sampler.epoch, self.epoch)


def install_task_sampling():
    """Use the trainer's dataset hook; only the training loader is replaced."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    original = RayPPOTrainer._create_dataloader
    if getattr(original, "_shopping_task_sampling", False):
        return

    @functools.wraps(original)
    def create(self, train_dataset, val_dataset, collate_fn, train_sampler=None):
        sampling = self.config.get("shopping_task_sampling", {})
        if sampling.get("enable", False):
            batch = self.config.data.get("gen_batch_size") or self.config.data.train_batch_size
            if int(batch) not in (2, 4) or train_dataset is None:
                raise ValueError(
                    "selected task sampler requires the native dataset and batches of 2 or 4"
                )
            task_ids = [row["task_id"] for row in train_dataset.dataframe["extra_info"]]
            output = Path(self.config.trainer.default_local_dir)
            train_sampler = StratifiedTaskSampler(
                task_ids, sampling, diagnostics_path=output / "task_sampling_diagnostics.jsonl",
                batch_size=int(batch),
            )
            print(
                "SHOPPING_TASK_SAMPLER " + json.dumps(train_sampler.summary(), ensure_ascii=False),
                flush=True,
            )
            self.shopping_task_sampler = train_sampler
        return original(self, train_dataset, val_dataset, collate_fn, train_sampler)

    create._shopping_task_sampling = True
    RayPPOTrainer._create_dataloader = create
