"""CPU coverage for frozen-pool sampling, distinct batches and checkpoint recovery."""

import copy
from collections import Counter
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from shopping_grpo.training.grpo.task_sampling import (
    ROOT,
    StratifiedTaskSampler,
    install_task_sampling,
)

SAMPLING = {
    "enable": True,
    "seed": 42,
    "difficulty_weights": [0.1, 0.3, 0.6],
    "preserve_domain_marginal": True,
}


def ids():
    return [
        r["task_id"]
        for r in pq.read_table(
            ROOT / "data/grpo/train.parquet", columns=["extra_info"]
        ).to_pydict()["extra_info"]
    ]


def make_sampler(**kwargs):
    return StratifiedTaskSampler(ids(), SAMPLING, **kwargs)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_exact_difficulty_and_natural_category_marginals_with_distinct_batches(batch_size):
    sampler = make_sampler(batch_size=batch_size)
    for epoch in range(3):
        drawn = sampler.schedule(epoch)
        rows = [sampler.rows[index] for index in drawn]
        assert len(drawn) == 1000
        assert Counter(r["difficulty"] for r in rows) == {"simple": 100, "medium": 300, "hard": 600}
        assert Counter(r["domain"] for r in rows) == sampler.domain_counts
        assert len(sampler.domain_counts) == 9
        assert all(len({r["group_id"] for r in rows[i:i + batch_size]}) == batch_size
                   for i in range(0, 1000, batch_size))
        assert {r["task_id"] for r in rows} <= set(ids())
    assert sampler.schedule(0) == make_sampler(batch_size=batch_size).schedule(0)
    assert sampler.schedule(0) != sampler.schedule(1)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_iterator_and_sampler_restore_exact_position_and_next_epoch(batch_size):
    sampler = make_sampler(batch_size=batch_size)
    iterator = iter(sampler)
    prefix = [next(iterator) for _ in range(38)]
    sampler_state, iterator_state = sampler.state_dict(), iterator.state_dict()
    rest = list(iterator)
    next_epoch = list(sampler)
    restored = make_sampler(batch_size=batch_size)
    restored.load_state_dict(sampler_state)
    restored_iterator = iter(restored)
    restored_iterator.load_state_dict(iterator_state)
    assert list(restored_iterator) == rest
    assert list(restored) == next_epoch
    assert prefix + rest == sampler.schedule(0)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_stateful_dataloader_checkpoint_replays_batches_and_epoch_boundary(batch_size):
    from torchdata.stateful_dataloader import StatefulDataLoader

    class Dataset:
        def __len__(self):
            return 1000

        def __getitem__(self, index):
            return int(index)

    def loader():
        return StatefulDataLoader(
            Dataset(), batch_size=batch_size, sampler=make_sampler(batch_size=batch_size),
            num_workers=0, drop_last=True
        )

    original = loader()
    iterator = iter(original)
    for _ in range(23):
        next(iterator)
    state = copy.deepcopy(original.state_dict())
    expected = [batch.tolist() for batch in iterator]
    expected_next = [batch.tolist() for batch in original]
    resumed = loader()
    resumed.load_state_dict(state)
    assert [batch.tolist() for batch in resumed] == expected
    assert [batch.tolist() for batch in resumed] == expected_next


def test_restore_rejects_different_mixture_and_modified_membership():
    state = make_sampler().state_dict()
    other = StratifiedTaskSampler(ids(), {**SAMPLING, "difficulty_weights": [0.2, 0.5, 0.3]})
    with pytest.raises(ValueError, match="different"):
        other.load_state_dict(state)
    with pytest.raises(ValueError, match="different"):
        make_sampler(batch_size=4).load_state_dict(state)
    with pytest.raises(ValueError, match="complete frozen"):
        StratifiedTaskSampler(ids()[:-1], SAMPLING)
    with pytest.raises(ValueError, match="complete frozen"):
        StratifiedTaskSampler([0, *ids()[1:]], SAMPLING)


@pytest.mark.parametrize("weights", [[0, 0.5, 0.5], [float("nan"), 0.3, 0.6], [0.1, 0.9]])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="three positive"):
        StratifiedTaskSampler(ids(), {**SAMPLING, "difficulty_weights": weights})


@pytest.mark.parametrize("batch_size", [2, 4])
def test_hook_replaces_only_training_sampler_and_is_idempotent(tmp_path, monkeypatch, batch_size):
    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    observed = []

    def original(self, train_dataset, val_dataset, collate_fn, train_sampler=None):
        observed.append((train_dataset, val_dataset, collate_fn, train_sampler))

    monkeypatch.setattr(RayPPOTrainer, "_create_dataloader", original)
    install_task_sampling()
    installed = RayPPOTrainer._create_dataloader
    install_task_sampling()
    assert RayPPOTrainer._create_dataloader is installed
    dataset = SimpleNamespace(dataframe={"extra_info": [{"task_id": i} for i in ids()]})
    validation, collate, old_sampler = object(), object(), object()
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "shopping_task_sampling": SAMPLING,
                "data": {"train_batch_size": batch_size},
                "trainer": {"default_local_dir": str(tmp_path)},
            }
        )
    )
    installed(trainer, dataset, validation, collate, old_sampler)
    assert observed[-1][:3] == (dataset, validation, collate)
    assert isinstance(observed[-1][3], StratifiedTaskSampler)
    assert trainer.shopping_task_sampler is observed[-1][3]
    assert trainer.shopping_task_sampler.batch_size == batch_size
    trainer.config.shopping_task_sampling.enable = False
    installed(trainer, dataset, validation, collate, old_sampler)
    assert observed[-1][3] is old_sampler


def test_planned_diagnostics_are_written_only_when_iterated(tmp_path):
    path = tmp_path / "sampling.jsonl"
    sampler = make_sampler(diagnostics_path=path)
    sampler.summary()
    sampler.schedule(0)
    assert not path.exists()
    iterator = iter(sampler)
    assert path.exists()
    assert len(list(iterator)) == 1000


def test_real_dataset_order_is_used_for_returned_indices():
    reversed_ids = list(reversed(ids()))
    sampler = StratifiedTaskSampler(reversed_ids, SAMPLING)
    drawn_ids = [reversed_ids[index] for index in sampler.schedule(0)]
    assert drawn_ids == [sampler.rows[index]["task_id"] for index in sampler.schedule(0)]


def test_four_task_repair_handles_duplicate_instruction_groups_without_losing_draws():
    sampler = object.__new__(StratifiedTaskSampler)
    sampler.batch_size = 4
    sampler.rows = [{"group_id": gid} for gid in ["a", "a", "b", "c", "d", "b", "c", "d"]]
    drawn = sampler._distinct_four_task_batches(list(range(8)))
    assert sorted(drawn) == list(range(8))
    assert all(len({sampler.rows[i]["group_id"] for i in drawn[start:start + 4]}) == 4
               for start in (0, 4))
