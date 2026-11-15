"""Deterministic, capacity-constrained category/difficulty task allocation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque

LABEL_VERSION = "shopping-structural-complexity-v1"
SPLIT_VERSION = "shopping-stratified-task-split-v1"
EVALUATION_ASSET = "shopsimulator-strict-stratified-200"
EVALUATION_CONTRACT = "environment-v2.1/strict-v1/stratified-task-split-v1"
LEVELS = ("simple", "medium", "hard")
POOL_SIZES = {"sft_train": 4500, "sft_validation": 500, "rl_train": 10494, "rl_validation": 300}
TRAIN_RATIOS = {"sft": (30, 50, 20), "rl": (20, 50, 30)}


def stable_rank(seed, namespace, value):
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def label_task(cleaned, product):
    n = sum(r["hard"] and r["kind"] != "category" for r in cleaned["requirements"])
    counts = [len(v) for v in product["customization_options"].values()]
    axes = sum(c > 1 for c in counts)
    maximum = max(counts, default=1)
    requirement_score = 0 if n <= 3 else 1 if n <= 6 else 2
    option_score = 2 if axes >= 2 or maximum >= 20 else 1 if maximum >= 6 else 0
    score = requirement_score + option_score
    return {
        "task_id": cleaned["task_id"],
        "asin": cleaned["asin"],
        "source_tag": product["tag"],
        "domain": product["domain_zh"],
        "category": product["category"],
        "hard_requirement_count": n,
        "variable_option_axes": axes,
        "max_axis_options": maximum,
        "requirement_score": requirement_score,
        "option_score": option_score,
        "complexity_score": score,
        "difficulty": "simple" if score == 0 else "medium" if score <= 2 else "hard",
        "group_id": cleaned["budget"]["query_sha256"],
        "record_hash": cleaned["record_hash"],
        "label_version": LABEL_VERSION,
    }


def proportional_counts(weights, total):
    """Hamilton rounding: exact total, every marginal within one of its ideal."""
    if (
        total < 0
        or not weights
        or any(v < 0 for v in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("invalid proportional allocation")
    denominator = sum(weights.values())
    result = {k: v * total // denominator for k, v in weights.items()}
    order = sorted(weights, key=lambda k: (-(weights[k] * total % denominator), str(k)))
    for key in order[: total - sum(result.values())]:
        result[key] += 1
    return result


def fit_marginals(weights, row_totals, col_totals):
    """IPF preserves observed associations while fitting both target margins."""
    if sum(row_totals) != sum(col_totals):
        raise ValueError("marginal totals differ")
    matrix = [[float(x) for x in row] for row in weights]
    for _ in range(10000):
        for i, target in enumerate(row_totals):
            actual = sum(matrix[i])
            if target and not actual:
                raise ValueError("empty category cannot meet quota")
            matrix[i] = [x * target / actual if actual else 0.0 for x in matrix[i]]
        for j, target in enumerate(col_totals):
            actual = sum(row[j] for row in matrix)
            if target and not actual:
                raise ValueError("empty difficulty cannot meet quota")
            for row in matrix:
                row[j] *= target / actual if actual else 0.0
        error = max((abs(sum(row) - target) for row, target in zip(matrix, row_totals)), default=0)
        if error < 1e-9:
            return matrix
    raise ValueError("category/difficulty margins are infeasible")


def bounded_transport(row_totals, col_totals, ideals, capacities=None, lower=None):
    """Integer min-cost transport with exact margins and absolute-error costs."""
    nr, nc = len(row_totals), len(col_totals)
    if sum(row_totals) != sum(col_totals):
        raise ValueError("marginal totals differ")
    capacities = capacities or [[min(r, c) for c in col_totals] for r in row_totals]
    lower = lower or [[0] * nc for _ in row_totals]
    result = [row[:] for row in lower]
    supply = [r - sum(row) for r, row in zip(row_totals, lower)]
    demand = [c - sum(lower[i][j] for i in range(nr)) for j, c in enumerate(col_totals)]
    if any(x < 0 for x in [*supply, *demand]):
        raise ValueError("group lower bounds exceed margins")
    graph = [[] for _ in range(nr + nc + 2)]
    source, sink = nr + nc, nr + nc + 1

    def add(a, b, capacity, cost):
        forward = [b, capacity, cost, len(graph[b])]
        reverse = [a, 0, -cost, len(graph[a])]
        graph[a].append(forward)
        graph[b].append(reverse)
        return forward

    for i, amount in enumerate(supply):
        add(source, i, amount, 0)
    for j, amount in enumerate(demand):
        add(nr + j, sink, amount, 0)
    edges = []
    scale = 1_000_000
    for i in range(nr):
        for j in range(nc):
            start, maximum = lower[i][j], capacities[i][j]
            if start < 0 or maximum < start:
                raise ValueError("invalid cell capacity")
            ideal = ideals[i][j]
            floor = math.floor(ideal)
            for end, cost in [
                (min(floor, maximum), -scale),
                (min(floor + 1, maximum), round(scale * (1 - 2 * (ideal - floor)))),
                (maximum, scale),
            ]:
                if end > start:
                    edge = add(i, nr + j, end - start, cost)
                    edges.append((i, j, edge, end - start))
                    start = end
    remaining = sum(supply)
    while remaining:
        dist = [math.inf] * len(graph)
        previous = [None] * len(graph)
        dist[source] = 0
        queue, queued = deque([source]), {source}
        while queue:
            node = queue.popleft()
            queued.remove(node)
            for index, edge in enumerate(graph[node]):
                target, capacity, cost, _ = edge
                if capacity and dist[node] + cost < dist[target]:
                    dist[target] = dist[node] + cost
                    previous[target] = (node, index)
                    if target not in queued:
                        queue.append(target)
                        queued.add(target)
        if previous[sink] is None:
            raise ValueError("insufficient tasks for requested category/difficulty quotas")
        amount, node = remaining, sink
        while node != source:
            parent, index = previous[node]
            amount = min(amount, graph[parent][index][1])
            node = parent
        node = sink
        while node != source:
            parent, index = previous[node]
            edge = graph[parent][index]
            edge[1] -= amount
            graph[node][edge[3]][1] += amount
            node = parent
        remaining -= amount
    for i, j, edge, original in edges:
        result[i][j] += original - edge[1]
    return result


def distribution(rows):
    return {
        "tasks": len(rows),
        "difficulty": {k: sum(r["difficulty"] == k for r in rows) for k in LEVELS},
        "domain": dict(sorted(Counter(r["domain"] for r in rows).items())),
        "joint": {
            f"{d}|{level}": count
            for (d, level), count in sorted(
                Counter((r["domain"], r["difficulty"]) for r in rows).items()
            )
        },
    }


def partition_pools(labels, seed=42):
    by_group = defaultdict(list)
    for row in labels:
        by_group[row["group_id"]].append(row)
    quarantine, testing, training = [], [], []
    for group in by_group.values():
        has_eval = any(r["source_tag"] == "eval" for r in group)
        for row in group:
            if row["source_tag"] == "eval":
                testing.append(row)
            elif row["source_tag"] != "train":
                raise ValueError("unsupported source tag")
            elif has_eval:
                quarantine.append(row)
            else:
                training.append(row)
    if len(training) != sum(POOL_SIZES.values()):
        raise ValueError("source population differs from fixed pool capacity plan")
    strata = sorted({(r["domain"], r["difficulty"]) for r in training})
    names = list(POOL_SIZES)
    cells = defaultdict(list)
    for row in training:
        cells[row["domain"], row["difficulty"]].append(row)
    preset = defaultdict(list)
    for gid, group in sorted(by_group.items()):
        if len(group) > 1 and all(r in training for r in group):
            # A seeded weighted assignment keeps every repeated instruction atomic.
            draw = int(stable_rank(seed, "duplicate-pool", gid), 16) % len(training)
            for name, size in POOL_SIZES.items():
                if draw < size:
                    preset[name].extend(group)
                    break
                draw -= size
    lower = [
        [sum((r["domain"], r["difficulty"]) == key for r in preset[name]) for name in names]
        for key in strata
    ]
    quotas = bounded_transport(
        [len(cells[key]) for key in strata],
        list(POOL_SIZES.values()),
        [
            [len(cells[key]) * size / len(training) for size in POOL_SIZES.values()]
            for key in strata
        ],
        lower=lower,
    )
    pools = {name: list(preset[name]) for name in names}
    assigned = {r["task_id"] for group in preset.values() for r in group}
    for i, key in enumerate(strata):
        available = sorted(
            (r for r in cells[key] if r["task_id"] not in assigned),
            key=lambda r: stable_rank(seed, "pool-cell", r["task_id"]),
        )
        cursor = 0
        for j, name in enumerate(names):
            count = quotas[i][j] - lower[i][j]
            pools[name].extend(available[cursor : cursor + count])
            cursor += count
        assert cursor == len(available)
    pools.update(evaluation=testing, quarantine=quarantine)
    return {
        name: sorted(rows, key=lambda r: stable_rank(seed, "pool-order:" + name, r["group_id"]))
        for name, rows in pools.items()
    }


def select_tasks(rows, count, *, seed=42, namespace="initial", ratios=None):
    """Match natural category margins and requested difficulty margins together."""
    if count < 1 or count > len(rows):
        raise ValueError("requested task count is outside the available pool")
    domains = sorted({r["domain"] for r in rows})
    domain_targets = proportional_counts(Counter(r["domain"] for r in rows), count)
    level_weights = (
        dict(zip(LEVELS, ratios)) if ratios is not None else Counter(r["difficulty"] for r in rows)
    )
    level_targets = proportional_counts({k: level_weights.get(k, 0) for k in LEVELS}, count)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    eligible = [r for group in groups.values() if len(group) == 1 for r in group]
    preset = []
    for gid, group in sorted(
        groups.items(), key=lambda item: stable_rank(seed, namespace, item[0])
    ):
        if len(eligible) + len(preset) >= count:
            break
        if len(group) > 1 and len(preset) + len(group) <= count:
            preset.extend(group)
    if len(eligible) + len(preset) < count:
        raise ValueError("requested count would split a repeated-instruction group")
    cells = defaultdict(list)
    for row in [*eligible, *preset]:
        cells[row["domain"], row["difficulty"]].append(row)
    ideals = fit_marginals(
        [
            [sum(r["domain"] == d and r["difficulty"] == k for r in rows) for k in LEVELS]
            for d in domains
        ],
        [domain_targets[d] for d in domains],
        [level_targets[k] for k in LEVELS],
    )
    lower = [
        [sum(r["domain"] == d and r["difficulty"] == k for r in preset) for k in LEVELS]
        for d in domains
    ]
    quotas = bounded_transport(
        [domain_targets[d] for d in domains],
        [level_targets[k] for k in LEVELS],
        ideals,
        capacities=[[len(cells[d, k]) for k in LEVELS] for d in domains],
        lower=lower,
    )
    selected = list(preset)
    preset_ids = {r["task_id"] for r in preset}
    for i, d in enumerate(domains):
        for j, level in enumerate(LEVELS):
            available = sorted(
                (r for r in cells[d, level] if r["task_id"] not in preset_ids),
                key=lambda r: stable_rank(seed, namespace, r["task_id"]),
            )
            selected.extend(available[: quotas[i][j] - lower[i][j]])
    assert len(selected) == count
    return sorted(selected, key=lambda r: stable_rank(seed, namespace + ":order", r["group_id"]))
