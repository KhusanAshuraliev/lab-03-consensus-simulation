"""Run the measurements behind my report and store the raw numbers as JSON.

    uv run python experiments/collect.py pow        # Part A (several minutes)
    uv run python experiments/collect.py pool       # bonus E2 (several minutes)
    uv run python experiments/collect.py pos        # Part B
    uv run python experiments/collect.py gossip     # Part C
    uv run python experiments/collect.py doublespend  # bonus E1

Timing experiments (pow, pool) should run one at a time on an otherwise idle
machine. Results go to evidence/data/.
"""

from __future__ import annotations

import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consensus.doublespend import attacker_success, confirmations_needed, simulate_attack  # noqa: E402
from consensus.network import run_partition_scenario  # noqa: E402
from consensus.pos import GWEI, inactivity_leak_trace, slashing_loss  # noqa: E402
from consensus.pow import MinerPool, measure_hash_rate, simulate_mining  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "evidence" / "data"


def save(name: str, payload: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {path.relative_to(DATA.parent.parent)}")


def summary(stats) -> dict:
    return {
        "blocks": len(stats.intervals),
        "mean": stats.mean_interval,
        "median": stats.median_interval,
        "stdev": stats.stdev_interval,
        "min": min(stats.intervals),
        "max": max(stats.intervals),
        "median_over_mean": stats.median_over_mean,
        "stdev_over_mean": stats.stdev_over_mean,
    }


def machine() -> dict:
    import multiprocessing

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "hash_rate_one_process": measure_hash_rate(300_000),
    }


# --------------------------------------------------------------------------- #


def collect_pow() -> None:
    info = machine()
    print(f"hash rate: {info['hash_rate_one_process']:,.0f} H/s")

    difficulty = 8
    _chain, stats = simulate_mining(
        200, retarget_interval=None, initial_difficulty=difficulty, workers=1
    )
    print(stats.report())
    save(
        "pow_intervals",
        {
            "machine": info,
            "workers": 1,
            "difficulty": difficulty,
            "summary": summary(stats),
            "intervals": stats.intervals,
        },
    )

    target_seconds = 1.0
    _chain, stats = simulate_mining(200, target_seconds=target_seconds, workers=1)
    for number, (window_difficulty, seconds) in enumerate(stats.windows, start=1):
        print(f"window {number:>2}: difficulty {window_difficulty:>3} took {seconds:6.2f} s")
    save(
        "pow_retarget",
        {
            "machine": info,
            "workers": 1,
            "target_seconds": target_seconds,
            "retarget_interval": 10,
            "windows": [{"difficulty": d, "seconds": s} for d, s in stats.windows],
            "final_difficulty": stats.difficulties[-1],
            "equilibrium_estimate": info["hash_rate_one_process"] * target_seconds / 65536,
        },
    )


def collect_pool() -> None:
    info = machine()
    difficulty = 8
    blocks = 600
    runs = {}
    for workers in (1, 2, 4):
        started = time.perf_counter()
        # A pool also for one worker, so that all three runs use the same code path.
        with MinerPool(workers) as pool:
            _chain, stats = simulate_mining(
                blocks, retarget_interval=None, initial_difficulty=difficulty, pool=pool
            )
        runs[workers] = summary(stats) | {"intervals": stats.intervals}
        print(
            f"{workers} worker(s): mean {stats.mean_interval:.4f} s  "
            f"median/mean {stats.median_over_mean:.3f}  stdev/mean {stats.stdev_over_mean:.3f}"
            f"  ({time.perf_counter() - started:.0f} s)"
        )
    base = runs[1]["mean"]
    for workers in runs:
        runs[workers]["speedup"] = base / runs[workers]["mean"]
    save("pool_scaling", {"machine": info, "difficulty": difficulty, "blocks": blocks, "runs": runs})


def collect_pos() -> None:
    table = {k: slashing_loss(k) / GWEI for k in range(1, 33)}
    for k in (1, 5, 10, 20, 21, 22, 30):
        print(f"{k:>2} offenders: {table[k]:.4f} ETH")
    rows = inactivity_leak_trace(epochs=100)
    for row in rows:
        if row["epoch"] in (1, 5, 11, 20, 40, 60, 80, 81, 82, 83, 100):
            print(
                f"epoch {row['epoch']:>3}: share {row['voting_share']:.4f} "
                f"justified={row['justified']} finalized={row['finalized']}"
            )
    save("pos", {"loss_eth_by_offenders": table, "leak_trace": rows})


def collect_gossip() -> None:
    def describe(result) -> dict:
        return {
            "events": result.events,
            "miners": result.miners,
            "group_a": result.group_a,
            "group_b": result.group_b,
            "split_heights": result.split_heights,
            "fork_height": result.fork_height,
            "lighter_side": result.lighter_side,
            "private_branch_length": result.private_branch_length,
            "all_reorg_depths": result.reorg_depths(),
            "heal_reorg_depths": result.heal_reorgs,
            "final_distinct_tips": result.distinct_tips(),
            "final_heights": sorted(set(result.heights().values())),
            "txs_returned": sum(n.stats.txs_returned for n in result.nodes.values()),
            "split_txs_confirmed": result.split_txs_confirmed,
            "delivered": result.network.delivered,
            "dropped": result.network.dropped,
            "report": result.report(),
        }

    runs = {}
    for seed in (1, 7, 42):
        result = asyncio.run(run_partition_scenario(12, mean_block_time=0.25, seed=seed))
        runs[seed] = describe(result)
        print(f"--- seed {seed}")
        for event in result.events:
            print("  " + event)
        print(
            f"  fork at {result.fork_height}, sides {result.split_heights}, lighter "
            f"{result.lighter_side or 'tie'}, private branch {result.private_branch_length}"
        )

    # How deep is the heal-time reorganisation as the partition lasts longer?
    sweep = {}
    for split_seconds in (0.5, 1.0, 2.0, 4.0, 8.0):
        depths = []
        for seed in range(100, 110):
            result = asyncio.run(
                run_partition_scenario(
                    12,
                    mean_block_time=0.25,
                    seed=seed,
                    connected_seconds=0.8,
                    split_seconds=split_seconds,
                    healed_seconds=1.2,
                )
            )
            depths.append(max(max(d, default=0) for d in result.heal_reorgs.values()))
        sweep[split_seconds] = {"depths": depths, "mean": statistics.fmean(depths)}
        print(f"split {split_seconds:>4} s: depths {depths} mean {statistics.fmean(depths):.1f}")
    save("gossip", {"runs": runs, "duration_sweep": sweep})


def collect_doublespend() -> None:
    rows = []
    for q, z, trials in ((0.1, 5, 5_000_000), (0.1, 6, 5_000_000), (0.3, 5, 500_000)):
        exact = attacker_success(q, z)
        estimate = simulate_attack(q, z, trials=trials)
        rows.append({"q": q, "z": z, "formula": exact, "monte_carlo": estimate, "trials": trials})
        print(f"q={q} z={z}: formula {exact:.7f}  Monte Carlo {estimate:.7f}  ratio {estimate / exact:.3f}")
    needed = {q: confirmations_needed(q) for q in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)}
    print(needed)
    save("doublespend", {"race": rows, "confirmations_needed": needed})


if __name__ == "__main__":
    jobs = {
        "pow": collect_pow,
        "pool": collect_pool,
        "pos": collect_pos,
        "gossip": collect_gossip,
        "doublespend": collect_doublespend,
    }
    for name in sys.argv[1:] or list(jobs):
        jobs[name]()
