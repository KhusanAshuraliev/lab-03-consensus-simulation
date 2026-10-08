"""Command-line demonstrations: python -m consensus [all|pow|pos|gossip|bonus]."""

from __future__ import annotations

import argparse
import asyncio

from .doublespend import attacker_success, confirmations_needed, simulate_attack
from .network import SimulationResult, run_partition_scenario
from .pos import (
    GWEI,
    Attestation,
    BeaconState,
    Checkpoint,
    honest_attestations,
    inactivity_leak_trace,
    slashing_loss,
)
from .pow import measure_hash_rate, retarget, simulate_mining

LINE = "=" * 78


def _title(text: str) -> None:
    print(LINE)
    print(text)
    print(LINE)


# --------------------------------------------------------------------------- #
# Part A
# --------------------------------------------------------------------------- #


def histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    """A text histogram with equal-width bins from 0 to the maximum."""
    top = max(values)
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, int(value / top * bins))] += 1
    lines = []
    for i, count in enumerate(counts):
        bar = "#" * round(count / max(counts) * width)
        lines.append(f"  {i * top / bins:7.3f}-{(i + 1) * top / bins:7.3f} s {count:>4}  {bar}")
    return "\n".join(lines)


def demo_pow(
    blocks: int = 30,
    workers: int | None = None,
    difficulty: int = 1,
    target_seconds: float = 0.30,
) -> None:
    _title("PART A -- proof of work: retargeting, mining, interval statistics")

    print("Retarget arithmetic:")
    print(f"  retarget(1000,   300, 600) = {retarget(1000, 300, 600)}   window twice too fast")
    print(f"  retarget(1000,  1200, 600) = {retarget(1000, 1200, 600)}    window twice too slow")
    print(f"  retarget(1000,     6, 600) = {retarget(1000, 6, 600)}   clamped at x4")
    print(f"  retarget(1000, 60000, 600) = {retarget(1000, 60000, 600)}    clamped at /4")
    print(f"  retarget(1, 10000, 1)      = {retarget(1, 10000, 1)}      never below 1")
    print()

    rate = measure_hash_rate()
    label = "1 process" if workers == 1 else f"pool of {workers or 'cpu-1'} processes"
    print(f"Hash rate of one process: {rate:,.0f} hashes/s")
    print(f"Mining {blocks} blocks at fixed difficulty {difficulty} ({label})...")
    chain, stats = simulate_mining(
        blocks, retarget_interval=None, initial_difficulty=difficulty, workers=workers
    )
    print()
    print(stats.report())
    print(f"chain height        : {chain.height}")
    print()
    print("Histogram of block intervals:")
    print(histogram(stats.intervals))
    print()

    print(f"Retargeting every 10 blocks towards {target_seconds:.2f} s per block ({label})...")
    _chain, stats = simulate_mining(blocks, target_seconds=target_seconds, workers=workers)
    for number, (window_difficulty, seconds) in enumerate(stats.windows, start=1):
        print(
            f"  window {number:>2}: difficulty {window_difficulty:>3}  "
            f"took {seconds:6.2f} s (goal {target_seconds * 10:.2f} s)"
        )
    print(f"  difficulty after the last window: {stats.difficulties[-1]}")
    print()


# --------------------------------------------------------------------------- #
# Part B
# --------------------------------------------------------------------------- #


def demo_pos() -> None:
    _title("PART B -- proof of stake: Casper FFG, slashing, inactivity leak")

    print("Scenario 1: finality with all validators honest")
    state = BeaconState(validator_count=64)
    previous = state.genesis_checkpoint
    for epoch in range(1, 4):
        target = Checkpoint(epoch, f"root-{epoch}")
        state.attest_many(honest_attestations(state, previous, target))
        result = state.process_epoch(epoch)
        print(f"  epoch {epoch}: justified={result['justified']}  finalized={result['finalized']}")
        previous = target
    print()

    genesis = Checkpoint(0, "genesis")
    chain_a, chain_b = Checkpoint(1, "chain-A"), Checkpoint(1, "chain-B")

    print("Scenario 2: one validator double-votes")
    state = BeaconState(validator_count=64)
    state.attest_many(honest_attestations(state, genesis, chain_a))
    state.attest(Attestation(63, genesis, chain_b))
    before = state.validators[63].balance
    state.process_epoch(1)
    after = state.validators[63].balance
    print(f"  balance : {before / GWEI:.4f} -> {after / GWEI:.4f} ETH")
    print(f"  loss    : {(before - after) / GWEI:.4f} ETH")
    print()

    print("Scenario 3: twenty validators double-vote together")
    state = BeaconState(validator_count=64)
    state.attest_many(honest_attestations(state, genesis, chain_a))
    for index in range(20):
        state.attest(Attestation(index, genesis, chain_b))
    before = state.validators[0].balance
    state.process_epoch(1)
    after = state.validators[0].balance
    print(f"  balance : {before / GWEI:.4f} -> {after / GWEI:.4f} ETH")
    print(f"  loss    : {(before - after) / GWEI:.4f} ETH each")
    print()

    print("Scenario 4: a surround vote")
    state = BeaconState(validator_count=32)
    state.attest(Attestation(5, Checkpoint(0, "g"), Checkpoint(6, "far")))
    state.attest(Attestation(5, Checkpoint(2, "b"), Checkpoint(4, "near")))
    state.process_epoch(6)
    print(f"  slashed : {state.validators[5].slashed}")
    print(f"  rule    : {state.slashings[0].rule}")
    print()

    print("Loss per validator against simultaneous offenders (64 validators x 32 ETH):")
    counts = (1, 5, 10, 20, 21, 22, 30)
    print("  offenders  " + "".join(f"{k:>8}" for k in counts))
    print("  loss (ETH) " + "".join(f"{slashing_loss(k) / GWEI:>8.2f}" for k in counts))
    print()

    print("Scenario 5: inactivity leak, 40 of 64 validators offline for 100 epochs")
    rows = inactivity_leak_trace(epochs=100)
    shown = {1, 4, 5, 11, 20, 40, 60, 80, 81, 82, 83, 100}
    for row in rows:
        if row["epoch"] in shown:
            print(
                f"  epoch {row['epoch']:>3}: voting share={row['voting_share']:6.1%}  "
                f"leaked={row['leaked_gwei'] / GWEI:6.3f} ETH  "
                f"justified={str(row['justified']):<16} finalized={row['finalized']}"
            )
    print(f"  absent validator : {rows[-1]['absent_balance_gwei'] / GWEI:.4f} ETH at the end")
    print(f"  present validator: {rows[-1]['present_balance_gwei'] / GWEI:.4f} ETH at the end")
    print()


# --------------------------------------------------------------------------- #
# Part C
# --------------------------------------------------------------------------- #


async def _run_gossip(node_count: int, mean_block_time: float, seed: int) -> SimulationResult:
    return await run_partition_scenario(node_count, mean_block_time=mean_block_time, seed=seed)


def demo_gossip(node_count: int = 8, seed: int = 7) -> SimulationResult:
    _title("PART C -- gossip network: partition, heal, reorganisation")
    print(f"{node_count} nodes, ring plus random edges, seed {seed}.")
    result = asyncio.run(_run_gossip(node_count, mean_block_time=0.25, seed=seed))
    print(f"miners: {result.miners}")
    print()
    for event in result.events:
        print(f"  {event}")
    print()
    print(result.report())
    print()
    if result.lighter_side:
        print(
            f"at the heal: fork at height {result.fork_height}, side heights {result.split_heights}, "
            f"lighter side {result.lighter_side} holds {result.private_branch_length} private block(s)"
        )
    else:
        print(f"at the heal: side heights {result.split_heights}, equal work on both sides")
    print("reorganisations at or after the heal, per node:")
    for nid, depths in result.heal_reorgs.items():
        side = "A" if nid in result.group_a else "B"
        print(f"  {nid:<7} side {side}: {depths}")
    returned = sum(node.stats.txs_returned for node in result.nodes.values())
    print()
    print(f"reorganisations observed across all nodes : {sum(n.stats.reorgs for n in result.nodes.values())}")
    print(f"transactions returned to a mempool        : {returned}")
    print(
        f"split-time transactions in the final chain: "
        f"{result.split_txs_confirmed} of {result.split_txs_total}"
    )
    print()
    return result


# --------------------------------------------------------------------------- #
# Bonus
# --------------------------------------------------------------------------- #


def demo_bonus(trials: int = 200_000) -> None:
    _title("BONUS E1 -- double-spend probability (Bitcoin whitepaper, section 11)")
    print("   q    z   formula      Monte Carlo   ratio")
    for q, z in ((0.1, 5), (0.1, 6), (0.3, 5)):
        exact = attacker_success(q, z)
        runs = trials * (25 if q == 0.1 else 1)  # rare events need more trials
        estimate = simulate_attack(q, z, trials=runs)
        print(f"  {q:.2f} {z:>3}   {exact:.7f}    {estimate:.7f}    {estimate / exact:.3f}   ({runs:,} trials)")
    print()
    print("Confirmations needed for P < 0.1 %:")
    for q in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        print(f"  q = {q:.2f} -> z = {confirmations_needed(q)}")
    print()


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="consensus", description="Lab 03 demonstrations")
    parser.add_argument(
        "part", nargs="?", default="all", choices=["all", "pow", "pos", "gossip", "bonus"]
    )
    parser.add_argument("--blocks", type=int, default=30, help="blocks to mine in Part A")
    parser.add_argument("--workers", type=int, default=None, help="mining processes in Part A")
    parser.add_argument("--difficulty", type=int, default=1, help="fixed difficulty in Part A")
    parser.add_argument("--target", type=float, default=0.30, help="retarget goal, s per block")
    parser.add_argument("--nodes", type=int, default=8, help="nodes in Part C")
    parser.add_argument("--seed", type=int, default=7, help="seed in Part C")
    args = parser.parse_args(argv)

    if args.part in ("all", "pow"):
        demo_pow(args.blocks, args.workers, args.difficulty, args.target)
    if args.part in ("all", "pos"):
        demo_pos()
    if args.part in ("all", "gossip"):
        demo_gossip(args.nodes, args.seed)
    if args.part == "bonus":
        demo_bonus()
    return 0
