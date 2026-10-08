"""Tests for Lab 03.

These are the specification for the `[STUDENT TASK]` implementation.
"""

from __future__ import annotations

import asyncio

import pytest

from consensus.chain import (
    Block,
    Chain,
    State,
    Transaction,
    genesis_block,
    make_block,
)
from consensus.network import Message, Network, Node, build_network
from consensus.pos import (
    MAX_EFFECTIVE_BALANCE,
    Attestation,
    BeaconState,
    Checkpoint,
    find_violations,
    honest_attestations,
    is_double_vote,
    is_surround_vote,
)
from consensus.pow import mine, mine_range, retarget, target_from_difficulty

# --------------------------------------------------------------------------- #
# State transition
# --------------------------------------------------------------------------- #


def test_state_applies_a_valid_transaction() -> None:
    state = State({"a": 100})
    state.apply(Transaction("a", "b", 30, nonce=0))
    assert state.balance_of("a") == 70
    assert state.balance_of("b") == 30
    assert state.nonce_of("a") == 1


def test_insufficient_balance_is_rejected() -> None:
    state = State({"a": 10})
    assert state.validate(Transaction("a", "b", 30, nonce=0)) == "insufficient balance"


def test_fee_is_deducted_from_the_sender_and_not_credited_to_the_recipient() -> None:
    state = State({"a": 100})
    state.apply(Transaction("a", "b", 30, nonce=0, fee=5))
    assert state.balance_of("a") == 65
    assert state.balance_of("b") == 30


def test_nonce_must_be_sequential() -> None:
    """Replay protection and strict ordering, the dual role of the nonce."""
    state = State({"a": 100})
    assert state.validate(Transaction("a", "b", 1, nonce=1)) is not None
    state.apply(Transaction("a", "b", 1, nonce=0))
    assert state.validate(Transaction("a", "b", 1, nonce=0)) is not None  # replay
    assert state.validate(Transaction("a", "b", 1, nonce=1)) is None


def test_state_transition_is_deterministic() -> None:
    """Same start plus same ordering must give the same end state."""
    txs = [Transaction("a", "b", 5, nonce=i) for i in range(5)]
    finals = []
    for _ in range(3):
        state = State({"a": 100})
        for tx in txs:
            state.apply(tx)
        finals.append((state.balances["a"], state.balances["b"], state.nonces["a"]))
    assert len(set(finals)) == 1


# --------------------------------------------------------------------------- #
# Chain and fork choice
# --------------------------------------------------------------------------- #


def test_block_hash_covers_the_transaction_set() -> None:
    base = make_block(Chain(), timestamp=1.0)
    with_tx = make_block(Chain(), txs=(Transaction("a", "b", 1, 0),), timestamp=1.0)
    assert base.hash != with_tx.hash


def test_unknown_parent_is_rejected() -> None:
    chain = Chain()
    orphan = Block(height=1, prev_hash="ff" * 32, timestamp=1.0)
    with pytest.raises(KeyError):
        chain.add_block(orphan, check_pow=False)


def test_heaviest_chain_wins_not_the_first_seen() -> None:
    """A fork is resolved by cumulative work, and the loser is kept, not deleted."""
    chain = Chain()
    a = make_block(chain, miner="A", timestamp=1.0)
    chain.add_block(a, check_pow=False)

    # A competing block at the same height.
    b = Block(height=1, prev_hash=chain.genesis_hash, timestamp=2.0, miner="B")
    moved = chain.add_block(b, check_pow=False)
    assert not moved, "equal work must not displace the incumbent tip"
    assert chain.has(b.hash), "the losing branch is still stored"

    # Extending the losing branch makes it heavier, forcing a reorganisation.
    b2 = Block(height=2, prev_hash=b.hash, timestamp=3.0, miner="B")
    assert chain.add_block(b2, check_pow=False)
    assert chain.tip_hash == b2.hash
    assert chain.reorg_count == 1


def test_reorg_depth_is_recorded() -> None:
    chain = Chain()
    a1 = make_block(chain, miner="A", timestamp=1.0)
    chain.add_block(a1, check_pow=False)
    a2 = make_block(chain, miner="A", timestamp=2.0)
    chain.add_block(a2, check_pow=False)

    b1 = Block(height=1, prev_hash=chain.genesis_hash, timestamp=1.5, miner="B")
    chain.add_block(b1, check_pow=False)
    b2 = Block(height=2, prev_hash=b1.hash, timestamp=2.5, miner="B")
    chain.add_block(b2, check_pow=False)
    b3 = Block(height=3, prev_hash=b2.hash, timestamp=3.5, miner="B")
    chain.add_block(b3, check_pow=False)

    assert chain.tip_hash == b3.hash
    assert chain.reorg_depths[-1] == 2


def test_replay_state_is_reproducible() -> None:
    chain = Chain()
    tx = Transaction("user0", "user1", 10, nonce=0)
    block = make_block(chain, txs=(tx,), miner="node0", timestamp=1.0)
    chain.add_block(block, check_pow=False)

    a = chain.replay_state({"user0": 100}, reward=50)
    b = chain.replay_state({"user0": 100}, reward=50)
    assert a.balances == b.balances
    assert a.balance_of("user1") == 10
    assert a.balance_of("node0") == 50


# --------------------------------------------------------------------------- #
# Proof of Work
# --------------------------------------------------------------------------- #


def test_higher_difficulty_means_a_lower_target() -> None:
    assert target_from_difficulty(2) < target_from_difficulty(1)


def test_mining_produces_a_hash_below_target() -> None:
    block = mine(make_block(Chain(), miner="m", timestamp=1.0))
    assert block.is_valid_pow()
    assert int.from_bytes(bytes.fromhex(block.hash), "big") < block.target


def test_tampering_invalidates_the_proof_of_work() -> None:
    """Changing any header field breaks the nonce that was found for it."""
    block = mine(make_block(Chain(), miner="m", timestamp=1.0))
    tampered = Block(
        height=block.height,
        prev_hash=block.prev_hash,
        timestamp=block.timestamp,
        txs=(Transaction("a", "b", 1, 0),),
        difficulty=block.difficulty,
        nonce=block.nonce,
        miner=block.miner,
    )
    assert not tampered.is_valid_pow()


def test_chain_rejects_a_block_with_invalid_pow() -> None:
    chain = Chain()
    block = make_block(chain, miner="m", timestamp=1.0)  # never mined
    with pytest.raises(ValueError, match="proof of work"):
        chain.add_block(block, check_pow=True)


def test_strided_search_covers_the_space() -> None:
    """Two workers on alternating nonces must jointly find the solution."""
    block = make_block(Chain(), miner="m", timestamp=1.0)
    even = mine_range(block, 0, 200_000, 2)
    odd = mine_range(block, 1, 200_000, 2)
    assert even is not None or odd is not None


@pytest.mark.parametrize(
    ("actual", "expected_direction"),
    [(300.0, "up"), (1200.0, "down"), (600.0, "same")],
)
def test_retarget_direction(actual: float, expected_direction: str) -> None:
    new = retarget(1000, actual, 600.0)
    if expected_direction == "up":
        assert new > 1000
    elif expected_direction == "down":
        assert new < 1000
    else:
        assert new == 1000


def test_retarget_is_clamped_in_both_directions() -> None:
    """A single window may not move difficulty by more than a factor of four."""
    assert retarget(1000, 1.0, 600.0) == 4000
    assert retarget(1000, 100_000.0, 600.0) == 250


def test_retarget_never_falls_below_one() -> None:
    assert retarget(1, 10_000.0, 1.0) >= 1


# --------------------------------------------------------------------------- #
# Proof of Stake -- slashing rules
# --------------------------------------------------------------------------- #


def test_double_vote_is_detected() -> None:
    src = Checkpoint(0, "g")
    a = Attestation(1, src, Checkpoint(1, "A"))
    b = Attestation(1, src, Checkpoint(1, "B"))
    assert is_double_vote(a, b)


def test_same_target_root_is_not_a_double_vote() -> None:
    src = Checkpoint(0, "g")
    a = Attestation(1, src, Checkpoint(1, "A"))
    assert not is_double_vote(a, Attestation(1, src, Checkpoint(1, "A")))


def test_different_validators_are_never_slashable_together() -> None:
    src = Checkpoint(0, "g")
    a = Attestation(1, src, Checkpoint(1, "A"))
    b = Attestation(2, src, Checkpoint(1, "B"))
    assert not is_double_vote(a, b)
    assert not is_surround_vote(a, b)


def test_surround_vote_is_detected() -> None:
    wide = Attestation(3, Checkpoint(0, "g"), Checkpoint(6, "far"))
    inner = Attestation(3, Checkpoint(2, "b"), Checkpoint(4, "near"))
    assert is_surround_vote(wide, inner)
    assert not is_surround_vote(inner, wide)


def test_nested_but_not_surrounding_is_not_slashable() -> None:
    a = Attestation(3, Checkpoint(0, "g"), Checkpoint(4, "x"))
    b = Attestation(3, Checkpoint(4, "x"), Checkpoint(6, "y"))
    assert not is_surround_vote(a, b)
    assert not is_surround_vote(b, a)


def test_find_violations_reports_the_rule() -> None:
    src = Checkpoint(0, "g")
    violations = find_violations(
        [Attestation(1, src, Checkpoint(1, "A")), Attestation(1, src, Checkpoint(1, "B"))]
    )
    assert len(violations) == 1
    assert violations[0][2] == "double_vote"


# --------------------------------------------------------------------------- #
# Proof of Stake -- Casper FFG
# --------------------------------------------------------------------------- #


def test_supermajority_threshold_is_two_thirds() -> None:
    state = BeaconState(validator_count=3)
    assert state.total_active_balance == 3 * MAX_EFFECTIVE_BALANCE
    assert state.supermajority_threshold() == 2 * MAX_EFFECTIVE_BALANCE


def test_justification_requires_more_than_two_thirds() -> None:
    state = BeaconState(validator_count=9)
    target = Checkpoint(1, "r1")
    # Exactly six of nine is two thirds, which is NOT more than two thirds.
    state.attest_many(
        honest_attestations(state, state.genesis_checkpoint, target, validators=list(range(6)))
    )
    assert state.process_epoch(1)["justified"] is None

    state.attest(Attestation(6, state.genesis_checkpoint, target))
    assert state.process_epoch(1)["justified"] == target


def test_finalisation_needs_two_consecutive_justified_checkpoints() -> None:
    state = BeaconState(validator_count=32)
    c1, c2 = Checkpoint(1, "r1"), Checkpoint(2, "r2")

    state.attest_many(honest_attestations(state, state.genesis_checkpoint, c1))
    result = state.process_epoch(1)
    assert result["justified"] == c1
    assert result["finalized"] is None, "one justified checkpoint is not enough"

    state.attest_many(honest_attestations(state, c1, c2))
    result = state.process_epoch(2)
    assert result["finalized"] == c1


def test_link_from_an_unjustified_source_does_not_justify() -> None:
    state = BeaconState(validator_count=32)
    bogus_source = Checkpoint(1, "never-justified")
    state.attest_many(honest_attestations(state, bogus_source, Checkpoint(2, "r2")))
    assert state.process_epoch(2)["justified"] is None


# --------------------------------------------------------------------------- #
# Proof of Stake -- penalties
# --------------------------------------------------------------------------- #


def test_lone_equivocator_loses_only_a_small_fraction() -> None:
    state = BeaconState(validator_count=64)
    src = Checkpoint(0, "g")
    state.attest(Attestation(0, src, Checkpoint(1, "A")))
    state.attest(Attestation(0, src, Checkpoint(1, "B")))

    before = state.validators[0].balance
    state.process_epoch(1)
    loss = before - state.validators[0].balance

    assert state.validators[0].slashed
    assert 0 < loss < before * 0.15


def test_correlated_slashing_punishes_coordination_far_harder() -> None:
    """The central design idea: penalty tracks systemic risk, not individual error."""

    def loss_for(attacker_count: int) -> float:
        state = BeaconState(validator_count=64)
        src = Checkpoint(0, "g")
        for i in range(attacker_count):
            state.attest(Attestation(i, src, Checkpoint(1, "A")))
            state.attest(Attestation(i, src, Checkpoint(1, "B")))
        before = state.validators[0].balance
        state.process_epoch(1)
        return (before - state.validators[0].balance) / before

    assert loss_for(1) < 0.10
    assert loss_for(22) > loss_for(1) * 5
    assert loss_for(22) > 0.5


def test_a_slashed_validator_stops_counting_towards_quorum() -> None:
    state = BeaconState(validator_count=3)
    src = Checkpoint(0, "g")
    state.attest(Attestation(0, src, Checkpoint(1, "A")))
    state.attest(Attestation(0, src, Checkpoint(1, "B")))
    state.process_epoch(1)
    assert state.total_active_balance == 2 * MAX_EFFECTIVE_BALANCE


def test_a_validator_is_slashed_only_once() -> None:
    state = BeaconState(validator_count=16)
    src = Checkpoint(0, "g")
    state.attest(Attestation(0, src, Checkpoint(1, "A")))
    state.attest(Attestation(0, src, Checkpoint(1, "B")))
    state.process_epoch(1)
    state.process_epoch(2)
    assert len([r for r in state.slashings if r.validator == 0]) == 1


def test_inactivity_leak_only_starts_after_the_grace_period() -> None:
    state = BeaconState(validator_count=64)
    present = list(range(20))  # far below the 2/3 quorum, so nothing finalises

    leaks = []
    for epoch in range(1, 10):
        target = Checkpoint(epoch, f"r{epoch}")
        source = Checkpoint(epoch - 1, f"r{epoch - 1}") if epoch > 1 else state.genesis_checkpoint
        state.attest_many(honest_attestations(state, source, target, validators=present))
        leaks.append(state.process_epoch(epoch)["inactivity_leak"])

    assert leaks[0] == 0, "no leak during the grace period"
    assert leaks[-1] > 0, "leak must engage once finality stalls"


def test_inactivity_leak_spares_participants() -> None:
    state = BeaconState(validator_count=64)
    present = list(range(20))
    for epoch in range(1, 9):
        target = Checkpoint(epoch, f"r{epoch}")
        source = Checkpoint(epoch - 1, f"r{epoch - 1}") if epoch > 1 else state.genesis_checkpoint
        state.attest_many(honest_attestations(state, source, target, validators=present))
        state.process_epoch(epoch)

    assert state.validators[0].balance == MAX_EFFECTIVE_BALANCE  # voted every epoch
    assert state.validators[63].balance < MAX_EFFECTIVE_BALANCE  # never voted


# --------------------------------------------------------------------------- #
# Gossip network
# --------------------------------------------------------------------------- #


def test_topology_is_connected() -> None:
    _network, nodes = build_network(node_count=8, seed=1)
    reachable = {"node0"}
    frontier = ["node0"]
    while frontier:
        current = frontier.pop()
        for peer in nodes[current].peers:
            if peer not in reachable:
                reachable.add(peer)
                frontier.append(peer)
    assert reachable == set(nodes)


def test_partition_blocks_delivery_both_ways() -> None:
    network, _nodes = build_network(node_count=4, seed=1)
    network.partition([{"node0", "node1"}, {"node2", "node3"}])
    assert network.can_reach("node0", "node1")
    assert not network.can_reach("node1", "node2")
    assert not network.can_reach("node2", "node1")
    network.heal()
    assert network.can_reach("node1", "node2")


async def test_transaction_floods_the_network() -> None:
    network, nodes = build_network(
        node_count=6, latency=0.001, seed=3, initial_balances={"user0": 100}
    )
    tasks = [asyncio.create_task(n.run()) for n in nodes.values()]
    try:
        await nodes["node0"].submit_transaction(Transaction("user0", "user1", 5, nonce=0))
        await asyncio.sleep(0.4)
        assert all(len(n.mempool) == 1 for n in nodes.values())
    finally:
        for n in nodes.values():
            n.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_duplicate_messages_are_suppressed() -> None:
    """Without deduplication, gossip flooding would never terminate."""
    network, nodes = build_network(
        node_count=5, latency=0.001, seed=4, initial_balances={"user0": 100}
    )
    tasks = [asyncio.create_task(n.run()) for n in nodes.values()]
    try:
        await nodes["node0"].submit_transaction(Transaction("user0", "user1", 5, nonce=0))
        await asyncio.sleep(0.4)
        assert sum(n.stats.duplicates_suppressed for n in nodes.values()) > 0
    finally:
        for n in nodes.values():
            n.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_invalid_transaction_is_not_relayed() -> None:
    network, nodes = build_network(
        node_count=4, latency=0.001, seed=5, initial_balances={"user0": 10}
    )
    tasks = [asyncio.create_task(n.run()) for n in nodes.values()]
    try:
        accepted = await nodes["node0"].submit_transaction(
            Transaction("user0", "user1", 1_000_000, nonce=0)
        )
        await asyncio.sleep(0.2)
        assert not accepted
        assert all(len(n.mempool) == 0 for n in nodes.values())
    finally:
        for n in nodes.values():
            n.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_partition_causes_divergence_and_healing_restores_consensus() -> None:
    """The core scenario of Part C, run deterministically and asserted."""
    network, nodes = build_network(
        node_count=6, latency=0.002, seed=11, initial_balances={"user0": 1000}
    )
    tasks = [asyncio.create_task(n.run()) for n in nodes.values()]

    async def announcer() -> None:
        while True:
            await asyncio.sleep(0.05)
            for n in nodes.values():
                await n.announce_head()

    tasks.append(asyncio.create_task(announcer()))

    try:
        group_a = {"node0", "node1", "node2"}
        group_b = {"node3", "node4", "node5"}
        network.partition([group_a, group_b])

        # Each side extends its own chain while isolated.
        for label, group, miner in (("A", group_a, "node0"), ("B", group_b, "node3")):
            node = nodes[miner]
            for _ in range(3 if label == "A" else 1):
                block = make_block(node.chain, miner=miner, timestamp=len(node.chain.entries))
                node.chain.add_block(block, check_pow=False)
                await node.broadcast(Message("block", block, miner))
        await asyncio.sleep(0.3)

        tips_during = {n.tip_hash for n in nodes.values()}
        assert len(tips_during) > 1, "the two sides must diverge while partitioned"
        assert nodes["node0"].height == 3
        assert nodes["node3"].height == 1

        network.heal()
        await asyncio.sleep(0.6)

        tips_after = {n.tip_hash for n in nodes.values()}
        assert len(tips_after) == 1, "healing must restore a single canonical tip"
        assert all(n.height == 3 for n in nodes.values()), "the heavier branch must win"
        # The minority side had to discard its block: that is a reorganisation.
        assert nodes["node3"].stats.reorgs >= 1
    finally:
        for n in nodes.values():
            n.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
