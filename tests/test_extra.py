"""My own tests, on top of the course suite in test_consensus.py.

They pin down the "Done when" column of the task table where the course suite
leaves room, and cover the parts I designed differently (worker pool, orphan
pool, locator sync, transactions returning to the mempool) and bonus E1.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import random
import statistics

import pytest

from consensus.chain import (
    GENESIS_TARGET,
    Block,
    Chain,
    State,
    Transaction,
    make_block,
    merkle_root,
)
from consensus.doublespend import attacker_success, confirmations_needed, simulate_attack
from consensus.network import Message, build_network, run_partition_scenario
from consensus.pos import (
    GWEI,
    MAX_EFFECTIVE_BALANCE,
    Attestation,
    BeaconState,
    Checkpoint,
    find_violations,
    honest_attestations,
    inactivity_leak_trace,
    slashing_loss,
)
from consensus.pow import (
    MinerPool,
    MiningStats,
    difficulty_from_target,
    mine,
    mine_parallel,
    mine_range,
    retarget,
    simulate_mining,
    target_from_difficulty,
)

# --------------------------------------------------------------------------- #
# 1. Ledger
# --------------------------------------------------------------------------- #


def test_skipped_and_repeated_nonces_are_rejected_with_a_reason() -> None:
    state = State({"a": 100})
    assert "gap" in state.validate(Transaction("a", "b", 1, nonce=2))
    state.apply(Transaction("a", "b", 1, nonce=0))
    assert "already used" in state.validate(Transaction("a", "b", 1, nonce=0))
    with pytest.raises(ValueError):
        state.apply(Transaction("a", "b", 1, nonce=0))


def test_negative_amounts_and_overspending_with_fee_are_rejected() -> None:
    state = State({"a": 100})
    assert state.validate(Transaction("a", "b", -1, nonce=0)) is not None
    assert state.validate(Transaction("a", "b", 96, nonce=0, fee=5)) == "insufficient balance"
    assert state.validate(Transaction("a", "b", 95, nonce=0, fee=5)) is None


def test_miner_receives_reward_and_fees() -> None:
    chain = Chain()
    block = make_block(
        chain, txs=(Transaction("a", "b", 10, 0, fee=3),), miner="m", timestamp=1.0
    )
    chain.add_block(block, check_pow=False)
    state = chain.replay_state({"a": 100}, reward=50)
    assert state.balance_of("m") == 53
    assert sum(state.balances.values()) == 150  # only the reward is new money


def test_txid_and_merkle_root_depend_on_content_and_order() -> None:
    t1, t2 = Transaction("a", "b", 1, 0), Transaction("a", "b", 1, 1)
    assert t1.txid != t2.txid
    assert Transaction("ab", "c", 1, 0).txid != Transaction("a", "bc", 1, 0).txid
    assert merkle_root([t1.txid, t2.txid]) != merkle_root([t2.txid, t1.txid])
    assert merkle_root([]) == "00" * 32
    assert merkle_root([t1.txid]) == t1.txid


def _extend(chain: Chain, parent: Block, miner: str, difficulty: int = 1) -> Block:
    block = Block(
        height=parent.height + 1,
        prev_hash=parent.hash,
        timestamp=float(len(chain.entries)),
        difficulty=difficulty,
        miner=miner,
    )
    chain.add_block(block, check_pow=False)
    return block


def test_reorg_keeps_the_old_branch_and_reports_what_was_rolled_back() -> None:
    chain = Chain()
    a = [chain.tip]
    for _ in range(3):
        a.append(_extend(chain, a[-1], "A"))
    b = [a[1]]  # fork after the first A block
    for _ in range(2):
        b.append(_extend(chain, b[-1], "B"))
    assert chain.tip_hash == a[3].hash, "equal work must not move the tip"
    assert chain.reorg_count == 0

    b.append(_extend(chain, b[-1], "B"))
    assert chain.tip_hash == b[3].hash
    assert chain.reorg_depths == [2]
    assert [blk.hash for blk in chain.last_disconnected] == [a[2].hash, a[3].hash]
    assert all(chain.has(blk.hash) for blk in a), "the losing branch stays stored"
    assert chain.fork_point(a[3].hash, b[3].hash).hash == a[1].hash

    # The old branch can win back the tip later.
    a.append(_extend(chain, a[-1], "A"))
    assert chain.tip_hash == b[3].hash  # equal work again
    a.append(_extend(chain, a[-1], "A"))
    assert chain.tip_hash == a[5].hash
    assert chain.reorg_depths == [2, 3]


def test_extending_the_tip_is_not_a_reorganisation() -> None:
    chain = Chain()
    tip = chain.tip
    for _ in range(5):
        tip = _extend(chain, tip, "A")
    assert chain.reorg_count == 0
    assert chain.last_disconnected == []


def test_fork_choice_compares_work_not_length() -> None:
    chain = Chain()
    genesis = chain.tip
    long_tip = genesis
    for _ in range(3):
        long_tip = _extend(chain, long_tip, "A", difficulty=1)
    heavy = _extend(chain, genesis, "B", difficulty=8)
    assert chain.tip_hash == heavy.hash, "one block of difficulty 8 outweighs three of difficulty 1"
    assert chain.height == 1
    assert chain.reorg_depths == [3]


def test_wrong_height_is_rejected() -> None:
    chain = Chain()
    with pytest.raises(ValueError, match="height"):
        chain.add_block(Block(height=5, prev_hash=chain.genesis_hash, timestamp=1.0), check_pow=False)


# --------------------------------------------------------------------------- #
# 2. Proof of work
# --------------------------------------------------------------------------- #


def test_target_and_difficulty_are_inverse() -> None:
    assert target_from_difficulty(1) == GENESIS_TARGET
    for difficulty in (1, 2, 7, 1000):
        assert difficulty_from_target(target_from_difficulty(difficulty)) == difficulty


def test_retarget_table_from_the_lab_sheet() -> None:
    assert retarget(1000, 300, 600) == 2000
    assert retarget(1000, 1200, 600) == 500
    assert retarget(1000, 6, 600) == 4000
    assert retarget(1000, 60000, 600) == 250
    assert retarget(1, 60000, 600) == 1
    assert retarget(1000, 0, 600) == 4000  # a zero-length window is clamped as well


def test_every_mined_block_is_below_its_target_and_accepted_by_the_chain() -> None:
    chain = Chain()
    for i in range(10):
        block = mine(make_block(chain, miner="m", difficulty=1 + i % 3, timestamp=float(i + 1)))
        assert int(block.hash, 16) < block.target
        assert chain.add_block(block, check_pow=True)
    assert chain.height == 10


def test_strided_ranges_partition_the_nonce_space() -> None:
    block = make_block(Chain(), miner="m", timestamp=7.0)
    first = mine_range(block, 0, 400_000)
    assert first is not None
    hits = [mine_range(block, offset, 400_000, 4) for offset in range(4)]
    assert min(h for h in hits if h is not None) == first
    for offset, hit in enumerate(hits):
        assert hit is None or hit % 4 == offset
    assert mine_range(block, 0, first) is None, "`stop` is exclusive"


def test_parallel_miner_returns_a_valid_nonce_and_leaves_no_process_behind() -> None:
    block = make_block(Chain(), miner="m", timestamp=3.0)
    mined = mine_parallel(block, workers=2)
    assert mined.is_valid_pow()
    assert mp.active_children() == []


def test_pool_reuses_the_same_processes_for_every_block() -> None:
    chain = Chain()
    with MinerPool(2) as pool:
        pids = [p.pid for p in pool.processes]
        for i in range(6):
            mined = pool.mine(make_block(chain, miner="m", timestamp=float(i + 1)))
            chain.add_block(mined, check_pow=True)
        assert [p.pid for p in pool.processes] == pids
        assert all(p.is_alive() for p in pool.processes)
    assert all(not p.is_alive() for p in pool.processes)
    assert chain.height == 6


def test_simulate_mining_retargets_at_window_boundaries() -> None:
    chain, stats = simulate_mining(20, retarget_interval=5, target_seconds=1e-6, workers=1)
    assert chain.height == 20
    assert len(stats.intervals) == 20 and len(stats.windows) == 4
    # Far too slow for a microsecond goal, so difficulty stays at the floor.
    assert set(stats.difficulties) == {1}

    _chain, stats = simulate_mining(4, retarget_interval=2, target_seconds=1e6, workers=1)
    assert stats.difficulties == [1, 1, 4, 4, 16], "far too fast: x4 per window"


# --------------------------------------------------------------------------- #
# 3. Interval statistics
# --------------------------------------------------------------------------- #


def test_statistics_of_an_exponential_sample() -> None:
    rng = random.Random(5)
    stats = MiningStats([rng.expovariate(4.0) for _ in range(20_000)], [1], [])
    assert stats.median_over_mean == pytest.approx(0.693, abs=0.02)
    assert stats.stdev_over_mean == pytest.approx(1.0, abs=0.03)


def test_hash_attempts_per_block_follow_the_exponential_prediction() -> None:
    """Deterministic version of the Part A measurement.

    With fixed timestamps the winning nonce is fixed, and nonce + 1 is the
    number of attempts, which is what the block interval is proportional to.
    """
    chain = Chain()
    attempts = []
    for i in range(150):
        block = mine(make_block(chain, miner="m", timestamp=float(i + 1)))
        chain.add_block(block)
        attempts.append(block.nonce + 1)
    mean = statistics.fmean(attempts)
    assert 0.60 <= statistics.median(attempts) / mean <= 0.80
    assert 0.80 <= statistics.pstdev(attempts) / mean <= 1.20
    assert mean == pytest.approx(65_536, rel=0.25)
    assert max(attempts) / min(attempts) > 10


# --------------------------------------------------------------------------- #
# 4. Casper FFG
# --------------------------------------------------------------------------- #


def test_repeating_a_vote_does_not_add_weight() -> None:
    state = BeaconState(validator_count=9)
    target = Checkpoint(1, "r1")
    for _ in range(5):
        state.attest_many(
            honest_attestations(state, state.genesis_checkpoint, target, validators=list(range(6)))
        )
    assert state.process_epoch(1)["justified"] is None


def test_a_skipped_epoch_justifies_but_does_not_finalise() -> None:
    state = BeaconState(validator_count=32)
    c1, c3, c4 = Checkpoint(1, "r1"), Checkpoint(3, "r3"), Checkpoint(4, "r4")
    state.attest_many(honest_attestations(state, state.genesis_checkpoint, c1))
    state.process_epoch(1)

    state.attest_many(honest_attestations(state, c1, c3))
    result = state.process_epoch(3)
    assert result["justified"] == c3
    assert result["finalized"] is None, "1 and 3 are not consecutive"
    assert state.finalized == state.genesis_checkpoint

    state.attest_many(honest_attestations(state, c3, c4))
    assert state.process_epoch(4)["finalized"] == c3


def test_an_unjustified_source_changes_nothing_at_all() -> None:
    state = BeaconState(validator_count=32)
    state.attest_many(honest_attestations(state, Checkpoint(1, "x"), Checkpoint(2, "r2")))
    result = state.process_epoch(2)
    assert result["justified"] is None and result["finalized"] is None
    assert state.justified == {state.genesis_checkpoint}
    assert state.current_justified == state.genesis_checkpoint


def test_a_slashed_voter_does_not_count_towards_the_link_it_voted_for() -> None:
    state = BeaconState(validator_count=10)
    source, target = state.genesis_checkpoint, Checkpoint(1, "A")
    state.attest_many(honest_attestations(state, source, target, validators=list(range(7))))
    state.attest(Attestation(0, source, Checkpoint(1, "B")))
    result = state.process_epoch(1)
    # 6 honest votes out of 9 remaining validators is exactly two thirds.
    assert [r.validator for r in result["slashed"]] == [0]
    assert result["justified"] is None


# --------------------------------------------------------------------------- #
# 5. Slashing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("offenders", "loss_eth"),
    [(1, 2.41), (5, 8.06), (10, 15.15), (20, 29.43), (22, 32.00), (30, 32.00)],
)
def test_loss_table_of_part_b(offenders: int, loss_eth: float) -> None:
    assert slashing_loss(offenders) / GWEI == pytest.approx(loss_eth, abs=0.005)


def test_penalty_is_initial_plus_correlated() -> None:
    state = BeaconState(validator_count=64)
    source = Checkpoint(0, "g")
    state.attest(Attestation(0, source, Checkpoint(1, "A")))
    state.attest(Attestation(0, source, Checkpoint(1, "B")))
    (record,) = state.process_epoch(1)["slashed"]
    assert record.rule == "double_vote"
    assert record.initial_penalty == MAX_EFFECTIVE_BALANCE // 32
    assert record.total_penalty == MAX_EFFECTIVE_BALANCE - state.validators[0].balance
    assert state.validators[0].slashed_epoch == 1


def test_surround_vote_is_slashed_in_either_order_of_arrival() -> None:
    wide = Attestation(5, Checkpoint(0, "g"), Checkpoint(6, "far"))
    inner = Attestation(5, Checkpoint(2, "b"), Checkpoint(4, "near"))
    for first, second in ((wide, inner), (inner, wide)):
        state = BeaconState(validator_count=32)
        state.attest(first)
        state.attest(second)
        state.process_epoch(6)
        assert state.validators[5].slashed
        assert state.slashings[0].rule == "surround_vote"
        assert find_violations([first, second])[0][2] == "surround_vote"


def test_slashed_validator_is_ignored_afterwards_and_never_slashed_again() -> None:
    state = BeaconState(validator_count=16)
    source = Checkpoint(0, "g")
    for root in ("A", "B", "C"):  # three conflicting votes, three slashable pairs
        state.attest(Attestation(0, source, Checkpoint(1, root)))
    state.process_epoch(1)
    balance = state.validators[0].balance
    state.attest(Attestation(0, source, Checkpoint(2, "D")))
    state.attest(Attestation(0, source, Checkpoint(2, "E")))
    state.process_epoch(2)
    assert len(state.slashings) == 1
    assert state.validators[0].balance == balance


def test_honest_votes_produce_no_violation() -> None:
    state = BeaconState(validator_count=8)
    previous = state.genesis_checkpoint
    for epoch in range(1, 6):
        target = Checkpoint(epoch, f"r{epoch}")
        state.attest_many(honest_attestations(state, previous, target))
        assert state.process_epoch(epoch)["slashed"] == []
        previous = target
    assert find_violations(state.attestation_log) == []


# --------------------------------------------------------------------------- #
# 6. Inactivity leak
# --------------------------------------------------------------------------- #


def test_leak_waits_for_the_grace_period_and_spares_voters() -> None:
    rows = inactivity_leak_trace(epochs=12)
    assert [row["leaked_gwei"] for row in rows[:4]] == [0, 0, 0, 0]
    assert all(row["leaked_gwei"] > 0 for row in rows[4:])
    assert rows[-1]["present_balance_gwei"] == MAX_EFFECTIVE_BALANCE
    assert rows[-1]["absent_balance_gwei"] < MAX_EFFECTIVE_BALANCE
    assert rows[0]["voting_share"] == pytest.approx(0.375)


def test_no_leak_while_the_chain_finalises() -> None:
    state = BeaconState(validator_count=64)
    voters = list(range(60))  # four validators are offline, the quorum holds
    previous = state.genesis_checkpoint
    for epoch in range(1, 20):
        target = Checkpoint(epoch, f"r{epoch}")
        state.attest_many(honest_attestations(state, previous, target, validators=voters))
        assert state.process_epoch(epoch)["inactivity_leak"] == 0
        previous = target
    assert state.validators[63].balance == MAX_EFFECTIVE_BALANCE


def test_finality_resumes_once_the_voters_exceed_two_thirds() -> None:
    rows = inactivity_leak_trace(epochs=100)
    first_justified = next(row["epoch"] for row in rows if row["justified"])
    first_finalized = next(row["epoch"] for row in rows if row["finalized"])
    assert first_justified in (81, 82)
    assert first_finalized == first_justified + 1
    # The share entering the justifying epoch is the one left by the epoch before.
    assert rows[first_justified - 2]["voting_share"] > 2 / 3
    assert rows[first_justified - 3]["voting_share"] <= 2 / 3
    # Once finality is back the leak stops and the share stays put.
    assert all(row["leaked_gwei"] == 0 for row in rows[first_finalized - 1 :])
    for epoch, share in ((20, 0.44), (40, 0.51), (60, 0.59)):
        assert rows[epoch - 1]["voting_share"] == pytest.approx(share, abs=0.01)


# --------------------------------------------------------------------------- #
# 7. Gossip network
# --------------------------------------------------------------------------- #


class Running:
    """Start the node loops of a network and stop them again."""

    def __init__(self, nodes, announce_every: float | None = None) -> None:
        self.nodes = nodes
        self.announce_every = announce_every
        self.tasks: list[asyncio.Task] = []

    async def _announce(self) -> None:
        while True:
            await asyncio.sleep(self.announce_every)
            for node in self.nodes.values():
                await node.announce_head()

    async def __aenter__(self) -> Running:
        self.tasks = [asyncio.create_task(n.run()) for n in self.nodes.values()]
        if self.announce_every:
            self.tasks.append(asyncio.create_task(self._announce()))
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        for node in self.nodes.values():
            node.stop()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


@pytest.mark.parametrize("count", [2, 3, 4, 8, 12])
def test_topology_is_symmetric_and_connected_for_any_size(count: int) -> None:
    _network, nodes = build_network(node_count=count, seed=count)
    for nid, node in nodes.items():
        assert nid not in node.peers
        assert all(nid in nodes[peer].peers for peer in node.peers)
        assert len(node.peers) >= min(2, count - 1)


async def test_each_node_processes_a_transaction_and_a_block_exactly_once() -> None:
    _network, nodes = build_network(
        node_count=8, latency=0.001, seed=2, initial_balances={"user0": 100}
    )
    async with Running(nodes):
        tx = Transaction("user0", "user1", 5, nonce=0)
        await nodes["node0"].submit_transaction(tx)
        await asyncio.sleep(0.2)

        origin = nodes["node0"]
        block = make_block(origin.chain, txs=(tx,), miner="node0", timestamp=1.0)
        origin.seen.add(block.hash)
        origin.chain.add_block(block, check_pow=False)
        await origin.broadcast(Message("block", block, "node0"))
        await asyncio.sleep(0.2)

    assert all(n.stats.txs_received == 1 for n in nodes.values())
    assert all(n.stats.blocks_received == 1 for n in nodes.values() if n is not origin)
    assert origin.stats.blocks_received == 0, "echoes of its own block are duplicates"
    assert all(n.height == 1 for n in nodes.values())
    assert all(len(n.mempool) == 0 for n in nodes.values() if n is not origin)
    # Every relay beyond the first one was recognised as a duplicate.
    links = sum(len(n.peers) for n in nodes.values())
    assert sum(n.stats.duplicates_suppressed for n in nodes.values()) > links / 2


async def test_no_message_crosses_a_partition() -> None:
    network, nodes = build_network(
        node_count=6, latency=0.001, seed=9, initial_balances={"user0": 100}
    )
    side_a, side_b = {"node0", "node1", "node2"}, {"node3", "node4", "node5"}
    async with Running(nodes, announce_every=0.02):
        network.partition([side_a, side_b])
        await nodes["node0"].submit_transaction(Transaction("user0", "user1", 5, nonce=0))
        await asyncio.sleep(0.2)
        assert all(len(nodes[n].mempool) == 1 for n in side_a)
        assert all(len(nodes[n].mempool) == 0 and not nodes[n].seen for n in side_b)
        assert network.dropped > 0


async def test_a_message_in_flight_is_lost_when_the_partition_appears() -> None:
    network, nodes = build_network(node_count=2, latency=0.05, seed=1)
    async with Running(nodes):
        await nodes["node0"].submit_transaction(Transaction("user0", "user1", 5, nonce=0))
        network.partition([{"node0"}, {"node1"}])  # the message is still travelling
        await asyncio.sleep(0.2)
        assert len(nodes["node1"].mempool) == 0
        assert network.dropped == 1 and network.delivered == 0


async def test_a_block_arriving_before_its_parent_waits_in_the_orphan_pool() -> None:
    _network, nodes = build_network(node_count=3, latency=0.001, seed=1)
    source, receiver = nodes["node0"], nodes["node1"]
    blocks = []
    for i in range(3):
        block = make_block(source.chain, miner="node0", timestamp=float(i + 1))
        source.chain.add_block(block, check_pow=False)
        blocks.append(block)

    await receiver.handle(Message("block", blocks[2], "node0"))
    await receiver.handle(Message("block", blocks[1], "node0"))
    assert receiver.height == 0 and len(receiver.orphans) == 2
    await receiver.handle(Message("block", blocks[0], "node0"))
    assert receiver.height == 3 and not receiver.orphans
    assert receiver.tip_hash == source.tip_hash


async def test_a_block_with_an_invalid_transaction_is_rejected_and_not_relayed() -> None:
    _network, nodes = build_network(
        node_count=4, latency=0.001, seed=1, initial_balances={"user0": 10}
    )
    async with Running(nodes):
        receiver = nodes["node1"]
        bad = make_block(
            receiver.chain,
            txs=(Transaction("user0", "user1", 999, nonce=0),),
            miner="node0",
            timestamp=1.0,
        )
        await receiver.handle(Message("block", bad, "node0"))
        await asyncio.sleep(0.1)
    assert receiver.stats.blocks_rejected == 1
    assert all(n.height == 0 for n in nodes.values())
    assert all(n.stats.blocks_received == 0 for n in nodes.values() if n is not receiver)


def test_locator_sync_returns_only_the_missing_blocks() -> None:
    _network, nodes = build_network(node_count=2, seed=1)
    ahead, behind = nodes["node0"], nodes["node1"]
    shared = []
    for i in range(30):
        block = make_block(ahead.chain, miner="x", timestamp=float(i + 1))
        ahead.chain.add_block(block, check_pow=False)
        if i < 20:
            behind.chain.add_block(block, check_pow=False)
            shared.append(block)
    locator = behind._locator()
    assert locator[0] == behind.tip_hash and locator[-1] == behind.chain.genesis_hash
    assert len(locator) < 21, "the locator thins out towards genesis"
    missing = ahead._blocks_after(locator)
    assert [b.height for b in missing] == list(range(21, 31))


async def test_transactions_from_orphaned_blocks_return_to_the_mempool() -> None:
    network, nodes = build_network(
        node_count=6, latency=0.002, seed=11, initial_balances={"user0": 1000}
    )
    side_a, side_b = {"node0", "node1", "node2"}, {"node3", "node4", "node5"}
    tx = Transaction("user0", "user1", 25, nonce=0, fee=1)

    async with Running(nodes, announce_every=0.05):
        network.partition([side_a, side_b])

        # Side B confirms the transaction in one block.
        await nodes["node3"].submit_transaction(tx)
        await asyncio.sleep(0.1)
        miner_b = nodes["node3"]
        block_b = make_block(miner_b.chain, txs=(tx,), miner="node3", timestamp=1.0)
        miner_b.seen.add(block_b.hash)
        miner_b._connect(block_b)
        await miner_b.broadcast(Message("block", block_b, "node3"))

        # Side A, which never saw the transaction, mines three empty blocks.
        miner_a = nodes["node0"]
        for i in range(3):
            block = make_block(miner_a.chain, miner="node0", timestamp=float(i + 1))
            miner_a.seen.add(block.hash)
            miner_a._connect(block)
            await miner_a.broadcast(Message("block", block, "node0"))
        await asyncio.sleep(0.3)

        assert all(nodes[n].state().balance_of("user1") == 25 for n in side_b)
        assert all(len(nodes[n].mempool) == 0 for n in nodes)

        network.heal()
        await asyncio.sleep(0.6)

        assert len({n.tip_hash for n in nodes.values()}) == 1
        assert all(n.height == 3 for n in nodes.values())
        for name in side_b:
            node = nodes[name]
            assert node.chain.reorg_depths == [1]
            assert node.stats.txs_returned == 1
            assert node.chain.has(block_b.hash), "the orphaned block is still stored"
            assert node.state().balance_of("user1") == 0, "the payment is unconfirmed again"
        # The transaction is pending everywhere, including on the side that
        # had never heard of it.
        assert all(tx.txid in n.mempool for n in nodes.values())

        # The next block confirms it again, this time on the surviving chain.
        final = make_block(
            miner_a.chain, txs=miner_a._select_transactions(), miner="node0", timestamp=9.0
        )
        assert final.txs == (tx,)
        miner_a.seen.add(final.hash)
        miner_a._connect(final)
        await miner_a.broadcast(Message("block", final, "node0"))
        await asyncio.sleep(0.3)
        assert all(n.state().balance_of("user1") == 25 for n in nodes.values())
        assert all(len(n.mempool) == 0 for n in nodes.values())


async def test_miner_orders_one_senders_transactions_by_nonce() -> None:
    _network, nodes = build_network(node_count=2, seed=1, initial_balances={"user0": 100})
    node = nodes["node0"]
    # Arrival order 2, 1, 0 and rising fees: fee order alone would be invalid.
    for nonce in (2, 1, 0):
        assert await node.submit_transaction(Transaction("user0", "user1", 1, nonce, fee=nonce))
    assert [tx.nonce for tx in node._select_transactions()] == [0, 1, 2]
    assert not await node.submit_transaction(Transaction("user0", "user1", 1, nonce=99))


@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_full_partition_scenario_ends_in_agreement_or_an_equal_work_tie(seed: int) -> None:
    result = await run_partition_scenario(
        8,
        mean_block_time=0.08,
        seed=seed,
        connected_seconds=0.5,
        split_seconds=0.8,
        healed_seconds=1.0,
    )
    split_trace = next(e for e in result.events if e.startswith("phase 2"))
    assert "distinct tips=1 " not in split_trace, "the two sides must diverge"

    works = {n.chain.total_work for n in result.nodes.values()}
    assert len(works) == 1, "tips of different work after healing would be a bug"
    assert result.converged() or result.distinct_tips() == 2

    # One side gives up its whole private branch in a single reorganisation.
    # Normally that is the side with less work at the heal; if it mines fast
    # enough before the two chains have been exchanged it can still win, so the
    # test accepts either side.
    private = {side: result.split_heights[side] - result.fork_height for side in ("A", "B")}
    groups = {"A": result.group_a, "B": result.group_b}
    if result.lighter_side and min(private.values()) >= 1:
        assert any(
            all(max(result.heal_reorgs[n], default=0) >= private[side] for n in groups[side])
            for side in ("A", "B")
        )


# --------------------------------------------------------------------------- #
# Bonus E1
# --------------------------------------------------------------------------- #


def test_double_spend_formula_reproduces_the_whitepaper_table() -> None:
    assert attacker_success(0.1, 0) == pytest.approx(1.0)
    assert attacker_success(0.1, 5) == pytest.approx(0.0009137, abs=5e-8)
    assert attacker_success(0.1, 6) == pytest.approx(0.0002428, abs=5e-8)
    assert attacker_success(0.3, 5) == pytest.approx(0.1773523, abs=5e-8)
    assert attacker_success(0.3, 50) == pytest.approx(0.0000006, abs=5e-8)


def test_confirmations_needed_table() -> None:
    expected = {0.10: 5, 0.15: 8, 0.20: 11, 0.25: 15, 0.30: 24, 0.35: 41, 0.40: 89, 0.45: 340}
    assert {q: confirmations_needed(q) for q in expected} == expected


def test_monte_carlo_agrees_with_the_formula() -> None:
    for q, z in ((0.3, 5), (0.2, 3), (0.1, 2)):
        estimate = simulate_attack(q, z, trials=100_000, seed=3)
        assert estimate == pytest.approx(attacker_success(q, z), rel=0.10)
