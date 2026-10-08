"""An asynchronous gossip network of full nodes, with partitions.

The transport is in-process: a message is handed to the event loop, which puts
it into the receiver's inbox after a random latency. Everything above that is
ordinary node logic: flooding with a deduplication filter, a mempool, block
relay, an orphan pool, fork choice through `Chain`, and a locator-based sync
that reconciles two sides after a partition heals.

Mining is modelled as an exponentially distributed waiting time (real hashing
is in `pow.py`), so block arrivals remain a Poisson process.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

from .chain import Block, Chain, State, Transaction, genesis_block, make_block

#: A transaction may be at most this many nonces ahead of its account.
MAX_NONCE_GAP = 16

#: Transactions per block.
BLOCK_CAPACITY = 4


@dataclass
class Message:
    kind: str  # "tx" | "block" | "head" | "getblocks" | "blocks"
    payload: Any
    sender: str


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


class Network:
    def __init__(self, latency: float = 0.01, seed: int = 0) -> None:
        self.nodes: dict[str, Node] = {}
        self.latency = latency
        self.rng = random.Random(seed)
        #: None when everybody can talk to everybody, else the disjoint groups.
        self.partitions: list[set[str]] | None = None
        self.delivered = 0
        self.dropped = 0

    def register(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        node.network = self

    def partition(self, groups: list[set[str]]) -> None:
        self.partitions = [set(group) for group in groups]

    def heal(self) -> None:
        self.partitions = None

    def can_reach(self, a: str, b: str) -> bool:
        if self.partitions is None:
            return True
        return any(a in group and b in group for group in self.partitions)

    def send(self, source: str, target: str, message: Message) -> None:
        """Schedule delivery after the latency; returns immediately.

        A message across the cut is dropped without any error, so the sender
        cannot tell an unreachable peer from a slow one.
        """
        if not self.can_reach(source, target):
            self.dropped += 1
            return
        delay = self.latency * (0.5 + self.rng.random())
        asyncio.get_running_loop().call_later(delay, self._arrive, source, target, message)

    def _arrive(self, source: str, target: str, message: Message) -> None:
        # Checked again on arrival: a message in flight when the cut appears is lost too.
        node = self.nodes.get(target)
        if node is None or not self.can_reach(source, target):
            self.dropped += 1
            return
        self.delivered += 1
        node.inbox.put_nowait(message)

    async def deliver(self, source: str, target: str, message: Message) -> None:
        self.send(source, target, message)


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #


@dataclass
class NodeStats:
    blocks_mined: int = 0
    blocks_received: int = 0
    blocks_rejected: int = 0
    txs_received: int = 0
    txs_rejected: int = 0
    txs_returned: int = 0
    duplicates_suppressed: int = 0
    reorgs: int = 0


class Node:
    def __init__(
        self,
        node_id: str,
        *,
        initial_balances: dict[str, int],
        hash_power: float = 1.0,
        block_reward: int = 50,
        genesis: Block | None = None,
        seed: int = 0,
    ) -> None:
        self.node_id = node_id
        self.network: Network | None = None
        self.peers: list[str] = []

        self.chain = Chain(genesis or genesis_block())
        self.initial_balances = dict(initial_balances)
        self.block_reward = block_reward

        self.mempool: dict[str, Transaction] = {}
        #: Ids of every transaction and block this node has already handled.
        self.seen: set[str] = set()
        #: Blocks whose parent has not arrived yet, keyed by the missing parent.
        self.orphans: dict[str, list[Block]] = {}

        self.hash_power = hash_power
        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.stats = NodeStats()
        self.running = False
        self._rng = random.Random(f"{seed}/{node_id}")
        self._state_cache: tuple[str, State] | None = None

    # -- views ---------------------------------------------------------------- #

    @property
    def tip_hash(self) -> str:
        return self.chain.tip_hash

    @property
    def height(self) -> int:
        return self.chain.height

    def state(self) -> State:
        """Account state at the current tip (cached per tip; do not mutate)."""
        if self._state_cache is None or self._state_cache[0] != self.chain.tip_hash:
            replayed = self.chain.replay_state(self.initial_balances, reward=self.block_reward)
            self._state_cache = (self.chain.tip_hash, replayed)
        return self._state_cache[1]

    # -- sending -------------------------------------------------------------- #

    def _send(self, peer: str, kind: str, payload: Any) -> None:
        assert self.network is not None
        self.network.send(self.node_id, peer, Message(kind, payload, self.node_id))

    def _relay(self, kind: str, payload: Any, *, heard_from: str | None = None) -> None:
        for peer in self.peers:
            if peer != heard_from:
                self._send(peer, kind, payload)

    async def broadcast(self, message: Message, *, exclude: str | None = None) -> None:
        self._relay(message.kind, message.payload, heard_from=exclude)

    async def announce_head(self) -> None:
        self._relay("head", self.chain.tip)

    # -- transactions --------------------------------------------------------- #

    async def submit_transaction(self, tx: Transaction) -> bool:
        """Entry point for a user. Returns whether the node accepted the transaction."""
        if not self._accept_transaction(tx):
            return False
        self._relay("tx", tx)
        return True

    def _accept_transaction(self, tx: Transaction) -> bool:
        if tx.txid in self.seen:
            self.stats.duplicates_suppressed += 1
            return False
        self.seen.add(tx.txid)
        self.stats.txs_received += 1

        state = self.state()
        account_nonce = state.nonce_of(tx.sender)
        acceptable = (
            tx.amount >= 0
            and tx.fee >= 0
            and tx.amount + tx.fee <= state.balance_of(tx.sender)
            # A small nonce gap is allowed: the missing transaction may simply
            # be still on its way through the network.
            and account_nonce <= tx.nonce <= account_nonce + MAX_NONCE_GAP
        )
        if not acceptable:
            self.stats.txs_rejected += 1
            return False
        self.mempool[tx.txid] = tx
        return True

    def _select_transactions(self) -> tuple[Transaction, ...]:
        """Highest fee first, but never out of nonce order for one sender."""
        state = self.state().copy()
        waiting = sorted(self.mempool.values(), key=lambda t: (-t.fee, t.sender, t.nonce))
        chosen: list[Transaction] = []
        progress = True
        while progress and len(chosen) < BLOCK_CAPACITY:
            progress = False
            for tx in waiting:
                if state.validate(tx) is None:
                    state.apply(tx)
                    chosen.append(tx)
                    waiting.remove(tx)
                    progress = True
                    break
        return tuple(chosen)

    # -- blocks --------------------------------------------------------------- #

    def _connect(self, block: Block) -> list[Block]:
        """Add a block and any orphans waiting for it. Returns the blocks stored.

        Raises KeyError when the parent is unknown and ValueError when the
        block is invalid.
        """
        stored: list[Block] = []
        pending = [block]
        while pending:
            current = pending.pop()
            if self.chain.has(current.hash):
                continue
            if current is not block and not self.chain.has(current.prev_hash):
                continue
            reorgs_before = self.chain.reorg_count
            try:
                self._check_transactions(current)
                moved = self.chain.add_block(current, check_pow=False)
            except ValueError:
                if current is block:
                    raise
                continue  # an invalid orphan does not invalidate its parent
            stored.append(current)
            if moved:
                rolled_back = self.chain.last_disconnected
                if self.chain.reorg_count > reorgs_before:
                    self.stats.reorgs += 1
                self._update_mempool(rolled_back)
            pending.extend(self.orphans.pop(current.hash, []))
        return stored

    def _check_transactions(self, block: Block) -> None:
        """A block is valid only if its transactions apply on its parent's state."""
        if not block.txs or not self.chain.has(block.prev_hash):
            return
        state = self.chain.replay_state(
            self.initial_balances, reward=self.block_reward, tip=block.prev_hash
        )
        for tx in block.txs:
            state.apply(tx)  # raises ValueError

    def _update_mempool(self, rolled_back: list[Block]) -> None:
        """Bring the mempool in line with a new tip.

        Transactions of blocks that just left the canonical chain become
        pending again and are announced once more, so that miners which never
        saw them can include them. Then everything the new chain has already
        consumed (nonce below the account nonce) is dropped.
        """
        state = self.state()
        for block in rolled_back:
            for tx in block.txs:
                if tx.nonce >= state.nonce_of(tx.sender) and tx.txid not in self.mempool:
                    self.mempool[tx.txid] = tx
                    self.stats.txs_returned += 1
                    self._relay("tx", tx)
        self.mempool = {
            txid: tx for txid, tx in self.mempool.items() if tx.nonce >= state.nonce_of(tx.sender)
        }

    def _locator(self) -> list[str]:
        """Hashes of my canonical chain, tip first, thinning out towards genesis."""
        hashes = [b.hash for b in reversed(self.chain.canonical())]
        picked, index, step = [], 0, 1
        while index < len(hashes):
            picked.append(hashes[index])
            if len(picked) > 8:
                step *= 2
            index += step
        if picked[-1] != self.chain.genesis_hash:
            picked.append(self.chain.genesis_hash)
        return picked

    def _blocks_after(self, locator: list[str]) -> list[Block]:
        """My canonical blocks above the first locator entry that is on my chain."""
        canonical = self.chain.canonical()
        height_of = {b.hash: b.height for b in canonical}
        start = next((height_of[h] for h in locator if h in height_of), 0)
        return canonical[start + 1 :]

    # -- message handling ----------------------------------------------------- #

    async def handle(self, message: Message) -> None:
        kind, sender = message.kind, message.sender

        if kind == "tx":
            if self._accept_transaction(message.payload):
                self._relay("tx", message.payload, heard_from=sender)

        elif kind == "block":
            block: Block = message.payload
            if block.hash in self.seen:
                self.stats.duplicates_suppressed += 1
                return
            self.seen.add(block.hash)
            self.stats.blocks_received += 1
            try:
                self._connect(block)
            except KeyError:
                self.orphans.setdefault(block.prev_hash, []).append(block)
                self._send(sender, "getblocks", self._locator())
                return
            except ValueError:
                self.stats.blocks_rejected += 1
                return
            self._relay("block", block, heard_from=sender)

        elif kind == "head":
            head: Block = message.payload
            if not self.chain.has(head.hash):
                self._send(sender, "getblocks", self._locator())

        elif kind == "getblocks":
            self._send(sender, "blocks", self._blocks_after(message.payload))

        elif kind == "blocks":
            for block in message.payload:
                self.seen.add(block.hash)
                try:
                    self._connect(block)
                except (KeyError, ValueError):
                    continue

    async def run(self) -> None:
        self.running = True
        while self.running:
            try:
                message = await asyncio.wait_for(self.inbox.get(), timeout=0.1)
            except TimeoutError:
                continue
            await self.handle(message)

    def stop(self) -> None:
        self.running = False

    # -- mining --------------------------------------------------------------- #

    async def mine_forever(self, mean_block_time: float) -> None:
        """Find blocks after exponential waiting times, at a rate set by hash power."""
        rate = self.hash_power / max(mean_block_time, 1e-6)
        while self.running:
            await asyncio.sleep(self._rng.expovariate(rate))
            if not self.running:
                return
            block = make_block(
                self.chain,
                txs=self._select_transactions(),
                miner=self.node_id,
                timestamp=asyncio.get_running_loop().time(),
            )
            self.seen.add(block.hash)
            self._connect(block)
            self.stats.blocks_mined += 1
            self._relay("block", block)


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #


def build_network(
    node_count: int = 8,
    *,
    peers_per_node: int = 3,
    latency: float = 0.01,
    seed: int = 7,
    initial_balances: dict[str, int] | None = None,
) -> tuple[Network, dict[str, Node]]:
    """A ring (which guarantees connectivity) plus random extra edges."""
    rng = random.Random(seed)
    balances = initial_balances or {f"user{i}": 1000 for i in range(4)}
    genesis = genesis_block()

    network = Network(latency=latency, seed=seed)
    nodes: dict[str, Node] = {}
    for i in range(node_count):
        node = Node(f"node{i}", initial_balances=balances, genesis=genesis, seed=seed)
        nodes[node.node_id] = node
        network.register(node)

    ids = list(nodes)
    links: dict[str, set[str]] = {nid: set() for nid in ids}

    def connect(a: str, b: str) -> None:
        if a != b:
            links[a].add(b)
            links[b].add(a)

    for i, nid in enumerate(ids):
        connect(nid, ids[(i + 1) % node_count])
    wanted = min(peers_per_node, node_count - 1)
    for nid in ids:
        candidates = [other for other in ids if other != nid and other not in links[nid]]
        rng.shuffle(candidates)
        while len(links[nid]) < wanted and candidates:
            connect(nid, candidates.pop())

    for nid in ids:
        nodes[nid].peers = sorted(links[nid])
    return network, nodes


# --------------------------------------------------------------------------- #
# The partition scenario of Part C
# --------------------------------------------------------------------------- #


@dataclass
class SimulationResult:
    nodes: dict[str, Node]
    network: Network
    events: list[str] = field(default_factory=list)
    group_a: list[str] = field(default_factory=list)
    group_b: list[str] = field(default_factory=list)
    miners: list[str] = field(default_factory=list)
    #: Blocks each side added to its best chain while the network was split.
    split_growth: dict[str, int] = field(default_factory=dict)
    #: Per node: reorganisation depths recorded after the heal.
    heal_reorgs: dict[str, list[int]] = field(default_factory=dict)
    #: Transactions submitted during the split that ended up in the final chain.
    #: Height of each side's best chain at the moment of the heal.
    split_heights: dict[str, int] = field(default_factory=dict)
    #: Height of the last block both sides share.
    fork_height: int = 0
    #: Side with less work at the heal ("A", "B", or "" for a tie) and the
    #: number of blocks it has to roll back when it adopts the other branch.
    lighter_side: str = ""
    private_branch_length: int = 0
    split_txs_confirmed: int = 0
    split_txs_total: int = 0

    def heights(self) -> dict[str, int]:
        return {nid: node.height for nid, node in self.nodes.items()}

    def distinct_tips(self) -> int:
        return len({node.tip_hash for node in self.nodes.values()})

    def converged(self) -> bool:
        return self.distinct_tips() == 1

    def reorg_depths(self) -> dict[str, list[int]]:
        return {nid: list(node.chain.reorg_depths) for nid, node in self.nodes.items()}

    def report(self) -> str:
        lines = ["node     side  height  tip           mined  recv  reorgs  depths"]
        lines.append("-" * 72)
        for nid, node in self.nodes.items():
            side = "A" if nid in self.group_a else "B"
            s = node.stats
            lines.append(
                f"{nid:<9}{side:<6}{node.height:>6}  {node.tip_hash[:12]}"
                f"{s.blocks_mined:>7}{s.blocks_received:>6}{s.reorgs:>8}  "
                f"{node.chain.reorg_depths}"
            )
        lines.append("")
        lines.append(f"messages delivered : {self.network.delivered}")
        lines.append(f"messages dropped   : {self.network.dropped}  (blocked by the partition)")
        lines.append(f"consensus reached  : {self.converged()}")
        return "\n".join(lines)


async def run_partition_scenario(
    node_count: int = 8,
    *,
    mean_block_time: float = 0.25,
    seed: int = 7,
    latency: float = 0.005,
    connected_seconds: float = 1.6,
    split_seconds: float = 1.5,
    healed_seconds: float = 3.0,
    settle: bool = True,
) -> SimulationResult:
    """Three phases: connected, split into two halves, healed.

    `mean_block_time` is the mean block interval of the whole network.
    """
    balances = {f"user{i}": 1000 for i in range(4)}
    network, nodes = build_network(
        node_count=node_count, latency=latency, seed=seed, initial_balances=balances
    )
    ids = list(nodes)
    half = node_count // 2
    result = SimulationResult(nodes=nodes, network=network, group_a=ids[:half], group_b=ids[half:])

    tasks = [asyncio.create_task(node.run()) for node in nodes.values()]
    # Every `stride`-th node mines, so that both halves contain miners. With
    # all miners on one side the other side would only stall and nothing would
    # be reorganised at the heal.
    stride = max(1, node_count // 4)
    result.miners = ids[::stride]
    miner_tasks = []
    for nid in result.miners:
        # `mean_block_time` is the interval of the whole network; each miner
        # holds an equal share of the hash power.
        nodes[nid].hash_power = 1.0 / len(result.miners)
        nodes[nid].running = True
        miner_tasks.append(asyncio.create_task(nodes[nid].mine_forever(mean_block_time)))
    tasks.extend(miner_tasks)

    async def announcer() -> None:
        while True:
            await asyncio.sleep(0.15)
            for node in nodes.values():
                await node.announce_head()

    tasks.append(asyncio.create_task(announcer()))

    def side_height(group: list[str]) -> int:
        return max(nodes[nid].height for nid in group)

    def trace(label: str) -> str:
        heights = sorted(set(result.heights().values()))
        return f"{label} distinct tips={result.distinct_tips()} heights={heights}"

    try:
        # Phase 1: connected.
        await asyncio.sleep(connected_seconds / 4)
        for i in range(4):
            tx = Transaction("user0", "user1", 10, nonce=i, fee=1)
            await nodes[ids[i % node_count]].submit_transaction(tx)
        await asyncio.sleep(connected_seconds * 3 / 4)
        result.events.append(trace("phase 1 (connected) ") + f" converged={result.converged()}")

        # Phase 2: split. Each side also receives a transaction of its own.
        network.partition([set(result.group_a), set(result.group_b)])
        result.events.append(f"PARTITION  A={result.group_a}  B={result.group_b}")
        before = {"A": side_height(result.group_a), "B": side_height(result.group_b)}
        split_txs = [
            (result.group_a[0], Transaction("user2", "user3", 5, nonce=0, fee=2)),
            (result.group_b[0], Transaction("user3", "user2", 7, nonce=0, fee=2)),
        ]
        for nid, tx in split_txs:
            await nodes[nid].submit_transaction(tx)
        await asyncio.sleep(split_seconds)
        result.split_growth = {
            "A": side_height(result.group_a) - before["A"],
            "B": side_height(result.group_b) - before["B"],
        }
        result.events.append(
            trace("phase 2 (split)     ")
            + f"  blocks added: A=+{result.split_growth['A']} B=+{result.split_growth['B']}"
        )
        best = {
            side: max((nodes[nid] for nid in group), key=lambda n: n.chain.total_work)
            for side, group in (("A", result.group_a), ("B", result.group_b))
        }
        side_tips = {side: node.tip_hash for side, node in best.items()}
        side_work = {side: node.chain.total_work for side, node in best.items()}
        result.split_heights = {side: node.height for side, node in best.items()}
        reorgs_at_heal = {nid: len(node.chain.reorg_depths) for nid, node in nodes.items()}

        # Phase 3: heal.
        network.heal()
        result.events.append("HEAL")
        # Peers exchange their heads as soon as the connection is back, as in a
        # handshake; the periodic announcement would do the same a bit later.
        for node in nodes.values():
            await node.announce_head()
        await asyncio.sleep(healed_seconds / 3)
        result.events.append(
            trace("  just after heal   ") + "  (an equal-work tie waits for the next block)"
        )
        await asyncio.sleep(healed_seconds * 2 / 3)
        # Stop mining and let the last blocks finish travelling, so that the
        # final snapshot does not catch a block halfway through the network.
        if settle:
            for task in miner_tasks:
                task.cancel()
            await asyncio.sleep(0.3)
        result.events.append(trace("phase 3 (healed)    ") + f" converged={result.converged()}")
    finally:
        for node in nodes.values():
            node.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    result.heal_reorgs = {
        nid: node.chain.reorg_depths[reorgs_at_heal[nid] :] for nid, node in nodes.items()
    }
    reference = nodes[ids[0]]
    if all(reference.chain.has(tip) for tip in side_tips.values()):
        fork = reference.chain.fork_point(side_tips["A"], side_tips["B"])
        result.fork_height = fork.height
        if side_work["A"] != side_work["B"]:
            result.lighter_side = "A" if side_work["A"] < side_work["B"] else "B"
            result.private_branch_length = (
                result.split_heights[result.lighter_side] - fork.height
            )
    confirmed = {tx.txid for block in reference.chain.canonical() for tx in block.txs}
    result.split_txs_total = len(split_txs)
    result.split_txs_confirmed = sum(1 for _nid, tx in split_txs if tx.txid in confirmed)
    return result
