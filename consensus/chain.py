"""The shared ledger: transactions, account state, blocks and the block tree.

Both consensus mechanisms of the lab order blocks of this ledger. The model is
account based (balance plus a sequential nonce per account), and the fork choice
is "most cumulative work", computed over a tree in which losing branches stay
stored so that they can win later.

The header is a fixed binary layout rather than text. Everything except the
nonce forms a constant prefix, which lets the miner hash the prefix once and
only feed the eight nonce bytes per attempt.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, replace
from functools import cached_property

MAX_TARGET = (1 << 256) - 1

#: Target of difficulty 1: one hash in 2**16 succeeds, so a Python miner needs
#: a fraction of a second per block.
GENESIS_TARGET = MAX_TARGET >> 16

ZERO_HASH = "00" * 32


def sha256d(data: bytes) -> bytes:
    """SHA-256 applied twice, as in Bitcoin."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def _field(text: str) -> bytes:
    """Length-prefixed UTF-8, so that adjacent fields cannot run into each other."""
    raw = text.encode()
    return struct.pack(">H", len(raw)) + raw


# --------------------------------------------------------------------------- #
# Transactions and state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Transaction:
    sender: str
    recipient: str
    amount: int
    nonce: int
    fee: int = 0

    def serialise(self) -> bytes:
        return (
            _field(self.sender)
            + _field(self.recipient)
            + struct.pack(">qqq", self.amount, self.nonce, self.fee)
        )

    @cached_property
    def txid(self) -> str:
        return sha256d(self.serialise()).hex()


class State:
    """Balances and nonces. `apply` is the state transition function.

    It uses nothing but its inputs (no clock, no randomness), so every node
    that replays the same blocks in the same order reaches the same state.
    """

    def __init__(self, balances: dict[str, int] | None = None) -> None:
        self.balances: dict[str, int] = dict(balances or {})
        self.nonces: dict[str, int] = {}

    def copy(self) -> State:
        twin = State(self.balances)
        twin.nonces = dict(self.nonces)
        return twin

    def balance_of(self, account: str) -> int:
        return self.balances.get(account, 0)

    def nonce_of(self, account: str) -> int:
        return self.nonces.get(account, 0)

    def validate(self, tx: Transaction) -> str | None:
        """None when the transaction can be applied now, otherwise the reason."""
        if tx.amount < 0 or tx.fee < 0:
            return "negative amount or fee"
        expected = self.nonce_of(tx.sender)
        if tx.nonce < expected:
            return f"nonce already used: account is at {expected}, got {tx.nonce}"
        if tx.nonce > expected:
            return f"nonce gap: account is at {expected}, got {tx.nonce}"
        if tx.amount + tx.fee > self.balance_of(tx.sender):
            return "insufficient balance"
        return None

    def apply(self, tx: Transaction) -> None:
        reason = self.validate(tx)
        if reason is not None:
            raise ValueError(f"cannot apply {tx}: {reason}")
        self.balances[tx.sender] = self.balance_of(tx.sender) - tx.amount - tx.fee
        self.balances[tx.recipient] = self.balance_of(tx.recipient) + tx.amount
        self.nonces[tx.sender] = tx.nonce + 1

    def apply_block(self, block: Block, reward: int = 0) -> None:
        """Apply all transactions, then pay reward and fees to the miner."""
        fees = 0
        for tx in block.txs:
            self.apply(tx)
            fees += tx.fee
        if block.miner:
            self.balances[block.miner] = self.balance_of(block.miner) + reward + fees


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


def merkle_root(txids: list[str]) -> str:
    """Bitcoin-style Merkle root: pair up, hash, duplicate the last when odd."""
    if not txids:
        return ZERO_HASH
    level = [bytes.fromhex(t) for t in txids]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha256d(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0].hex()


@dataclass(frozen=True)
class Block:
    height: int
    prev_hash: str
    timestamp: float
    txs: tuple[Transaction, ...] = ()
    difficulty: int = 1
    nonce: int = 0
    miner: str = ""

    @property
    def target(self) -> int:
        return GENESIS_TARGET // max(1, self.difficulty)

    @property
    def work(self) -> int:
        """Expected number of hashes behind this block (Bitcoin's formula)."""
        return (1 << 256) // (self.target + 1)

    @cached_property
    def merkle_root(self) -> str:
        return merkle_root([tx.txid for tx in self.txs])

    @cached_property
    def header_prefix(self) -> bytes:
        """Every header field except the nonce."""
        return (
            struct.pack(">Q", self.height)
            + bytes.fromhex(self.prev_hash)
            + bytes.fromhex(self.merkle_root)
            + struct.pack(">dQ", float(self.timestamp), self.difficulty)
            + _field(self.miner)
        )

    def header_bytes(self, nonce: int | None = None) -> bytes:
        return self.header_prefix + struct.pack(">Q", self.nonce if nonce is None else nonce)

    @cached_property
    def hash(self) -> str:
        return sha256d(self.header_bytes()).hex()

    def is_valid_pow(self) -> bool:
        return int(self.hash, 16) < self.target

    def with_nonce(self, nonce: int) -> Block:
        return replace(self, nonce=nonce)


def genesis_block(difficulty: int = 1) -> Block:
    """The same genesis for every node: all fields are constants."""
    return Block(height=0, prev_hash=ZERO_HASH, timestamp=0.0, difficulty=difficulty)


# --------------------------------------------------------------------------- #
# The block tree
# --------------------------------------------------------------------------- #


@dataclass
class ChainEntry:
    block: Block
    total_work: int


class Chain:
    """All known blocks, plus a pointer to the tip with the most cumulative work."""

    def __init__(self, genesis: Block | None = None) -> None:
        genesis = genesis or genesis_block()
        self.entries: dict[str, ChainEntry] = {genesis.hash: ChainEntry(genesis, genesis.work)}
        self.genesis_hash = genesis.hash
        self.tip_hash = genesis.hash
        self.reorg_count = 0
        self.reorg_depths: list[int] = []
        #: Blocks that left the canonical chain in the most recent tip change
        #: (empty when the tip was simply extended), oldest first.
        self.last_disconnected: list[Block] = []

    # -- reading -------------------------------------------------------------- #

    @property
    def tip(self) -> Block:
        return self.entries[self.tip_hash].block

    @property
    def height(self) -> int:
        return self.tip.height

    @property
    def total_work(self) -> int:
        return self.entries[self.tip_hash].total_work

    def has(self, block_hash: str) -> bool:
        return block_hash in self.entries

    def path_to(self, block_hash: str) -> list[Block]:
        """Blocks from genesis up to and including the given block."""
        path: list[Block] = []
        cursor = block_hash
        while True:
            block = self.entries[cursor].block
            path.append(block)
            if cursor == self.genesis_hash:
                break
            cursor = block.prev_hash
        path.reverse()
        return path

    def canonical(self) -> list[Block]:
        return self.path_to(self.tip_hash)

    def fork_point(self, hash_a: str, hash_b: str) -> Block:
        """Last common ancestor of two stored blocks."""
        a = self.entries[hash_a].block
        b = self.entries[hash_b].block
        while a.hash != b.hash:
            if a.height >= b.height:
                a = self.entries[a.prev_hash].block
            else:
                b = self.entries[b.prev_hash].block
        return a

    # -- writing -------------------------------------------------------------- #

    def add_block(self, block: Block, *, check_pow: bool = True) -> bool:
        """Store a block; return True when it became the new tip.

        The tip moves only on strictly more work. A block that does not win is
        kept anyway, because a later block may extend it past the current tip.
        """
        if block.hash in self.entries:
            return False
        parent = self.entries.get(block.prev_hash)
        if parent is None:
            raise KeyError(f"unknown parent {block.prev_hash[:12]} of block {block.hash[:12]}")
        if block.height != parent.block.height + 1:
            raise ValueError("height is not parent height + 1")
        if check_pow and not block.is_valid_pow():
            raise ValueError("invalid proof of work: header hash is not below the target")

        total = parent.total_work + block.work
        self.entries[block.hash] = ChainEntry(block, total)
        if total <= self.total_work:
            return False

        old_tip = self.tip_hash
        ancestor = self.fork_point(old_tip, block.prev_hash)
        rolled_back = self.path_to(old_tip)[ancestor.height + 1 :]
        self.last_disconnected = rolled_back
        if rolled_back:
            self.reorg_count += 1
            self.reorg_depths.append(len(rolled_back))
        self.tip_hash = block.hash
        return True

    # -- derived -------------------------------------------------------------- #

    def replay_state(
        self, initial: dict[str, int], reward: int = 50, *, tip: str | None = None
    ) -> State:
        """State after replaying the chain from genesis to `tip` (default: the tip)."""
        state = State(initial)
        for block in self.path_to(tip or self.tip_hash)[1:]:
            state.apply_block(block, reward=reward)
        return state

    def block_intervals(self) -> list[float]:
        stamps = [b.timestamp for b in self.canonical()]
        return [later - earlier for earlier, later in zip(stamps, stamps[1:], strict=False)]


def make_block(
    chain: Chain,
    *,
    txs: tuple[Transaction, ...] = (),
    difficulty: int | None = None,
    miner: str = "",
    timestamp: float | None = None,
) -> Block:
    """An unmined candidate on top of the current tip."""
    tip = chain.tip
    return Block(
        height=tip.height + 1,
        prev_hash=tip.hash,
        timestamp=time.time() if timestamp is None else timestamp,
        txs=tuple(txs),
        difficulty=tip.difficulty if difficulty is None else difficulty,
        miner=miner,
    )
