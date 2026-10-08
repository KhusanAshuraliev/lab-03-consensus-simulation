"""Proof of work: difficulty, retargeting, mining and interval statistics.

The parallel miner is a pool of worker processes that are started once and then
reused for every block. A worker receives the header prefix and the target
through its own queue and walks a strided nonce sequence (worker i of W tries
i, i+W, i+2W, ...). A shared job number tells the workers which search is
current; as soon as it changes, they drop what they are doing.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import queue as queue_module
import statistics
import struct
import time
from dataclasses import dataclass

from .chain import GENESIS_TARGET, Block, Chain, genesis_block, make_block

#: One retarget may change difficulty by at most this factor, up or down.
MAX_ADJUSTMENT_FACTOR = 4

#: Hashes a pool worker computes between two looks at the shared job number.
CHECK_EVERY = 1024

_NONCE = struct.Struct(">Q")


# --------------------------------------------------------------------------- #
# Difficulty
# --------------------------------------------------------------------------- #


def target_from_difficulty(difficulty: int) -> int:
    return GENESIS_TARGET // max(1, difficulty)


def difficulty_from_target(target: int) -> int:
    return max(1, GENESIS_TARGET // max(1, target))


def retarget(
    current_difficulty: int,
    actual_seconds: float,
    target_seconds: float,
    *,
    clamp: int = MAX_ADJUSTMENT_FACTOR,
) -> int:
    """New difficulty after a window that took `actual_seconds`.

    target_new = target_old * actual / expected, and difficulty is the inverse
    of target, so difficulty_new = difficulty_old * expected / actual. The
    measured time is first limited to [expected / clamp, expected * clamp],
    which bounds the change to a factor of `clamp`; the result never drops
    below 1.
    """
    shortest = target_seconds / clamp
    longest = target_seconds * clamp
    bounded = min(max(actual_seconds, shortest), longest)
    return max(1, int(current_difficulty * target_seconds / bounded))


# --------------------------------------------------------------------------- #
# Searching
# --------------------------------------------------------------------------- #


def _scan(prefix: bytes, target: int, start: int, step: int, attempts: int) -> int | None:
    """Try `attempts` nonces start, start+step, ... and return the first that wins.

    The header prefix is absorbed once; each attempt copies that hash state and
    adds only the nonce.
    """
    base = hashlib.sha256(prefix)
    sha256 = hashlib.sha256
    pack = _NONCE.pack
    from_bytes = int.from_bytes
    nonce = start
    for _ in range(attempts):
        inner = base.copy()
        inner.update(pack(nonce))
        if from_bytes(sha256(inner.digest()).digest(), "big") < target:
            return nonce
        nonce += step
    return None


def mine_range(block: Block, start: int, stop: int, step: int = 1) -> int | None:
    """Search the nonces start, start+step, ... below `stop`."""
    attempts = max(0, -(-(stop - start) // step))
    return _scan(block.header_prefix, block.target, start, step, attempts)


def mine(block: Block, max_nonce: int = 2**64) -> Block:
    """Mine in the calling process."""
    prefix, target = block.header_prefix, block.target
    start = 0
    while start < max_nonce:
        attempts = min(1 << 16, max_nonce - start)
        nonce = _scan(prefix, target, start, 1, attempts)
        if nonce is not None:
            return block.with_nonce(nonce)
        start += attempts
    raise RuntimeError("nonce space exhausted")


# --------------------------------------------------------------------------- #
# Parallel mining
# --------------------------------------------------------------------------- #


def _pool_worker(index: int, stride: int, jobs, results, current_job) -> None:
    """Body of one worker process. Must stay a module-level function (spawn)."""
    results.put(("ready", index))
    while True:
        job = jobs.get()
        if job is None:
            return
        job_id, prefix, target = job
        nonce = index
        while current_job.value == job_id:
            hit = _scan(prefix, target, nonce, stride, CHECK_EVERY)
            if hit is not None:
                results.put((job_id, hit))
                break
            nonce += stride * CHECK_EVERY


class MinerPool:
    """Worker processes that stay alive between blocks."""

    def __init__(self, workers: int | None = None, *, start_timeout: float = 60.0) -> None:
        self.workers = workers or max(1, mp.cpu_count() - 1)
        ctx = mp.get_context("spawn")
        self._current_job = ctx.Value("q", 0)
        self._results = ctx.Queue()
        self._job_queues = [ctx.Queue() for _ in range(self.workers)]
        self._job_id = 0
        self.processes = [
            ctx.Process(
                target=_pool_worker,
                args=(i, self.workers, self._job_queues[i], self._results, self._current_job),
                daemon=True,
            )
            for i in range(self.workers)
        ]
        for process in self.processes:
            process.start()
        # Wait until every worker has imported the module, so that process
        # start-up is never measured as mining time.
        for _ in self.processes:
            try:
                self._results.get(timeout=start_timeout)
            except queue_module.Empty as exc:
                self.close()
                raise RuntimeError("mining workers did not start") from exc

    def mine(self, block: Block, timeout: float = 120.0) -> Block:
        self._job_id += 1
        job_id = self._job_id
        self._current_job.value = job_id
        job = (job_id, block.header_prefix, block.target)
        for jobs in self._job_queues:
            jobs.put(job)

        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("parallel mining timed out")
                try:
                    found_job, nonce = self._results.get(timeout=remaining)
                except queue_module.Empty as exc:
                    raise RuntimeError("parallel mining timed out") from exc
                if found_job == job_id:  # anything else is a late answer to an old job
                    break
        finally:
            self._current_job.value = 0  # early termination for the losers

        mined = block.with_nonce(nonce)
        if not mined.is_valid_pow():
            raise RuntimeError("a worker returned a nonce that does not satisfy the target")
        return mined

    def close(self) -> None:
        self._current_job.value = 0
        for jobs in self._job_queues:
            jobs.put(None)
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join()
        for q in (*self._job_queues, self._results):
            q.close()
            q.join_thread()

    def __enter__(self) -> MinerPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def mine_parallel(block: Block, workers: int | None = None, timeout: float = 60.0) -> Block:
    """Mine one block with a temporary pool. All workers have exited on return."""
    with MinerPool(workers) as pool:
        return pool.mine(block, timeout=timeout)


def measure_hash_rate(samples: int = 100_000) -> float:
    """Hashes per second of one process, measured on a target that never hits."""
    block = Block(height=1, prev_hash="11" * 32, timestamp=1.0)
    started = time.perf_counter()
    _scan(block.header_prefix, 0, 0, 1, samples)
    return samples / (time.perf_counter() - started)


# --------------------------------------------------------------------------- #
# Statistics and simulation
# --------------------------------------------------------------------------- #


@dataclass
class MiningStats:
    intervals: list[float]
    difficulties: list[int]
    #: (difficulty in the window, seconds the window took), one per retarget window
    windows: list[tuple[int, float]]

    @property
    def mean_interval(self) -> float:
        return statistics.fmean(self.intervals) if self.intervals else 0.0

    @property
    def median_interval(self) -> float:
        return statistics.median(self.intervals) if self.intervals else 0.0

    @property
    def stdev_interval(self) -> float:
        return statistics.pstdev(self.intervals) if len(self.intervals) > 1 else 0.0

    @property
    def median_over_mean(self) -> float:
        return self.median_interval / self.mean_interval if self.mean_interval else 0.0

    @property
    def stdev_over_mean(self) -> float:
        return self.stdev_interval / self.mean_interval if self.mean_interval else 0.0

    def report(self) -> str:
        return "\n".join(
            [
                f"blocks mined        : {len(self.intervals)}",
                f"mean interval       : {self.mean_interval:.4f} s",
                f"median interval     : {self.median_interval:.4f} s",
                f"stdev of intervals  : {self.stdev_interval:.4f} s",
                f"min / max interval  : {min(self.intervals):.4f} / {max(self.intervals):.4f} s",
                f"difficulty start/end: {self.difficulties[0]} -> {self.difficulties[-1]}",
                f"median / mean       : {self.median_over_mean:.3f}   (exponential: ln 2 = 0.693)",
                f"stdev / mean        : {self.stdev_over_mean:.3f}   (exponential: 1.0)",
            ]
        )


def simulate_mining(
    blocks: int = 30,
    *,
    retarget_interval: int | None = 10,
    target_seconds: float = 0.30,
    initial_difficulty: int = 1,
    workers: int | None = None,
    miner: str = "miner-0",
    pool: MinerPool | None = None,
) -> tuple[Chain, MiningStats]:
    """Mine `blocks` blocks and record how long each one took.

    With `retarget_interval=None` the difficulty stays fixed, which is what the
    interval statistics need: a sample drawn at one difficulty is one
    exponential distribution, a sample across retargets is a mixture of several.

    `workers=1` mines in the calling process; any other value starts a pool for
    the duration of the run. A pool passed in as `pool` is used as it is and
    left running.
    """
    chain = Chain(genesis_block(initial_difficulty))
    intervals: list[float] = []
    difficulties = [initial_difficulty]
    windows: list[tuple[int, float]] = []
    difficulty = initial_difficulty

    own_pool = pool is None and workers != 1
    if own_pool:
        pool = MinerPool(workers)
    try:
        window_time = 0.0
        for number in range(1, blocks + 1):
            candidate = make_block(chain, difficulty=difficulty, miner=miner)
            started = time.perf_counter()
            mined = pool.mine(candidate) if pool else mine(candidate)
            elapsed = time.perf_counter() - started

            chain.add_block(mined)
            intervals.append(elapsed)
            window_time += elapsed

            if retarget_interval and number % retarget_interval == 0:
                windows.append((difficulty, window_time))
                difficulty = retarget(difficulty, window_time, target_seconds * retarget_interval)
                window_time = 0.0
            difficulties.append(difficulty)
    finally:
        if own_pool:
            pool.close()

    return chain, MiningStats(intervals, difficulties, windows)
