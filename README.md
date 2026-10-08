# lab-03-consensus-simulation — my Lab 03 implementation

This is my implementation of proof of work, Casper FFG proof of stake and an asynchronous gossip network, which I wrote for **OE Blockchain, Lab 03 (Consensus: Proof of Work, Proof of Stake and gossip networking)**.

All three parts run on one ledger that I wrote first (`consensus/chain.py`). I used the Python standard library only: `hashlib`, `struct`, `asyncio`, `multiprocessing`, `statistics`, `random`. I do not use any blockchain, networking or consensus library.

## Report

- [My Lab 03 report (PDF)](report/Lab03_Report_Khusanjon_Ashuraliev.pdf)
- [`evidence/`](evidence/) — the output of my demos and test runs, and the raw numbers behind every figure of the report (`evidence/data/*.json`)

## Usage

```bash
uv sync
uv run pytest                          # 95 passed
uv run python -m consensus             # Parts A, B and C
uv run python -m consensus pow --blocks 100 --workers 1 --difficulty 8 --target 1.0
uv run python -m consensus pos
uv run python -m consensus gossip --nodes 12 --seed 42
uv run python -m consensus bonus       # bonus E1: double-spend probability
```

## How I covered the requirements

| # | Requirement | Where I implemented it | How I tested it |
|---|---|---|---|
| 1 | Ledger: balances, sequential nonce, block tree, heaviest chain, retained branches, reorg depth | `chain.py` → `State`, `Block`, `Chain.add_block`, `Chain.fork_point` | skipped / repeated nonce and overspending rejected; the tip moves only on strictly more work; the old branch stays stored and can win back; depth equals the blocks rolled back; one block of difficulty 8 beats three of difficulty 1 |
| 2 | Proof of work: header hash, target / difficulty, clamped retarget, multiprocess miner with strided nonces and early termination | `pow.py` → `retarget`, `mine_range`, `mine`, `MinerPool`, `mine_parallel` | every mined block is below its target and accepted with `check_pow=True`; 2000 / 500 / 4000 / 250 and the floor of 1; four strides together find the same first nonce as the plain search; no child process is left after `mine_parallel` |
| 3 | Interval statistics | `pow.py` → `MiningStats`, `simulate_mining` | a deterministic test over 150 blocks: attempts per block give median / mean and stdev / mean inside the ranges of Part A |
| 4 | Casper FFG: strict two thirds, consecutive checkpoints, justified source | `pos.py` → `BeaconState._justify_and_finalise` | 6 of 9 do not justify and 7 do; repeating a vote adds no weight; a link 1 → 3 justifies but finalises nothing; an unjustified source changes nothing |
| 5 | Slashing: double vote, surround vote, initial and correlated penalty | `pos.py` → `violated_rule`, `BeaconState._slash` | the loss table 2.41 / 8.06 / 15.15 / 29.43 / 32.00; a surround vote is caught in either order of arrival; three conflicting votes slash once |
| 6 | Inactivity leak | `pos.py` → `BeaconState._leak`, `inactivity_leak_trace` | no leak in epochs 1–4; voters keep exactly 32 ETH; no leak while the chain finalises; justification returns in epoch 82 and finality in epoch 83 |
| 7 | Gossip network: mempool, deduplication, block relay, partition, reconciliation | `network.py` → `Network`, `Node`, `run_partition_scenario` | each node counts a transaction and a block exactly once; nothing crosses a partition, not even a message already in flight; after the heal all nodes agree and the transaction of the orphaned block is pending again on every node, then confirmed again |

`tests/test_consensus.py` is the course's reference suite (42 tests). I copied it **unchanged**.
In `tests/test_extra.py` I added 53 further tests of my own.

## Layout

| File | What I put there |
|---|---|
| `consensus/chain.py` | transactions, account state, binary block header, Merkle root, block tree and fork choice |
| `consensus/pow.py` | target and difficulty, retargeting, the hash loop, the worker pool, interval statistics |
| `consensus/pos.py` | checkpoints and attestations, the two slashing rules, the beacon state machine, the two Part B experiments |
| `consensus/network.py` | transport with latency and partitions, the node, topology, the three-phase partition scenario |
| `consensus/doublespend.py` | bonus E1: Nakamoto's formula and a Monte Carlo simulation of the same race |
| `consensus/cli.py` | the demonstrations |
| `experiments/collect.py` | runs the measurements of my report and stores the numbers in `evidence/data/` |
| `experiments/hashrate.py` | measures the raw hash rate of 1, 2 and 4 processes running at once (used for bonus E2) |
| `experiments/charts.py` | draws the figures of my report as SVG from those numbers |

## My design decisions

- **Binary header with a constant prefix.** I serialise the header as fixed binary fields with the nonce last. The miner hashes the prefix once and copies that SHA-256 state for every attempt, so one attempt costs only the eight nonce bytes plus the second hash. On my machine this gives about 1.5 million hashes per second in one process.
- **A worker pool instead of processes per block.** My parallel miner starts its processes once. Each worker gets the header prefix and the target through its own queue and searches the nonces `i, i+W, i+2W, …`. A shared job number tells the workers which search is current; they look at it every 1,024 hashes and stop when it changes. `mine_parallel` is the same pool used for a single block, and all its processes have exited when it returns. This is also bonus challenge E2.
- **Fixed difficulty for the statistics.** Because my miner is fast enough for retargeting to move the difficulty, I measure the interval distribution with retargeting switched off. A sample across several difficulties would be a mixture of exponential distributions and not one.
- **Finality follows the link.** A checkpoint becomes final when a supermajority link leads from it to the checkpoint of the very next epoch. I count each validator at most once per link, and I look for slashable pairs at the moment the second vote arrives, so an epoch never has to compare all votes with all votes.
- **Transactions really return to the mempool.** `Chain` remembers which blocks left the canonical chain in a reorganisation. The node puts their transactions back into its mempool and announces them again, because the nodes on the winning side may never have seen them.
- **Orphan pool and locator sync.** A block that arrives before its parent waits in an orphan pool. To catch up, a node sends a short list of its own block hashes (dense near the tip, sparse towards genesis) and receives only the blocks after the first hash the peer recognises.
- **No tie-break.** Two branches of equal work both stay, and each node keeps the one it saw first. The next block decides.
- **Messages are scheduled, not awaited.** Sending hands the message to the event loop with its latency and returns at once, so a node is never blocked while its own messages travel. A message that is still in flight when a partition appears is dropped on arrival.

## Bonus challenges

I attempted E1 (double-spend probability) and E2 (a miner that scales). Both have their own section in the report. I did not attempt E3, E4 or E5.
