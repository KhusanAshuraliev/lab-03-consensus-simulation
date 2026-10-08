"""Total hash rate of 1, 2 and 4 processes hashing at the same time.

    uv run python experiments/hashrate.py

This separates the hardware from the miner: it shows how much hash rate the
machine really delivers when several processes are busy at once.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consensus.pow import measure_hash_rate  # noqa: E402

SAMPLES = 3_000_000


def main() -> None:
    result = {}
    ctx = mp.get_context("spawn")
    for processes in (1, 2, 4):
        with ctx.Pool(processes) as pool:
            pool.map(measure_hash_rate, [50_000] * processes)  # warm up
            rates = pool.map(measure_hash_rate, [SAMPLES] * processes)
        result[processes] = {"per_process": rates, "total": sum(rates)}
        print(
            f"{processes} process(es): total {sum(rates):>12,.0f} H/s   "
            f"per process {[round(r) for r in rates]}"
        )
    base = result[1]["total"]
    for processes, entry in result.items():
        entry["speedup"] = entry["total"] / base
        print(f"{processes} process(es): {entry['speedup']:.2f}x the hash rate of one")
    out = Path(__file__).resolve().parent.parent / "evidence" / "data" / "hashrate_scaling.json"
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
