"""Draw the figures of my report as SVG files from evidence/data/*.json.

    uv run python experiments/charts.py

Standard library only, like the rest of the project.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "evidence" / "data"
FIGURES = ROOT / "report" / "figures"

BLUE, RED, GREY, GREEN = "#1f5fa8", "#c0392b", "#8a8f98", "#2e7d32"


class Chart:
    """A small x/y chart with linear axes."""

    def __init__(self, title, xlabel, ylabel, xmax, ymax, *, xmin=0.0, ymin=0.0, w=640, h=330):
        self.w, self.h = w, h
        self.left, self.right, self.top, self.bottom = 62, 18, 34, 46
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.items: list[str] = [
            f'<rect width="{w}" height="{h}" fill="white"/>',
            f'<text x="{w / 2}" y="18" text-anchor="middle" font-size="13" font-weight="bold">{title}</text>',
            f'<text x="{(self.left + w - self.right) / 2}" y="{h - 8}" text-anchor="middle" font-size="11">{xlabel}</text>',
            f'<text transform="translate(14,{(self.top + h - self.bottom) / 2}) rotate(-90)" text-anchor="middle" font-size="11">{ylabel}</text>',
        ]

    def x(self, value: float) -> float:
        span = self.w - self.left - self.right
        return self.left + (value - self.xmin) / (self.xmax - self.xmin) * span

    def y(self, value: float) -> float:
        span = self.h - self.top - self.bottom
        return self.h - self.bottom - (value - self.ymin) / (self.ymax - self.ymin) * span

    def axes(self, xticks, yticks, xfmt="{:g}", yfmt="{:g}") -> None:
        x0, x1, y0, y1 = self.x(self.xmin), self.x(self.xmax), self.y(self.ymin), self.y(self.ymax)
        for tick in yticks:
            self.items.append(
                f'<line x1="{x0}" y1="{self.y(tick):.1f}" x2="{x1}" y2="{self.y(tick):.1f}" stroke="#e3e5e8"/>'
                f'<text x="{x0 - 6}" y="{self.y(tick) + 3.5:.1f}" text-anchor="end" font-size="10">{yfmt.format(tick)}</text>'
            )
        for tick in xticks:
            self.items.append(
                f'<line x1="{self.x(tick):.1f}" y1="{y0}" x2="{self.x(tick):.1f}" y2="{y0 + 4}" stroke="#333"/>'
                f'<text x="{self.x(tick):.1f}" y="{y0 + 16}" text-anchor="middle" font-size="10">{xfmt.format(tick)}</text>'
            )
        self.items.append(
            f'<polyline points="{x0},{y1} {x0},{y0} {x1},{y0}" fill="none" stroke="#333"/>'
        )

    def line(self, points, colour=BLUE, width=1.8, dash="") -> None:
        path = " ".join(f"{self.x(px):.1f},{self.y(py):.1f}" for px, py in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="{width}"{extra}/>'
        )

    def dots(self, points, colour=BLUE, radius=3.2, fill=None) -> None:
        for px, py in points:
            self.items.append(
                f'<circle cx="{self.x(px):.1f}" cy="{self.y(py):.1f}" r="{radius}" '
                f'fill="{fill or colour}" stroke="{colour}"/>'
            )

    def bar(self, x_from, x_to, value, colour=BLUE) -> None:
        self.items.append(
            f'<rect x="{self.x(x_from) + 1:.1f}" y="{self.y(value):.1f}" '
            f'width="{self.x(x_to) - self.x(x_from) - 2:.1f}" '
            f'height="{self.y(self.ymin) - self.y(value):.1f}" fill="{colour}" opacity="0.85"/>'
        )

    def label(self, px, py, text, colour="#333", anchor="start", dx=0, dy=0) -> None:
        self.items.append(
            f'<text x="{self.x(px) + dx:.1f}" y="{self.y(py) + dy:.1f}" text-anchor="{anchor}" '
            f'font-size="10" fill="{colour}">{text}</text>'
        )

    def save(self, name: str) -> None:
        FIGURES.mkdir(parents=True, exist_ok=True)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'font-family="Helvetica, Arial, sans-serif">' + "".join(self.items) + "</svg>"
        )
        (FIGURES / name).write_text(svg)
        print(f"wrote report/figures/{name}")


def nice_ticks(top: float, count: int = 5) -> list[float]:
    raw = top / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return [round(i * step, 10) for i in range(math.ceil(top / step) + 1)]


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


# --------------------------------------------------------------------------- #


def interval_histogram(source: str, out: str, title: str, bins: int = 12) -> None:
    data = load(source) if isinstance(source, str) else source
    intervals = data["intervals"]
    mean, top = data["summary"]["mean"], max(intervals)
    width = top / bins
    counts = [0] * bins
    for value in intervals:
        counts[min(bins - 1, int(value / width))] += 1
    yticks = nice_ticks(max(counts) * 1.08)
    chart = Chart(title, "block interval (s)", "blocks", top, yticks[-1] + (yticks[1] - yticks[0]) * 0.4)
    chart.axes([i * width for i in range(0, bins + 1, 2)], yticks, xfmt="{:.2f}")
    for i, count in enumerate(counts):
        chart.bar(i * width, (i + 1) * width, count)
    # Expected count per bin for an exponential distribution with the same mean.
    n = len(intervals)
    curve = [
        (t, n * (math.exp(-max(0.0, t - width / 2) / mean) - math.exp(-(t + width / 2) / mean)))
        for t in (width / 2 + i * width / 4 for i in range(4 * bins - 3))
    ]
    chart.line(curve, colour=RED, dash="5 3")
    chart.label(top * 0.42, yticks[-1] * 0.82, f"dashed: exponential with the measured mean ({mean:.2f} s)", RED)
    chart.save(out)


def difficulty_plot() -> None:
    data = load("pow_retarget")
    windows = data["windows"]
    difficulties = [w["difficulty"] for w in windows] + [data["final_difficulty"]]
    yticks = nice_ticks(max(difficulties) * 1.1)
    chart = Chart(
        "Difficulty per retarget window (10 blocks, goal 1.0 s per block)",
        "retarget window",
        "difficulty",
        len(difficulties) + 0.5,
        yticks[-1] + (yticks[1] - yticks[0]) * 0.3,
        xmin=0.5,
    )
    chart.axes(range(1, len(difficulties) + 1, 2), yticks)
    steps = []
    for i, d in enumerate(difficulties, start=1):
        steps += [(i - 0.5, d), (i + 0.5, d)]
    chart.line(steps)
    chart.dots([(i, d) for i, d in enumerate(difficulties, start=1)])
    equilibrium = data["equilibrium_estimate"]
    chart.line([(0.5, equilibrium), (len(difficulties) + 0.5, equilibrium)], colour=RED, dash="5 3")
    chart.label(
        len(difficulties) * 0.45, equilibrium, f"hash rate x 1 s / 65,536 = {equilibrium:.1f}",
        RED, dy=-5,
    )
    chart.save("difficulty.svg")


def slashing_plot() -> None:
    table = {int(k): v for k, v in load("pos")["loss_eth_by_offenders"].items()}
    chart = Chart(
        "Loss per offender against simultaneous offenders (64 validators)",
        "validators slashed together",
        "loss per validator (ETH)",
        32.5,
        35,
    )
    chart.axes([1, 5, 10, 15, 20, 25, 30], [0, 8, 16, 24, 32])
    chart.line(sorted(table.items()))
    chart.dots([(k, table[k]) for k in (1, 5, 10, 20, 30)])
    for k in (1, 5, 10, 20, 30):
        chart.label(k, table[k], f"{table[k]:.2f}", anchor="middle", dy=-8 if k < 30 else 14)
    third = 64 / 3
    chart.line([(third, 0), (third, 35)], colour=RED, dash="5 3")
    chart.label(third, 3, "1/3 of the stake", RED, dx=5)
    chart.save("slashing.svg")


def leak_plot() -> None:
    rows = load("pos")["leak_trace"]
    chart = Chart(
        "Inactivity leak: stake share of the 24 validators that keep voting",
        "epoch",
        "voting share of active stake",
        100,
        0.75,
        ymin=0.30,
    )
    chart.axes(range(0, 101, 10), [0.3, 0.4, 0.5, 0.6, 0.7], yfmt="{:.0%}")
    chart.line([(0, 0.375)] + [(r["epoch"], r["voting_share"]) for r in rows])
    chart.line([(0, 2 / 3), (100, 2 / 3)], colour=RED, dash="5 3")
    chart.label(2, 2 / 3, "two thirds", RED, dy=-5)
    justified = next(r["epoch"] for r in rows if r["justified"])
    finalized = next(r["epoch"] for r in rows if r["finalized"])
    chart.line([(4.5, 0.30), (4.5, 0.75)], colour=GREY, dash="2 3")
    chart.label(4.5, 0.32, "leak starts (epoch 5)", GREY, dx=4)
    chart.line([(justified, 0.30), (justified, 0.75)], colour=GREEN, dash="2 3")
    chart.label(
        justified, 0.32, f"justified again: {justified}, final: {finalized}", GREEN, "end", dx=-4
    )
    chart.save("leak.svg")


def reorg_sweep_plot() -> None:
    sweep = {float(k): v for k, v in load("gossip")["duration_sweep"].items()}
    top = max(max(v["depths"]) for v in sweep.values())
    yticks = nice_ticks(top * 1.1)
    chart = Chart(
        "Deepest reorganisation at the heal against partition length (12 nodes)",
        "partition length (s)",
        "blocks rolled back",
        max(sweep) * 1.06,
        yticks[-1] + (yticks[1] - yticks[0]) * 0.3,
    )
    chart.axes(sorted(sweep), yticks)
    for seconds, entry in sweep.items():
        chart.dots([(seconds, d) for d in entry["depths"]], colour=GREY, radius=2.4, fill="none")
    means = sorted((seconds, entry["mean"]) for seconds, entry in sweep.items())
    chart.line(means)
    chart.dots(means)
    chart.label(means[-1][0], means[-1][1], "mean of 10 seeds", BLUE, "end", dx=-8, dy=-6)
    chart.save("reorg_sweep.svg")


def pool_plots() -> None:
    data = load("pool_scaling")
    runs = {int(k): v for k, v in data["runs"].items()}
    chart = Chart(
        f"Worker pool: speed-up at fixed difficulty {data['difficulty']} ({data['blocks']} blocks per point)",
        "worker processes",
        "speed-up against one worker",
        4.4,
        4.6,
        xmin=0.6,
    )
    chart.axes([1, 2, 3, 4], [0, 1, 2, 3, 4])
    chart.line([(0.6, 0.6), (4.4, 4.4)], colour=GREY, dash="5 3")
    chart.label(3.3, 3.3, "linear", GREY, dx=6, dy=10)
    points = sorted((w, r["speedup"]) for w, r in runs.items())
    chart.line(points)
    chart.dots(points)
    for workers, speedup in points:
        chart.label(workers, speedup, f"{speedup:.2f}x", anchor="end", dx=-7, dy=-5)
    chart.save("pool_speedup.svg")
    interval_histogram(
        {"intervals": runs[4]["intervals"], "summary": runs[4]},
        "pool_histogram.svg",
        f"Block intervals with a pool of 4 workers ({data['blocks']} blocks)",
    )


if __name__ == "__main__":
    interval_histogram(
        "pow_intervals", "intervals.svg", "Block intervals, one worker, fixed difficulty (200 blocks)"
    )
    difficulty_plot()
    slashing_plot()
    leak_plot()
    reorg_sweep_plot()
    pool_plots()
