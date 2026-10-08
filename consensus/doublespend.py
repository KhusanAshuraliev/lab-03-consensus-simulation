"""Bonus E1: the double-spend race of section 11 of the Bitcoin whitepaper.

An attacker with hash share q waits until the merchant has seen z confirmations
and then tries to overtake the honest chain with a private one.
"""

from __future__ import annotations

import math
import random


def catch_up_probability(q: float, deficit: int) -> float:
    """Gambler's ruin: chance of ever making up `deficit` blocks."""
    p = 1.0 - q
    if deficit <= 0 or q >= p:
        return 1.0
    return (q / p) ** deficit


def attacker_success(q: float, z: int) -> float:
    """Nakamoto's formula.

    While the honest chain mines z blocks, the attacker's progress is taken as
    Poisson with mean lambda = z*q/p; from k blocks it still has to make up
    z - k.
    """
    p = 1.0 - q
    lam = z * q / p
    total = 1.0
    poisson = math.exp(-lam)  # P(k = 0)
    for k in range(z + 1):
        total -= poisson * (1.0 - (q / p) ** (z - k))
        poisson *= lam / (k + 1)
    return total


def confirmations_needed(q: float, limit: float = 0.001) -> int:
    """Smallest z for which the attacker succeeds with probability below `limit`."""
    z = 0
    while attacker_success(q, z) >= limit:
        z += 1
    return z


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's method; fine for the small means used here."""
    threshold = math.exp(-lam)
    k, product = 0, rng.random()
    while product > threshold:
        k += 1
        product *= rng.random()
    return k


def simulate_attack(q: float, z: int, trials: int = 100_000, seed: int = 1) -> float:
    """Monte Carlo estimate of the same race.

    Per trial: draw the attacker's head start from the Poisson distribution,
    then play the race block by block (attacker wins a block with probability
    q) until the attacker has caught up or is hopelessly behind.
    """
    rng = random.Random(seed)
    lam = z * q / (1.0 - q)
    give_up = z + 60  # (q/p)**60 is negligible for every q used here
    wins = 0
    for _ in range(trials):
        deficit = z - _poisson(rng, lam)
        while 0 < deficit < give_up:
            deficit += -1 if rng.random() < q else 1
        if deficit <= 0:
            wins += 1
    return wins / trials
