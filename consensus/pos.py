"""Proof of stake: Casper FFG justification and finality, slashing, inactivity leak.

A vote ("attestation") is a link from a source checkpoint to a target
checkpoint. The state machine below counts stake per link once per epoch and
applies four rules in a fixed order: slashing, justification, finalisation,
inactivity leak. Slashing comes first so that a validator caught equivocating
no longer counts towards the quorum it was trying to build.
"""

from __future__ import annotations

from dataclasses import dataclass

GWEI = 10**9
MAX_EFFECTIVE_BALANCE = 32 * GWEI

#: The immediate penalty is 1/32 of the effective balance.
INITIAL_SLASHING_QUOTIENT = 32

#: Factor in the correlated penalty: min(1, 3 * slashed stake / total stake).
PROPORTIONAL_SLASHING_MULTIPLIER = 3

#: The leak starts once finality has not advanced for more than this many epochs.
MIN_EPOCHS_TO_INACTIVITY_PENALTY = 4

#: An absent validator loses 1/64 of its effective balance per leaking epoch.
INACTIVITY_PENALTY_QUOTIENT = 64


@dataclass(frozen=True)
class Checkpoint:
    epoch: int
    root: str

    def __str__(self) -> str:
        return f"({self.epoch}, {self.root})"


@dataclass(frozen=True)
class Attestation:
    validator: int
    source: Checkpoint
    target: Checkpoint


@dataclass
class Validator:
    index: int
    balance: int = MAX_EFFECTIVE_BALANCE
    slashed: bool = False
    slashed_epoch: int | None = None

    @property
    def is_active(self) -> bool:
        return not self.slashed and self.balance > 0

    @property
    def effective_balance(self) -> int:
        return min(self.balance, MAX_EFFECTIVE_BALANCE)


@dataclass
class SlashingRecord:
    validator: int
    epoch: int
    rule: str
    initial_penalty: int
    correlated_penalty: int = 0

    @property
    def total_penalty(self) -> int:
        return self.initial_penalty + self.correlated_penalty


# --------------------------------------------------------------------------- #
# The two slashing conditions
# --------------------------------------------------------------------------- #


def is_double_vote(a: Attestation, b: Attestation) -> bool:
    """Same validator, same target epoch, different target root."""
    return (
        a.validator == b.validator
        and a.target.epoch == b.target.epoch
        and a.target.root != b.target.root
    )


def is_surround_vote(a: Attestation, b: Attestation) -> bool:
    """`a` strictly surrounds `b`: a.source < b.source and b.target < a.target."""
    return (
        a.validator == b.validator
        and a.source.epoch < b.source.epoch
        and b.target.epoch < a.target.epoch
    )


def violated_rule(a: Attestation, b: Attestation) -> str | None:
    if is_double_vote(a, b):
        return "double_vote"
    if is_surround_vote(a, b) or is_surround_vote(b, a):
        return "surround_vote"
    return None


def find_violations(attestations: list[Attestation]) -> list[tuple[Attestation, Attestation, str]]:
    """Every slashable pair in a list of attestations, with the rule it breaks."""
    by_validator: dict[int, list[Attestation]] = {}
    found: list[tuple[Attestation, Attestation, str]] = []
    for attestation in attestations:
        earlier_votes = by_validator.setdefault(attestation.validator, [])
        for earlier in earlier_votes:
            rule = violated_rule(earlier, attestation)
            if rule:
                found.append((earlier, attestation, rule))
        earlier_votes.append(attestation)
    return found


# --------------------------------------------------------------------------- #
# The state machine
# --------------------------------------------------------------------------- #


class BeaconState:
    def __init__(self, validator_count: int, genesis_root: str = "genesis") -> None:
        self.validators = [Validator(i) for i in range(validator_count)]
        self.genesis_checkpoint = Checkpoint(0, genesis_root)
        self.justified: set[Checkpoint] = {self.genesis_checkpoint}
        self.current_justified = self.genesis_checkpoint
        self.finalized = self.genesis_checkpoint
        self.current_epoch = 0
        #: Epoch in which the finalised checkpoint last moved.
        self.last_finality_epoch = 0

        self._votes: dict[int, list[Attestation]] = {}
        #: Slashable pairs noticed when the second vote arrived, not yet punished.
        self._evidence: list[tuple[Attestation, Attestation, str]] = []
        self.slashings: list[SlashingRecord] = []
        self.slashed_stake = 0

    # -- stake ---------------------------------------------------------------- #

    @property
    def total_active_balance(self) -> int:
        return sum(v.effective_balance for v in self.validators if v.is_active)

    def supermajority_threshold(self) -> int:
        """A link needs strictly more stake than this value (2/3 of active stake)."""
        return 2 * self.total_active_balance // 3

    def stake_share(self, indices: list[int]) -> float:
        """Fraction of the active stake held by the listed validators."""
        total = self.total_active_balance
        held = sum(
            self.validators[i].effective_balance for i in indices if self.validators[i].is_active
        )
        return held / total if total else 0.0

    # -- votes ---------------------------------------------------------------- #

    def attest(self, attestation: Attestation) -> None:
        if not self.validators[attestation.validator].is_active:
            return
        history = self._votes.setdefault(attestation.validator, [])
        if attestation in history:
            return
        for earlier in history:
            rule = violated_rule(earlier, attestation)
            if rule:
                self._evidence.append((earlier, attestation, rule))
        history.append(attestation)

    def attest_many(self, attestations: list[Attestation]) -> None:
        for attestation in attestations:
            self.attest(attestation)

    @property
    def attestation_log(self) -> list[Attestation]:
        return [a for history in self._votes.values() for a in history]

    # -- one epoch ------------------------------------------------------------ #

    def process_epoch(self, epoch: int) -> dict[str, object]:
        self.current_epoch = epoch
        slashed = self._slash(epoch)
        justified, finalized = self._justify_and_finalise(epoch)
        leaked = self._leak(epoch)
        return {
            "epoch": epoch,
            "slashed": slashed,
            "justified": justified,
            "finalized": finalized,
            "inactivity_leak": leaked,
        }

    def _slash(self, epoch: int) -> list[SlashingRecord]:
        """Initial penalty for each new offender, then the correlated penalty."""
        new_records: list[SlashingRecord] = []
        for first, _second, rule in self._evidence:
            validator = self.validators[first.validator]
            if validator.slashed:
                continue  # one offence is enough, a validator is slashed once
            initial = validator.effective_balance // INITIAL_SLASHING_QUOTIENT
            validator.balance -= initial
            validator.slashed = True
            validator.slashed_epoch = epoch
            self.slashed_stake += validator.effective_balance
            record = SlashingRecord(validator.index, epoch, rule, initial)
            new_records.append(record)
            self.slashings.append(record)
        self._evidence.clear()

        if new_records:
            # Stake that could still vote plus stake already slashed.
            total = self.total_active_balance + self.slashed_stake
            scaled = min(PROPORTIONAL_SLASHING_MULTIPLIER * self.slashed_stake, total)
            for record in new_records:
                validator = self.validators[record.validator]
                extra = min(validator.effective_balance * scaled // total, validator.balance)
                validator.balance -= extra
                record.correlated_penalty = extra
        return new_records

    def _justify_and_finalise(self, epoch: int) -> tuple[Checkpoint | None, Checkpoint | None]:
        """Count stake per link whose target lies in this epoch.

        Only links that start at an already justified checkpoint are counted.
        A validator contributes to a link at most once.
        """
        weight: dict[tuple[Checkpoint, Checkpoint], int] = {}
        for index, history in self._votes.items():
            validator = self.validators[index]
            if not validator.is_active:
                continue
            for link in {(a.source, a.target) for a in history if a.target.epoch == epoch}:
                if link[0] in self.justified:
                    weight[link] = weight.get(link, 0) + validator.effective_balance

        total = self.total_active_balance
        justified = finalized = None
        for (source, target), stake in weight.items():
            if 3 * stake <= 2 * total:  # exactly two thirds is not enough
                continue
            self.justified.add(target)
            justified = target
            if target.epoch > self.current_justified.epoch:
                self.current_justified = target
            # Second phase: the source becomes final when its direct successor
            # is justified by a link from it.
            if target.epoch == source.epoch + 1 and source.epoch > self.finalized.epoch:
                self.finalized = source
                self.last_finality_epoch = epoch
                finalized = source
        return justified, finalized

    def _leak(self, epoch: int) -> int:
        """Drain validators that did not vote, once finality has stalled long enough."""
        if epoch - self.last_finality_epoch <= MIN_EPOCHS_TO_INACTIVITY_PENALTY:
            return 0
        leaked = 0
        for validator in self.validators:
            if not validator.is_active:
                continue
            voted = any(a.target.epoch == epoch for a in self._votes.get(validator.index, ()))
            if voted:
                continue
            penalty = min(
                validator.effective_balance // INACTIVITY_PENALTY_QUOTIENT, validator.balance
            )
            validator.balance -= penalty
            leaked += penalty
        return leaked

    # -- display -------------------------------------------------------------- #

    def summary(self) -> str:
        active = sum(1 for v in self.validators if v.is_active)
        slashed = sum(1 for v in self.validators if v.slashed)
        lines = [
            f"epoch              : {self.current_epoch}",
            f"validators         : {len(self.validators)} ({active} active, {slashed} slashed)",
            f"total active stake : {self.total_active_balance / GWEI:.2f} ETH",
            f"current justified  : {self.current_justified}",
            f"finalized          : {self.finalized}",
        ]
        for r in self.slashings:
            lines.append(
                f"  slashed validator {r.validator:>2} for {r.rule} in epoch {r.epoch}: "
                f"-{r.total_penalty / GWEI:.4f} ETH (initial {r.initial_penalty / GWEI:.4f}"
                f" + correlated {r.correlated_penalty / GWEI:.4f})"
            )
        return "\n".join(lines)


def honest_attestations(
    state: BeaconState,
    source: Checkpoint,
    target: Checkpoint,
    validators: list[int] | None = None,
) -> list[Attestation]:
    """The same link from every listed validator (default: all active ones)."""
    if validators is None:
        validators = [v.index for v in state.validators if v.is_active]
    return [Attestation(index, source, target) for index in validators]


# --------------------------------------------------------------------------- #
# Experiments used by the demo and the report
# --------------------------------------------------------------------------- #


def slashing_loss(offenders: int, validator_count: int = 64) -> int:
    """Loss in gwei of one offender when `offenders` validators double-vote together."""
    state = BeaconState(validator_count=validator_count)
    source = state.genesis_checkpoint
    state.attest_many(honest_attestations(state, source, Checkpoint(1, "chain-A")))
    for index in range(offenders):
        state.attest(Attestation(index, source, Checkpoint(1, "chain-B")))
    before = state.validators[0].balance
    state.process_epoch(1)
    return before - state.validators[0].balance


def inactivity_leak_trace(
    epochs: int = 100, validator_count: int = 64, absent_count: int = 40
) -> list[dict[str, object]]:
    """One row per epoch while `absent_count` validators stay offline.

    The validators that are present always link from the latest justified
    checkpoint, so that their votes can justify again once they hold more than
    two thirds of the remaining stake.
    """
    state = BeaconState(validator_count=validator_count)
    present = list(range(absent_count, validator_count))
    rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        target = Checkpoint(epoch, f"root-{epoch}")
        state.attest_many(honest_attestations(state, state.current_justified, target, present))
        result = state.process_epoch(epoch)
        rows.append(
            {
                "epoch": epoch,
                "voting_share": state.stake_share(present),
                "leaked_gwei": result["inactivity_leak"],
                "justified": str(result["justified"]) if result["justified"] else None,
                "finalized": str(result["finalized"]) if result["finalized"] else None,
                "absent_balance_gwei": state.validators[0].balance,
                "present_balance_gwei": state.validators[present[0]].balance,
            }
        )
    return rows
