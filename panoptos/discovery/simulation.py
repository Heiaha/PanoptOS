"""Synthetic environment for evaluating the discovery policy offline.

This is a deliberately minimal stand-in for the real Hiscores. Each table is
modeled as a finite population of accounts; pulling a (table, page) yields
one slot from that page, and the simulation draws a fresh label (bot vs
human) for newly-discovered accounts. Two effects are modeled:

* **Per-table base bot rate** — different skills attract different amounts of
  automation. (e.g. agility, thieving, hunter are bot-heavy; construction is
  not.)
* **Page-rank dependence** — within a table, lower-ranked accounts (higher
  page numbers) have higher bot rates than the top of the leaderboard. This
  is what makes the bandit's *page* sampling and the discovery-rate
  saturation term matter: top pages saturate fast but yield more humans;
  deeper pages stay fresh but the bot rate climbs.

Two policies are provided for comparison:

* `BanditPolicy` — wraps `DiscoveryPolicy` from `policy.py`.
* `UniformPolicy` — picks a table uniformly at random and a page uniformly
  within its `max_page`. This is the natural "no-learning" baseline; the
  bandit's value is the gap between its discovery curve and this one.

Running the module as a script (`python -m panoptos.discovery.simulation`)
prints a side-by-side comparison and a small efficiency ratio.

None of this code talks to Jagex.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from panoptos.discovery.policy import DiscoveryPolicy, Table


# ── synthetic environment ───────────────────────────────────────────────────


@dataclass
class SyntheticTable:
    """A simulated Hiscores table.

    Parameters
    ----------
    key:
        Table identifier.
    population:
        Total accounts on the table.
    base_bot_rate:
        Bot fraction at the very top of the leaderboard.
    rank_bot_slope:
        Additive increase in bot fraction at the bottom of the leaderboard.
        Bot rate at page p is ``base_bot_rate + rank_bot_slope * (p / max_page)``,
        clipped to [0, 1].
    """

    key: str
    population: int
    base_bot_rate: float
    rank_bot_slope: float = 0.0
    seen: set[int] = field(default_factory=set)
    labels: dict[int, bool] = field(default_factory=dict)  # account_id → is_bot

    @property
    def max_page(self) -> int:
        return max(1, self.population // 25)

    def bot_rate_at_page(self, page: int) -> float:
        return min(1.0, max(0.0, self.base_bot_rate + self.rank_bot_slope * (page / self.max_page)))


# ── policies ────────────────────────────────────────────────────────────────


class Policy(Protocol):
    """Anything that can pick a (table, page) given a list of tables."""

    def select(
        self, tables: list[Table], rng: random.Random | None = None
    ) -> tuple[Table, int]: ...

    def update_posterior(self, key: str, *, banned: int = 0, normal: int = 0) -> None: ...

    def record_discovery(self, key: str, *, success: bool) -> None: ...


class BanditPolicy:
    """Thin wrapper exposing `DiscoveryPolicy` through the `Policy` protocol."""

    def __init__(self, gamma: float = 0.996, epsilon: float = 0.02) -> None:
        self.inner = DiscoveryPolicy(gamma=gamma, epsilon=epsilon)

    def register(self, key: str) -> None:
        self.inner.register(key)

    def select(
        self, tables: list[Table], rng: random.Random | None = None
    ) -> tuple[Table, int]:
        return self.inner.select(tables, rng=rng)

    def update_posterior(self, key: str, *, banned: int = 0, normal: int = 0) -> None:
        self.inner.update_posterior(key, banned=banned, normal=normal)

    def record_discovery(self, key: str, *, success: bool) -> None:
        self.inner.record_discovery(key, success=success)


class UniformPolicy:
    """Picks a table uniformly at random and a page uniformly within it.

    Ignores all observations — the natural no-learning baseline.
    """

    def select(
        self, tables: list[Table], rng: random.Random | None = None
    ) -> tuple[Table, int]:
        rng = rng or random
        table = rng.choice(tables)
        page = rng.randint(1, table.max_page)
        return table, page

    def update_posterior(self, key: str, *, banned: int = 0, normal: int = 0) -> None:
        pass

    def record_discovery(self, key: str, *, success: bool) -> None:
        pass


# ── simulation harness ──────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    steps: int
    discovered_bots: int
    discovered_humans: int
    per_table_pulls: dict[str, int]
    discovery_curve: list[int]  # cumulative bots discovered, indexed by step

    @property
    def bots_per_pull(self) -> float:
        return self.discovered_bots / max(1, self.steps)


def run_simulation(
    tables: list[SyntheticTable],
    policy: Policy,
    steps: int,
    seed: int = 0,
) -> SimulationResult:
    """Run `policy` against the synthetic environment for `steps` pulls."""
    rng = random.Random(seed)
    policy_tables = [Table(key=t.key, max_page=t.max_page) for t in tables]

    if isinstance(policy, BanditPolicy):
        for t in tables:
            policy.register(t.key)

    by_key = {t.key: t for t in tables}
    pulls: dict[str, int] = {t.key: 0 for t in tables}
    discovered_bots = 0
    discovered_humans = 0
    curve: list[int] = []

    for _ in range(steps):
        chosen, page = policy.select(policy_tables, rng=rng)
        env_table = by_key[chosen.key]

        # An account is identified by (page, slot). Labels are sticky once drawn.
        slot = rng.randrange(25)
        account_id = page * 25 + slot

        is_new = account_id not in env_table.seen
        env_table.seen.add(account_id)

        if account_id not in env_table.labels:
            env_table.labels[account_id] = rng.random() < env_table.bot_rate_at_page(page)
        is_bot = env_table.labels[account_id]

        if is_new:
            if is_bot:
                discovered_bots += 1
                policy.update_posterior(chosen.key, banned=1)
            else:
                discovered_humans += 1
                policy.update_posterior(chosen.key, normal=1)

        policy.record_discovery(chosen.key, success=is_new)
        pulls[chosen.key] += 1
        curve.append(discovered_bots)

    return SimulationResult(
        steps=steps,
        discovered_bots=discovered_bots,
        discovered_humans=discovered_humans,
        per_table_pulls=pulls,
        discovery_curve=curve,
    )


# ── demo ────────────────────────────────────────────────────────────────────


def _default_tables() -> list[SyntheticTable]:
    """A toy population that's heterogeneous enough for the bandit to matter.

    Mixes bot-friendly skills (agility, hunter, thieving) against skills and
    endgame boss tables that bots cannot meaningfully participate in. The
    boss tables in particular are small and have near-zero bot rates because
    the content is mechanically gated (gear, quest progression, group
    coordination, sub-second reactions). A uniform sampler wastes pulls
    there; the bandit learns to ignore them within a few hundred steps.
    """
    return [
        # bot-heavy skills
        SyntheticTable("agility",            population=20_000, base_bot_rate=0.30, rank_bot_slope=0.30),
        SyntheticTable("hunter",             population=30_000, base_bot_rate=0.20, rank_bot_slope=0.25),
        SyntheticTable("thieving",           population=15_000, base_bot_rate=0.40, rank_bot_slope=0.30),
        # bot-light skills
        SyntheticTable("construction",       population=10_000, base_bot_rate=0.03, rank_bot_slope=0.05),
        SyntheticTable("overall",            population=80_000, base_bot_rate=0.08, rank_bot_slope=0.10),
        # endgame bosses — small populations, near-zero bot rate
        SyntheticTable("chambers_of_xeric",  population=4_000,  base_bot_rate=0.005, rank_bot_slope=0.005),
        SyntheticTable("tombs_of_amascut",   population=3_500,  base_bot_rate=0.005, rank_bot_slope=0.005),
        SyntheticTable("theatre_of_blood",   population=2_500,  base_bot_rate=0.005, rank_bot_slope=0.005),
        SyntheticTable("vorkath",            population=8_000,  base_bot_rate=0.02,  rank_bot_slope=0.02),
        SyntheticTable("zulrah",             population=12_000, base_bot_rate=0.04,  rank_bot_slope=0.03),
        SyntheticTable("inferno",            population=1_500,  base_bot_rate=0.001, rank_bot_slope=0.001),
    ]


def _print_result(name: str, result: SimulationResult) -> None:
    print(f"\n{name}")
    print(f"  bots discovered:   {result.discovered_bots:,}")
    print(f"  humans discovered: {result.discovered_humans:,}")
    print(f"  bots per pull:     {result.bots_per_pull:.3f}")
    print("  pulls per table:")
    for key, n in sorted(result.per_table_pulls.items(), key=lambda kv: -kv[1]):
        print(f"    {key:14s} {n:6,}")


if __name__ == "__main__":
    STEPS = 50_000
    SEED = 0

    bandit = run_simulation(_default_tables(), BanditPolicy(), steps=STEPS, seed=SEED)
    uniform = run_simulation(_default_tables(), UniformPolicy(), steps=STEPS, seed=SEED)

    _print_result("Bandit", bandit)
    _print_result("Uniform baseline", uniform)

    ratio = bandit.discovered_bots / max(1, uniform.discovered_bots)
    print(f"\nBandit / Uniform bot-discovery ratio: {ratio:.2f}×")
    print(
        f"(After {STEPS:,} pulls, the bandit found {bandit.discovered_bots:,} bots "
        f"vs {uniform.discovered_bots:,} for uniform sampling.)"
    )
