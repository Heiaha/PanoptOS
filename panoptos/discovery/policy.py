"""Active-discovery policy: probability matching over Bayesian utility estimates.

For each Hiscores table ``k`` we maintain two scalar estimates and combine them
into a per-arm utility ``û_k = θ̂_k · ρ̂_k``. Tables are sampled with probability

    p_k = (1 − ε) · û_k / Σ û_j  +  ε / K

(ε-mixed probability matching), so every arm keeps a non-zero pull rate
regardless of its current utility. Page selection within a table uses a
Beta(2, 1) draw scaled to ``max_page``, biasing toward later (lower-ranked)
pages — where accounts have less cumulative XP and are typically earlier in
their lifecycle, so discovering them there maximizes the post-discovery
observation window.

**Ban rate.** ``θ̂_k`` is a Laplace-smoothed ratio over decay-weighted
observations,

    θ̂_k = (1 + W⁺_k) / (2 + W⁺_k + W⁻_k),

where ``W⁺_k`` and ``W⁻_k`` are sums of banned and non-banned discoveries
weighted by ``exp(−(age − δ) / τ)``, with observations younger than ``δ``
excluded so bans have time to materialize before they enter the estimate.
In the production system ``W⁺_k`` / ``W⁻_k`` are recomputed periodically
from the discovery database; here we expose ``set_ban_counts`` for that
pattern, and an additive ``update_posterior`` for use against the offline
synthetic environment.

**Discovery rate.** ``ρ̂_k`` is a discount-smoothed running ratio,

    a_k ← γ a_k + 1[new],   b_k ← γ b_k + 1[¬new]
    ρ̂_k = a_k / (a_k + b_k),

with ``γ = 0.996`` giving an effective averaging window of ``1/(1 − γ) ≈ 250``
fetches. We use a uniform ``Beta(1, 1)`` prior, initializing every arm at
``a_k = b_k = 1`` so the initial estimate is ``ρ̂_k = 1/2`` regardless of
table size, matching the Laplace offset on ``θ̂_k``.

This file is the *decision logic* extracted from the production discover
worker. The orchestration (live HTTP fetches, queue management, database
writes, periodic recomputation of ``W⁺ / W⁻``) lives in a private repo; here
we expose only the algorithmic core, which is what the technical report
describes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Table:
    """A Hiscores table identified by an opaque key."""

    key: str
    max_page: int


class DiscoveryPolicy:
    """Probability matching over Bayesian utility estimates.

    Parameters
    ----------
    gamma:
        Discount factor for the discovery-rate update. Effective averaging
        window is ``1 / (1 − γ)``. Production uses ``0.996`` (≈ 250 fetches).
    epsilon:
        Uniform exploration floor in the ε-mixed sampling rule. Production
        uses ``0.02``.
    """

    def __init__(
        self,
        gamma: float = 0.996,
        epsilon: float = 0.02,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        # Decay-weighted ban / non-ban pseudo-counts (W⁺, W⁻).
        self.banned_counts: dict[str, float] = {}
        self.normal_counts: dict[str, float] = {}
        # Discount-weighted discovery counts (a_k, b_k).
        self.discovery_a: dict[str, float] = {}
        self.discovery_b: dict[str, float] = {}

    # ── arm management ──────────────────────────────────────────────────────

    def register(self, key: str) -> None:
        """Register a table and seed its prior pseudo-counts.

        Ban counts start at zero (so ``θ̂_k = 1/2`` until evidence arrives).
        Discovery counts start at the uniform ``Beta(1, 1)`` prior
        ``a_k = b_k = 1``, giving an initial ``ρ̂_k = 1/2`` at effective
        sample size two regardless of table size.
        """
        self.banned_counts.setdefault(key, 0.0)
        self.normal_counts.setdefault(key, 0.0)
        self.discovery_a[key] = 1.0
        self.discovery_b[key] = 1.0

    # ── observations ────────────────────────────────────────────────────────

    def set_ban_counts(self, key: str, banned: float, normal: float) -> None:
        """Replace the decay-weighted ban counts ``(W⁺, W⁻)`` for an arm.

        Mirrors the production pattern of recomputing ``W⁺ / W⁻`` periodically
        from a database snapshot using ``exp(−(age − δ) / τ)`` weights.
        """
        self.banned_counts[key] = banned
        self.normal_counts[key] = normal

    def update_posterior(self, key: str, *, banned: float = 0, normal: float = 0) -> None:
        """Additively increment ban counts.

        Convenience for the offline synthetic environment, which observes one
        outcome per pull and has no notion of time decay. Production uses
        ``set_ban_counts`` instead.
        """
        self.banned_counts[key] = self.banned_counts.get(key, 0.0) + banned
        self.normal_counts[key] = self.normal_counts.get(key, 0.0) + normal

    def record_discovery(self, key: str, *, success: bool) -> None:
        """Discount-weighted update on the discovery-rate counts."""
        a = self.discovery_a.get(key, 0.0) * self.gamma
        b = self.discovery_b.get(key, 0.0) * self.gamma
        if success:
            a += 1.0
        else:
            b += 1.0
        self.discovery_a[key] = a
        self.discovery_b[key] = b

    # ── estimators ──────────────────────────────────────────────────────────

    def ban_rate(self, key: str) -> float:
        """Laplace-smoothed ban-rate estimate ``θ̂_k``."""
        w_pos = self.banned_counts.get(key, 0.0)
        w_neg = self.normal_counts.get(key, 0.0)
        return (1.0 + w_pos) / (2.0 + w_pos + w_neg)

    def discovery_rate(self, key: str) -> float:
        """Discount-smoothed discovery-rate estimate ``ρ̂_k``."""
        a = self.discovery_a.get(key, 0.0)
        b = self.discovery_b.get(key, 0.0)
        total = a + b
        if total <= 0:
            return 0.0
        return a / total

    def utilities(self, tables: list[Table]) -> dict[str, float]:
        """Per-arm utility ``û_k = θ̂_k · ρ̂_k``."""
        return {t.key: self.ban_rate(t.key) * self.discovery_rate(t.key) for t in tables}

    def probabilities(self, tables: list[Table]) -> dict[str, float]:
        """ε-mixed sampling probabilities ``p_k``.

        Falls back to the uniform distribution when total utility is
        non-positive (e.g. before any observations have arrived).
        """
        u = self.utilities(tables)
        total = sum(u.values())
        k = len(tables)
        if total <= 0:
            return {t.key: 1.0 / k for t in tables}
        return {
            t.key: (1.0 - self.epsilon) * u[t.key] / total + self.epsilon / k
            for t in tables
        }

    # ── sampling ────────────────────────────────────────────────────────────

    def sample_table(self, tables: list[Table], rng: random.Random | None = None) -> Table:
        """Sample a table by ε-mixed probability matching."""
        rng = rng or random
        p = self.probabilities(tables)
        return rng.choices(tables, weights=[p[t.key] for t in tables])[0]

    def sample_page(self, table: Table, rng: random.Random | None = None) -> int:
        """Draw a page within ``table``, biased toward later (lower-rank) pages.

        Uses a Beta(2, 1) draw, which has PDF ``f(x) = 2x`` on [0, 1] (mean 2/3,
        monotonically increasing) — so deeper pages get more mass. The point is
        not that deeper pages are bot-richer (though they often are) but that
        accounts there are earlier in their lifecycle: discovering an account
        when it has less cumulative XP gives the model a longer post-discovery
        observation window before the account is either enforced against or
        ages out of the model's relevant horizon.
        """
        rng = rng or random
        return int(rng.betavariate(2, 1) * table.max_page) + 1

    def select(
        self, tables: list[Table], rng: random.Random | None = None
    ) -> tuple[Table, int]:
        """Convenience: draw a table and a page in one call."""
        rng = rng or random
        table = self.sample_table(tables, rng=rng)
        return table, self.sample_page(table, rng=rng)
