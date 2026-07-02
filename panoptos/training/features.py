"""Feature engineering for player snapshot sequences."""

import datetime
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from constants import (
        COMBAT_INDICES,
        GATHERING_INDICES,
        KC_ACTIVITY_INDICES,
        MAX_SEQUENCE_LENGTH,
        MIN_OVERALL_XP,
        MIN_SEQUENCE_LENGTH,
        OVERALL_INDEX,
        RAW_FEATURES,
        SKILL_INDICES,
        encode,
    )
except ImportError:
    from panoptos.training.constants import (
        COMBAT_INDICES,
        GATHERING_INDICES,
        KC_ACTIVITY_INDICES,
        MAX_SEQUENCE_LENGTH,
        MIN_OVERALL_XP,
        MIN_SEQUENCE_LENGTH,
        OVERALL_INDEX,
        RAW_FEATURES,
        SKILL_INDICES,
        encode,
    )


@dataclass
class Skipped:
    """Feature extraction was skipped due to insufficient data."""

    reason: str


@dataclass
class Computed:
    """Successfully computed features."""

    features: np.ndarray
    label: np.float32
    name: np.ndarray
    elapsed_days: np.ndarray


FeatureResult = Skipped | Computed


@dataclass
class PlayerSequence:
    """A single player's sequence of snapshots with label.

    `timestamps` and `discovered_at` are epoch seconds (int/float).
    """

    name: np.ndarray
    states: np.ndarray
    timestamps: np.ndarray
    discovered_at: float
    label: int

    @property
    def length(self) -> int:
        return len(self.timestamps)

    @classmethod
    def from_snapshots(
        cls,
        name: str,
        snapshots: list[Mapping[str, Any]],
        discovered_at: datetime.datetime,
        label: int = 0,
    ) -> "PlayerSequence":
        """Build a PlayerSequence from row dicts (e.g. DB snapshots)."""
        states = np.array(
            [[snap[col] for col in RAW_FEATURES] for snap in snapshots],
            dtype=np.float64,
        )
        timestamps = np.fromiter(
            (snap["captured_at"].timestamp() for snap in snapshots),
            dtype=np.float64,
            count=len(snapshots),
        )
        return cls(
            name=np.array(encode(name), dtype=np.uint8),
            states=states,
            timestamps=timestamps,
            discovered_at=discovered_at.timestamp(),
            label=label,
        )

    def __post_init__(self) -> None:
        if self.states.shape[0] != len(self.timestamps):
            raise ValueError(
                f"{self.name}: states len {self.states.shape[0]} != "
                f"timestamps len {len(self.timestamps)}"
            )
        if len(self.timestamps) > 1 and (np.diff(self.timestamps) < 0).any():
            raise ValueError(f"{self.name}: snapshots are not ordered by captured_at")

    def _truncate(
        self,
        prefix_days: float | None,
        training: bool,
        snapshot_dropout: float = 0.0,
    ) -> np.ndarray | None:
        """Select the snapshot indices to keep.

        Returns None if insufficient data.
        """
        total = len(self.timestamps)

        if total < MIN_SEQUENCE_LENGTH:
            return None

        if training:
            # Per-position supervision covers every prefix length within the
            # window, so only the window placement needs to be random.
            length = min(total, MAX_SEQUENCE_LENGTH)
            start = random.randint(0, total - length)
            window = np.arange(start, start + length)
            if snapshot_dropout > 0:
                window = self._drop_snapshots(window, snapshot_dropout)
            return window

        if prefix_days is not None:
            cutoff = self.discovered_at + prefix_days * 86400.0
            end_idx = int(np.searchsorted(self.timestamps, cutoff, side="right"))
            return np.arange(end_idx) if end_idx >= MIN_SEQUENCE_LENGTH else None

        end_idx = min(total, MAX_SEQUENCE_LENGTH)
        if end_idx < MIN_SEQUENCE_LENGTH:
            return None
        return np.arange(total - end_idx, total)

    @staticmethod
    def _drop_snapshots(window: np.ndarray, max_rate: float) -> np.ndarray:
        """Randomly thin interior snapshots to vary the observed cadence.

        The polling scheduler's cadence is label-correlated, so contiguous
        windows would let the model read the schedule itself as a feature.
        A per-sequence rate drawn from [0, max_rate) decides how many interior
        snapshots to drop; the window's endpoints always survive so the
        elapsed-time span and the XP-based skip checks are preserved.
        """
        rate = random.uniform(0.0, max_rate)
        keep = max(MIN_SEQUENCE_LENGTH, round(len(window) * (1.0 - rate)))
        if keep >= len(window):
            return window

        interior = random.sample(range(1, len(window) - 1), keep - 2)
        return window[sorted([0, len(window) - 1, *interior])]

    @staticmethod
    def compute_time_deltas(timestamps: np.ndarray) -> np.ndarray:
        """Compute time deltas (seconds) between consecutive snapshots.

        deltas[0] has no preceding interval and is 0; the velocity
        functions only read hours[1:], so it never feeds a feature.
        """
        seq_len = len(timestamps)
        deltas = np.zeros(seq_len)

        if seq_len > 1:
            deltas[1:] = np.diff(timestamps)
        return deltas

    @staticmethod
    def compute_velocities(
        skill_diffs: np.ndarray,
        time_deltas: np.ndarray,
        seq_len: int,
        min_hours: float = 0.1,
    ) -> np.ndarray:
        """Compute XP per hour for each skill."""
        hours = np.maximum(time_deltas / 3600, min_hours)

        velocities = np.zeros((seq_len, len(SKILL_INDICES)))
        if seq_len > 1:
            velocities[1:] = skill_diffs / hours[1:, np.newaxis]

        return velocities

    @staticmethod
    def compute_ratios(
        skills_xp: np.ndarray,
        raw: np.ndarray,
        eps: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute skill ratios and combat/gathering ratio."""
        total_xp = raw[:, OVERALL_INDEX] + eps
        combat_xp = raw[:, COMBAT_INDICES].sum(axis=1)
        gathering_xp = raw[:, GATHERING_INDICES].sum(axis=1) + eps

        skill_ratios = skills_xp / total_xp[:, np.newaxis]
        combat_gathering_ratio = np.log1p(combat_xp / gathering_xp)

        return skill_ratios, combat_gathering_ratio

    @staticmethod
    def compute_entropy(
        skills_xp: np.ndarray,
        raw: np.ndarray,
        eps: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute skill entropy and max skill ratio."""
        total_xp = raw[:, OVERALL_INDEX] + eps
        n_skills = skills_xp.shape[1]

        skill_probs = skills_xp / total_xp[:, np.newaxis]
        skill_entropy = -np.sum(
            skill_probs * np.log(skill_probs + eps), axis=1
        ) / np.log(n_skills + eps)

        max_skill_ratio = skills_xp.max(axis=1) / total_xp

        return skill_entropy, max_skill_ratio

    @staticmethod
    def compute_gain_entropy(
        skill_gains: np.ndarray,
        n_skills: int,
        seq_len: int,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Compute entropy of XP gain distribution across skills per interval.

        Unlike skill entropy (which measures cumulative XP spread), this captures
        how focused or diverse training is *within each interval*. Bots tend to
        produce near-zero entropy (all gains in one skill); humans spread gains
        across multiple skills.
        """
        gain_entropy = np.zeros(seq_len)
        if seq_len > 1:
            total_gains = skill_gains.sum(axis=1, keepdims=True) + eps
            probs = skill_gains / total_gains
            gain_entropy[1:] = -np.sum(probs * np.log(probs + eps), axis=1) / np.log(
                n_skills + eps
            )

        return gain_entropy

    @staticmethod
    def compute_activity_counts(
        skills_xp: np.ndarray,
        activities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute activity engagement metrics."""
        n_skills = (skills_xp > 0).sum(axis=1)
        n_activities = (activities > 0).sum(axis=1)

        return n_skills, n_activities

    @staticmethod
    def compute_activity_concentration(
        raw: np.ndarray,
        eps: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute activity entropy and max activity ratio over cumulative KC.

        Uses only KC-based activities, excluding rank/score activities
        (e.g. pvp_arena_rank) which have non-zero defaults and different scales.
        """
        activities = raw[:, KC_ACTIVITY_INDICES]
        total_kc = activities.sum(axis=1) + eps

        probs = activities / total_kc[:, np.newaxis]
        activity_entropy = -np.sum(probs * np.log(probs + eps), axis=1) / np.log(
            activities.shape[1] + eps
        )

        max_activity_ratio = activities.max(axis=1) / total_kc

        return activity_entropy, max_activity_ratio

    @staticmethod
    def compute_active_deltas(
        skill_gains: np.ndarray,
        activity_gains: np.ndarray,
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute number of skills/activities with any gain per interval.

        Unlike n_skills/n_activities (which count cumulative engagement),
        this counts how many skills and activities *changed* in each interval.
        Bots tend to have the same small count every interval; humans vary.
        """
        active_skills = np.zeros(seq_len)
        active_activities = np.zeros(seq_len)

        if seq_len > 1:
            active_skills[1:] = (skill_gains > 0).sum(axis=1)
            active_activities[1:] = (activity_gains > 0).sum(axis=1)

        return active_skills, active_activities

    @staticmethod
    def compute_activity_switch(
        activity_gains: np.ndarray,
        seq_len: int,
    ) -> np.ndarray:
        """Binary flag: did the dominant activity change from previous interval.

        Dominant activity is the one with the highest KC gain. 0 at timesteps
        0-1 (no previous pair to compare). Bots rarely switch; humans do.
        """
        switch = np.zeros(seq_len)

        if seq_len < 3:
            return switch

        dominant = activity_gains.argmax(axis=1)
        has_any = activity_gains.sum(axis=1) > 0

        switched = has_any[1:] & has_any[:-1] & (dominant[1:] != dominant[:-1])
        switch[2:] = switched

        return switch

    @staticmethod
    def compute_activity_streak(
        activity_gains: np.ndarray,
        seq_len: int,
    ) -> np.ndarray:
        """Compute consecutive intervals with the same single active activity.

        At each timestep, tracks how many consecutive intervals the same
        activity has been the only one gaining KC. Bots grinding a single
        boss produce long streaks; humans switch activities.
        """
        streak = np.zeros(seq_len)

        if seq_len < 2:
            return streak

        active_mask = activity_gains > 0
        active_counts = active_mask.sum(axis=1)

        # Dominant activity index per interval (-1 if zero or multiple)
        dominant = np.where(
            active_counts == 1,
            active_mask.argmax(axis=1),
            -1,
        )

        current_streak = 0
        for i in range(len(dominant)):
            if dominant[i] >= 0 and (i == 0 or dominant[i] == dominant[i - 1]):
                current_streak += 1
            else:
                current_streak = 1 if dominant[i] >= 0 else 0
            streak[i + 1] = current_streak

        return streak

    @staticmethod
    def compute_velocity_delta(
        raw: np.ndarray,
        time_deltas: np.ndarray,
        seq_len: int,
        min_hours: float = 0.1,
    ) -> np.ndarray:
        """Signed-log change in total XP/hr between consecutive intervals.

        Bots hold a steady rate (delta ~0); humans are bursty and swing
        in both directions. Compressed as sign(d) * log1p(|d|) to keep
        the scale comparable to the log1p'd velocity features.
        """
        delta = np.zeros(seq_len)

        if seq_len < 3:
            return delta

        hours = np.maximum(time_deltas / 3600, min_hours)
        total_xp_gains = np.maximum(np.diff(raw[:, OVERALL_INDEX]), 0)
        rates = total_xp_gains / hours[1:]

        d = np.diff(rates)
        delta[2:] = np.sign(d) * np.log1p(np.abs(d))

        return delta

    @staticmethod
    def compute_gain_cosine_similarity(
        skill_gains: np.ndarray,
        seq_len: int,
    ) -> np.ndarray:
        """Compute cosine similarity between consecutive XP gain vectors.

        Measures how similar the skill training profile is between adjacent
        intervals. Bots produce nearly identical gain vectors every interval
        (similarity ~1.0); humans vary their training.
        """
        sim = np.zeros(seq_len)

        if seq_len < 3:
            return sim

        a = skill_gains[:-1]
        b = skill_gains[1:]
        norms = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
        valid = norms > 0
        sim[2:][valid] = (a * b).sum(axis=1)[valid] / norms[valid]

        return sim

    def calculate_features(
        self,
        prefix_days: float | None = None,
        training: bool = False,
        snapshot_dropout: float = 0.0,
        eps: float = 1e-12,
    ) -> FeatureResult:
        """Calculate all features for the sequence.

        Output is capped to the trailing MAX_SEQUENCE_LENGTH positions;
        only the prefix_days path can produce longer windows internally.
        """
        idx = self._truncate(prefix_days, training, snapshot_dropout)
        if idx is None:
            return Skipped("insufficient snapshots")

        # Diffs must happen in float64: above ~2^31 total XP, float32 spacing
        # is 256+, so small interval gains would quantize to 0. Derived values
        # are small and cast to float32 safely at the end.
        raw = self.states[idx].astype(np.float64)
        timestamps = self.timestamps[idx]
        seq_len = raw.shape[0]

        if raw[-1, OVERALL_INDEX] < MIN_OVERALL_XP:
            return Skipped("below XP floor")

        if raw[0, OVERALL_INDEX] == raw[-1, OVERALL_INDEX]:
            return Skipped("no XP change observed")

        # Precompute shared intermediates. Count- and gain-based activity
        # features use only KC activities: rank/score activities (e.g.
        # pvp_arena_rank) have non-zero defaults that would count as
        # engagement, and their ratings move both ways, so every upward
        # fluctuation would register as a "gain".
        skills_xp = raw[:, SKILL_INDICES[1:]]
        kc_activities = raw[:, KC_ACTIVITY_INDICES]
        skill_diffs = np.maximum(np.diff(raw[:, SKILL_INDICES], axis=0), 0)
        skill_gains = np.maximum(np.diff(skills_xp, axis=0), 0)
        activity_gains = np.maximum(np.diff(kc_activities, axis=0), 0)

        time_deltas = self.compute_time_deltas(timestamps)
        elapsed_days = np.cumsum(time_deltas) / 86400.0

        velocities = self.compute_velocities(skill_diffs, time_deltas, seq_len)
        skill_ratios, combat_gathering_ratio = self.compute_ratios(skills_xp, raw, eps)
        skill_entropy, max_skill_ratio = self.compute_entropy(skills_xp, raw, eps)
        gain_entropy = self.compute_gain_entropy(
            skill_gains, skills_xp.shape[1], seq_len, eps
        )
        n_skills, n_activities = self.compute_activity_counts(skills_xp, kc_activities)
        activity_entropy, max_activity_ratio = self.compute_activity_concentration(
            raw, eps
        )
        active_skills, active_activities = self.compute_active_deltas(
            skill_gains, activity_gains, seq_len
        )
        activity_switch = self.compute_activity_switch(activity_gains, seq_len)
        velocity_delta = self.compute_velocity_delta(raw, time_deltas, seq_len)
        activity_streak = self.compute_activity_streak(activity_gains, seq_len)
        gain_cosine_similarity = self.compute_gain_cosine_similarity(
            skill_gains, seq_len
        )

        # Order must match constants.ALL_FEATURES.
        x = np.concatenate(
            [
                np.log1p(raw),
                np.log1p(velocities),
                np.log1p(skill_ratios),
                # Cumulative skill distribution
                combat_gathering_ratio[:, np.newaxis],
                skill_entropy[:, np.newaxis],
                max_skill_ratio[:, np.newaxis],
                n_skills[:, np.newaxis],
                # Cumulative activity distribution
                n_activities[:, np.newaxis],
                activity_entropy[:, np.newaxis],
                max_activity_ratio[:, np.newaxis],
                # Per-interval skill dynamics
                gain_entropy[:, np.newaxis],
                gain_cosine_similarity[:, np.newaxis],
                velocity_delta[:, np.newaxis],
                active_skills[:, np.newaxis],
                # Per-interval activity dynamics
                active_activities[:, np.newaxis],
                activity_switch[:, np.newaxis],
                activity_streak[:, np.newaxis],
            ],
            axis=1,
            dtype=np.float32,
        )

        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Cap the output here rather than truncating indices upfront: the
        # prefix window must be feature-computed in full so elapsed_days
        # stays anchored at the window start (horizons.py reads it as days
        # since discovery when selecting per-horizon positions).
        return Computed(
            features=x[-MAX_SEQUENCE_LENGTH:],
            label=np.float32(self.label),
            name=self.name.astype(np.int64),
            elapsed_days=elapsed_days[-MAX_SEQUENCE_LENGTH:].astype(np.float32),
        )
