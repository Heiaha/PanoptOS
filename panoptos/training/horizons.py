"""Shared one-pass horizon evaluation for the training pipeline.

The causal model emits a logit at every position, so a single forward pass
over prefix-windowed sequences (prefix_days = max(horizons)) yields every
horizon's score by reading off each account's last scoreable position within
the horizon. 02_train.py uses this for per-epoch ITAUC model selection;
03_evaluate.py builds its bootstrap-CI horizon table on top of it.
"""

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from constants import MIN_SEQUENCE_LENGTH
    from model import PanoptOSTransformer  # type: ignore[import-not-found]
except ImportError:
    from panoptos.training.constants import MIN_SEQUENCE_LENGTH
    from panoptos.training.model import PanoptOSTransformer

DEFAULT_HORIZONS = list(range(4, 31))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities."""
    probs: np.ndarray = 1 / (1 + np.exp(-logits))
    return probs


def platt_scale(
    logits: np.ndarray,
    lengths: np.ndarray,
    calibration: Mapping[str, float],
) -> np.ndarray:
    """Apply length-conditioned Platt: P(y=1|s, n) = sigmoid(a*s + c*log(n) + b)."""
    a, b, c = calibration["a"], calibration["b"], calibration["c"]
    z = a * logits + c * np.log(lengths) + b
    return sigmoid(np.clip(z, -500, 500))


@torch.no_grad()
def collect_position_predictions(
    model: PanoptOSTransformer,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: str,
    desc: str = "Collecting per-position predictions",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One causal pass keeping every position.

    Expects a loader whose dataset was constructed with
    `prefix_days = max(horizons)` so windows are anchored at discovery.
    Returns (labels, account_idx, positions, elapsed, logits, spans) where
    labels and spans (full-history observation span in days) are per-account
    and the remaining arrays are flattened over each account's unpadded
    positions, ordered by (account, position).
    """
    all_labels: list[np.ndarray] = []
    all_accounts: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    all_elapsed: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_spans: list[np.ndarray] = []
    n_accounts = 0

    for batch in tqdm(loader, desc=desc, disable=not desc):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        idx_b, idx_t = (~out.padding_mask).nonzero(as_tuple=True)

        all_labels.append(batch["label"].cpu().numpy())
        all_accounts.append((idx_b + n_accounts).cpu().numpy())
        all_positions.append(idx_t.cpu().numpy())
        all_elapsed.append(batch["elapsed_days"][idx_b, idx_t].cpu().numpy())
        all_logits.append(out.logits[idx_b, idx_t].cpu().numpy())
        all_spans.append(batch["span_days"].cpu().numpy())
        n_accounts += out.logits.size(0)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_accounts),
        np.concatenate(all_positions),
        np.concatenate(all_elapsed),
        np.concatenate(all_logits),
        np.concatenate(all_spans),
    )


def horizon_selections(
    n_accounts: int,
    account_idx: np.ndarray,
    positions: np.ndarray,
    elapsed: np.ndarray,
    horizons: Sequence[int],
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (days, present, sel): each account's last scoreable position per horizon.

    `present` is a boolean (n_accounts,) mask of accounts with at least one
    scoreable position within `days` of the window start; `sel` indexes the
    flattened position arrays at those accounts' last in-horizon positions.
    Positions with fewer than MIN_SEQUENCE_LENGTH snapshots are never scored
    (matching inference). Input rows must be ordered by (account, position),
    as collect_position_predictions produces, so the max flat index per
    account is its latest position.
    """
    scoreable = positions >= MIN_SEQUENCE_LENGTH - 1
    for days in horizons:
        in_horizon = np.flatnonzero(scoreable & (elapsed <= days))
        last = np.full(n_accounts, -1, dtype=np.int64)
        np.maximum.at(last, account_idx[in_horizon], in_horizon)
        present = last >= 0
        yield days, present, last[present]


def integrate_horizon(
    metrics: Mapping[int, Mapping[str, float | int | None]], key: str
) -> float:
    """Mean AUC over evaluated time horizons via trapezoidal rule."""
    valid = [(d, m[key]) for d, m in sorted(metrics.items()) if m[key] is not None]
    if len(valid) < 2:
        return float("nan")

    days_tuple, values_tuple = zip(*valid)
    days = np.asarray(days_tuple, dtype=float)
    values = np.asarray(values_tuple, dtype=float)

    return float(np.trapezoid(values, days) / (days[-1] - days[0]))
