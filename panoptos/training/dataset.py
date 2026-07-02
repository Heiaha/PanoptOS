"""PyTorch IterableDataset for streaming from flat Parquet shards."""

import random
from pathlib import Path
from typing import assert_never

import numpy as np
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

try:
    from constants import (
        FEATURE_PAD_VALUE,
        NAME_PAD_VALUE,
        RAW_FEATURES,
        encode,
    )
    from features import Computed, PlayerSequence, Skipped
except ImportError:
    from panoptos.training.constants import (
        FEATURE_PAD_VALUE,
        NAME_PAD_VALUE,
        RAW_FEATURES,
        encode,
    )
    from panoptos.training.features import Computed, PlayerSequence, Skipped


class PanoptOSDataset(IterableDataset):
    """Streaming dataset that reads flat Parquet and groups by player.

    `training`, `prefix_days`, and `snapshot_dropout` are fixed at
    construction. Build a separate instance per configuration (e.g. a
    prefix-windowed loader for horizon evaluation) rather than mutating
    these between passes. `snapshot_dropout` is the maximum per-sequence
    rate of randomly thinned snapshots and applies to training windows only.
    """

    def __init__(
        self,
        cache_dir: Path,
        training: bool = False,
        prefix_days: float | None = None,
        snapshot_dropout: float = 0.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.training = training
        self.prefix_days = prefix_days
        self.snapshot_dropout = snapshot_dropout
        self.paths = list(self.cache_dir.rglob("*.parquet"))

    def __iter__(self):
        paths = self.paths.copy()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]

        if self.training:
            random.shuffle(paths)

        for path in paths:
            yield from self._stream_file(path)

    def _stream_file(self, path: Path):
        df = pl.read_parquet(path).sort("name", "captured_at")

        names = df["name"].to_numpy()
        states_all = df.select(RAW_FEATURES).to_numpy().astype(np.float64, copy=False)
        timestamps_all = df["captured_at"].dt.epoch("s").to_numpy()
        discovered_all = df["discovered_at"].dt.epoch("s").to_numpy()
        labels_all = df["label"].to_numpy()

        change_points = np.flatnonzero(names[1:] != names[:-1]) + 1
        boundaries = np.concatenate([[0], change_points, [df.height]])
        groups = list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))

        if self.training:
            random.shuffle(groups)

        for start, end in groups:
            name = names[start]
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            seq = PlayerSequence(
                name=np.array(encode(name), dtype=np.uint8),
                states=states_all[start:end],
                timestamps=timestamps_all[start:end],
                discovered_at=float(discovered_all[start]),
                label=int(labels_all[start]),
            )
            features = seq.calculate_features(
                prefix_days=self.prefix_days,
                training=self.training,
                snapshot_dropout=self.snapshot_dropout,
            )
            # Full-history observation span, before any windowing. For banned
            # accounts this proxies time-to-ban (the stream ends at the ban),
            # which the evaluation uses to stratify recall by ban latency.
            span_days = (timestamps_all[end - 1] - seq.discovered_at) / 86400.0
            match features:
                case Computed() as result:
                    yield {
                        "features": torch.from_numpy(result.features),
                        "elapsed_days": torch.from_numpy(result.elapsed_days),
                        "label": torch.tensor(result.label),
                        "name": torch.from_numpy(result.name),
                        "span_days": torch.tensor(span_days, dtype=torch.float32),
                    }
                case Skipped():
                    continue
                case _:
                    assert_never(features)

    @staticmethod
    def collate_fn(batch):
        features = [b["features"] for b in batch]
        names = [b["name"] for b in batch]
        elapsed_days = [b["elapsed_days"] for b in batch]
        labels = torch.stack([b["label"] for b in batch])

        return {
            "features": pad_sequence(
                features, batch_first=True, padding_value=FEATURE_PAD_VALUE
            ),
            "name": pad_sequence(names, batch_first=True, padding_value=NAME_PAD_VALUE),
            "elapsed_days": pad_sequence(
                elapsed_days, batch_first=True, padding_value=0.0
            ),
            "label": labels,
            "span_days": torch.stack([b["span_days"] for b in batch]),
        }
