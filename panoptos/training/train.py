"""Step 2: Train the PanoptOS Transformer model.

Loads the Parquet dataset built by 01_build_dataset.py, fits a StandardScaler,
and trains the model with checkpointing on best validation AUPRC.

Usage:
    python 02_train.py --cache-path ./cache --epochs 20 --batch-size 256 --lr 1e-3
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from tqdm import tqdm

try:
    from constants import MIN_SEQUENCE_LENGTH
    from dataset import PanoptOSDataset
    from horizons import (
        DEFAULT_HORIZONS,
        collect_position_predictions,
        horizon_selections,
        integrate_horizon,
        platt_scale,
    )
    from model import ModelOutput, PanoptOSTransformer
except ImportError:
    from panoptos.training.constants import MIN_SEQUENCE_LENGTH
    from panoptos.training.dataset import PanoptOSDataset
    from panoptos.training.horizons import (
        DEFAULT_HORIZONS,
        collect_position_predictions,
        horizon_selections,
        integrate_horizon,
        platt_scale,
    )
    from panoptos.training.model import ModelOutput, PanoptOSTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def cap_positives(
    labels: np.ndarray,
    max_pos_frac: float = 0.20,
    seed: int = 42,
) -> np.ndarray:
    """Indices downsampling positives to at most `max_pos_frac` of the set."""
    pos_mask = labels == 1
    n_pos, n_neg = pos_mask.sum(), (~pos_mask).sum()
    max_pos = int(n_neg * max_pos_frac / (1 - max_pos_frac))

    if n_pos <= max_pos:
        return np.arange(len(labels))

    rng = np.random.default_rng(seed=seed)
    pos_idx = rng.choice(np.where(pos_mask)[0], size=max_pos, replace=False)
    cal_idx = np.concatenate([np.where(~pos_mask)[0], pos_idx])
    rng.shuffle(cal_idx)
    return cal_idx


def fit_platt(
    logits: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.1,
    max_iter: int = 2000,
    tol: float = 1e-7,
) -> tuple[float, float, float]:
    """Fit length-conditioned Platt scaling via gradient descent.

    Returns (a, c, b) for P(y=1|s, n) = sigmoid(a*s + c*log(n) + b), where n
    is the scored snapshot count. The length covariate absorbs the score
    drift across snapshot counts so one tipoff threshold carries the same
    expected precision at every account maturity. The covariate is centered
    during the fit for conditioning; the returned b folds the offset back in.
    """
    log_lengths = np.log(lengths)
    mean_ll = log_lengths.mean()
    centered = log_lengths - mean_ll

    a, c, b = 1.0, 0.0, 0.0
    for _ in range(max_iter):
        z = np.clip(a * logits + c * centered + b, -500, 500)
        probs = 1 / (1 + np.exp(-z))
        grad_a = np.mean((probs - labels) * logits)
        grad_c = np.mean((probs - labels) * centered)
        grad_b = np.mean(probs - labels)
        a -= lr * grad_a
        c -= lr * grad_c
        b -= lr * grad_b
        if abs(grad_a) < tol and abs(grad_c) < tol and abs(grad_b) < tol:
            break
    return float(a), float(c), float(b - c * mean_ll)


def compute_feature_stats(
    cache_path: Path, num_workers: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dataset-wide mean/std using Welford's online algorithm.

    Uses its own non-training dataset instance so windows are deterministic,
    independent of the augmenting dataset the training loop consumes.
    """
    loader = DataLoader(
        PanoptOSDataset(cache_path / "split=train", training=False),
        batch_size=None,
        num_workers=num_workers,
    )

    n_samples = 0
    mean = None
    m2 = None

    for item in tqdm(loader, desc="Computing dataset statistics"):
        data = item["features"].to(torch.float64)
        T = data.size(0)

        if mean is None:
            C = data.size(1)
            mean = torch.zeros(C, dtype=torch.float64)
            m2 = torch.zeros(C, dtype=torch.float64)

        sample_mean = data.mean(dim=0)
        sample_var = data.var(dim=0, unbiased=False)

        delta = sample_mean - mean
        total_samples = n_samples + T
        mean = mean + delta * T / total_samples
        m2 = m2 + sample_var * T + delta**2 * n_samples * T / total_samples
        n_samples = total_samples

    std = torch.sqrt(m2 / n_samples)
    return mean, std


def sequence_bce_loss(out: ModelOutput, labels: torch.Tensor) -> torch.Tensor:
    """Per-position BCE against the sequence label.

    Positions that have seen fewer than MIN_SEQUENCE_LENGTH snapshots are
    not supervised (inference never scores them). Averaged within each
    sequence before averaging over the batch so long sequences don't
    dominate the gradient.
    """
    logits = out.logits
    positions = torch.arange(logits.size(1), device=logits.device)
    supervised = ~out.padding_mask & (positions >= MIN_SEQUENCE_LENGTH - 1)

    targets = labels.unsqueeze(1).expand_as(logits)
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    losses = losses * supervised

    per_sequence = losses.sum(dim=1) / supervised.sum(dim=1).clamp(min=1)
    return per_sequence.mean()


def train_epoch(
    model: PanoptOSTransformer,
    loader: DataLoader,
    optimizer: AdamW,
    device: str,
) -> tuple[float, float]:
    """Train for one epoch. Returns (auroc, auprc)."""
    model.train()

    auroc, auprc = BinaryAUROC(), BinaryAUPRC()

    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        out = model(batch)
        sequence_bce_loss(out, batch["label"]).backward()
        optimizer.step()

        probs = out.final_logits.detach().sigmoid()
        auroc.update(probs, batch["label"])
        auprc.update(probs, batch["label"])

    return auroc.compute().item(), auprc.compute().item()


@torch.no_grad()
def validate(
    model: PanoptOSTransformer,
    loader: DataLoader,
    device: str,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate on validation set. Returns (auroc, auprc, logits, labels, lengths)."""
    model.eval()

    auroc, auprc = BinaryAUROC(), BinaryAUPRC()
    all_logits, all_labels, all_lengths = [], [], []

    for batch in tqdm(loader, desc="Validating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(batch)
        logits = out.final_logits
        auroc.update(logits.sigmoid(), batch["label"])
        auprc.update(logits.sigmoid(), batch["label"])

        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].cpu())
        all_lengths.append(out.lengths.cpu())

    return (
        auroc.compute().item(),
        auprc.compute().item(),
        torch.cat(all_logits).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_lengths).numpy(),
    )


@torch.no_grad()
def compute_itauc_pr(
    model: PanoptOSTransformer,
    loader: DataLoader,
    device: str,
    calibration: dict,
) -> float:
    """ITAUC-PR over DEFAULT_HORIZONS via the shared one-pass horizon eval.

    Point estimates only (no bootstrap); scores are the epoch's calibrated
    probabilities, matching what 03_evaluate.py reports. `loader` must be
    prefix-windowed with prefix_days = max(DEFAULT_HORIZONS).
    """
    model.eval()
    labels, account_idx, positions, elapsed, logits, _ = collect_position_predictions(
        model, loader, device, desc="Horizon validation"
    )

    metrics: dict[int, dict[str, float | None]] = {}
    for days, present, sel in horizon_selections(
        len(labels), account_idx, positions, elapsed, DEFAULT_HORIZONS
    ):
        if present.any():
            scores = platt_scale(logits[sel], positions[sel] + 1, calibration)
            auprc = float(average_precision_score(labels[present], scores))
        else:
            auprc = None
        metrics[days] = {"auprc": auprc}

    return integrate_horizon(metrics, "auprc")


def restore_checkpoint(
    checkpoint_path: Path,
    model: PanoptOSTransformer,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: str,
) -> tuple[int, float, int]:
    """Resume training state in place. Returns (start_epoch, best_auprc, best_epoch)."""
    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    best_epoch, best_auprc = checkpoint["epoch"], checkpoint["val_auprc"]
    logger.info("Resuming from epoch %d (best AUPRC=%.4f)", best_epoch + 1, best_auprc)
    return best_epoch + 1, best_auprc, best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PanoptOS Transformer model")
    parser.add_argument("--cache-path", type=Path, default=Path("./cache"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cal-pos-frac",
        type=float,
        default=0.1,
        help="Max positive fraction for Platt calibration (approximate OSRS bot rate)",
    )
    parser.add_argument(
        "--snapshot-dropout",
        type=float,
        default=0.25,
        help=(
            "Max per-sequence rate of randomly thinned snapshots in training "
            "windows, so the model can't read the polling cadence as a feature"
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=("pr", "itauc"),
        default="itauc",
        help=(
            "Checkpoint selection: 'pr' uses full-window validation PR-AUC; "
            "'itauc' uses per-epoch ITAUC-PR so checkpoints are chosen on "
            "horizon performance, the deployment objective"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", args.device)

    # Data setup
    train_dataset = PanoptOSDataset(
        args.cache_path / "split=train",
        training=True,
        snapshot_dropout=args.snapshot_dropout,
    )
    val_dataset = PanoptOSDataset(args.cache_path / "split=test", training=False)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=PanoptOSDataset.collate_fn,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_dataset, pin_memory=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    horizon_loader = None
    if args.selection_metric == "itauc":
        horizon_dataset = PanoptOSDataset(
            args.cache_path / "split=test",
            training=False,
            prefix_days=float(max(DEFAULT_HORIZONS)),
        )
        horizon_loader = DataLoader(horizon_dataset, **loader_kwargs)

    start_epoch = 0
    best_score, best_epoch = -float("inf"), -1
    checkpoint_path = args.checkpoint_dir / "best.pt"
    resuming = checkpoint_path.exists()

    # Fit feature stats fresh runs only; for resumed runs the stats are already
    # fused into the checkpoint's input_proj weights.
    if resuming:
        feature_mean, feature_std = None, None
    else:
        feature_mean, feature_std = compute_feature_stats(
            args.cache_path, num_workers=args.num_workers
        )

    model = PanoptOSTransformer(
        feature_mean=feature_mean,
        feature_std=feature_std,
    ).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if resuming:
        # The checkpoint stores the full-window AUPRC, which stands in as the
        # incumbent selection score on resume. With ITAUC selection this is
        # conservative (ITAUC-PR runs below the full-window PR), so the first
        # improving epoch after a resume may take a little longer to appear.
        start_epoch, best_score, best_epoch = restore_checkpoint(
            checkpoint_path, model, optimizer, scheduler, args.device
        )

    logger.info("Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_roc, train_pr = train_epoch(model, train_loader, optimizer, args.device)
        val_roc, val_pr, val_logits, val_labels, val_lengths = validate(
            model, val_loader, args.device
        )

        cal_idx = cap_positives(val_labels, max_pos_frac=args.cal_pos_frac)
        a, c, b = fit_platt(
            val_logits[cal_idx], val_lengths[cal_idx], val_labels[cal_idx]
        )
        calibration = {"method": "platt_length", "a": a, "b": b, "c": c}

        if horizon_loader is not None:
            itauc_pr = compute_itauc_pr(model, horizon_loader, args.device, calibration)
            selection = itauc_pr
        else:
            itauc_pr = float("nan")
            selection = val_pr

        logger.info(
            "Epoch %d - Train ROC: %.4f, PR: %.4f | "
            "Val ROC: %.4f, PR: %.4f, ITAUC-PR: %.4f | "
            "Platt A=%.4f, C=%.4f, B=%.4f",
            epoch,
            train_roc,
            train_pr,
            val_roc,
            val_pr,
            itauc_pr,
            a,
            c,
            b,
        )
        scheduler.step()

        if selection > best_score:
            best_score, best_epoch = selection, epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_auroc": val_roc,
                    "val_auprc": val_pr,
                    "calibration": calibration,
                },
                checkpoint_path,
            )
            logger.info(
                "Saved best model @ epoch %d (selection=%.4f, Val AUPRC=%.4f)",
                epoch,
                selection,
                val_pr,
            )

    logger.info(
        "Training complete. Best epoch: %d (selection=%.4f)", best_epoch, best_score
    )


if __name__ == "__main__":
    main()
