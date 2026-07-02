from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

try:
    from constants import (
        ALL_CHARACTERS,
        ALL_FEATURES,
        FEATURE_PAD_VALUE,
        NAME_PAD_VALUE,
        OVERALL_INDEX,
    )
except ImportError:
    from panoptos.training.constants import (
        ALL_CHARACTERS,
        ALL_FEATURES,
        FEATURE_PAD_VALUE,
        NAME_PAD_VALUE,
        OVERALL_INDEX,
    )


@dataclass
class TransformerConfig:
    # Feature branch
    feature_dim: int = 96
    n_transformer_layers: int = 3
    feature_n_heads: int = 4
    feature_ff_mult: int = 4
    n_frequencies: int = 16

    # Name branch
    name_embed_dim: int = 16
    name_channels: int = 16
    name_n_heads: int = 2
    max_name_len: int = 12

    # Shared
    dropout_rate: float = 0.2


class ModelOutput(NamedTuple):
    """Per-position logits plus the padding mask that produced them.

    A NamedTuple (not a dataclass) so torch.export's pytree flattening
    handles it without registration.
    """

    logits: torch.Tensor  # (B, T); position t is the prediction given snapshots 0..t
    padding_mask: torch.Tensor  # (B, T); True at padded positions

    @property
    def lengths(self) -> torch.Tensor:
        """Unpadded length of each sequence, shape (B,)."""
        return (~self.padding_mask).sum(dim=1)

    @property
    def final_logits(self) -> torch.Tensor:
        """Each sequence's last unpadded logit, shape (B,)."""
        last = self.lengths.clamp(min=1) - 1
        return self.logits.gather(1, last.unsqueeze(1)).squeeze(1)


class TimeFourierEncoding(nn.Module):
    def __init__(
        self,
        n_frequencies: int,
        min_period_days: float = 0.4,  # 0.5 day min frequency +/- 20% jitter
        max_period_days: float = 365.0,
    ):
        super().__init__()

        periods = torch.logspace(
            start=torch.log10(torch.tensor(min_period_days)).item(),
            end=torch.log10(torch.tensor(max_period_days)).item(),
            steps=n_frequencies,
        )
        freqs = 2.0 * torch.pi / periods

        self.register_buffer("freqs", freqs)

    def forward(self, elapsed_days: torch.Tensor) -> torch.Tensor:
        angles = elapsed_days.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        assert hidden_size // 4 > 1
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.attention(sequence)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))

        weights = scores.softmax(dim=1)
        return (sequence * weights).sum(dim=1)


class FeatureTransformerEncoder(nn.Module):
    """Causal transformer encoder for temporal feature sequences."""

    def __init__(
        self,
        input_size: int,
        config: TransformerConfig,
    ):
        super().__init__()
        hidden_size = config.feature_dim

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.time_encoding = TimeFourierEncoding(
            n_frequencies=config.n_frequencies,
        )

        self.time_proj = nn.Linear(2 * config.n_frequencies, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.feature_n_heads,
            dim_feedforward=hidden_size * config.feature_ff_mult,
            dropout=config.dropout_rate,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_transformer_layers,
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        features: torch.Tensor,
        elapsed_days: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position hidden states, shape (batch, seq_len, hidden)."""
        x = self.input_proj(features) + self.time_proj(self.time_encoding(elapsed_days))

        # (T, T) causal mask, True where position i may not see j > i. Built
        # from tensor ops on the input (rather than Python ints) so ONNX
        # tracing keeps the sequence dimension dynamic.
        positions = torch.ones_like(elapsed_days[0]).cumsum(0)
        causal_mask = positions.unsqueeze(0) > positions.unsqueeze(1)

        # is_causal=True is a hint that skips torch's mask auto-detection,
        # which is data-dependent and breaks dynamo ONNX export.
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=True,
        )

        return self.norm(x)


class NameTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        config: TransformerConfig,
    ):
        super().__init__()

        embed_dim = config.name_embed_dim
        output_dim = config.name_channels

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=NAME_PAD_VALUE)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_name_len, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.name_n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=config.dropout_rate,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.pool = AttentionPooling(output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        names: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = names.size(1)
        x = self.embedding(names) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.proj(x)
        x = self.pool(x, key_padding_mask)

        return self.norm(x)


class PanoptOSTransformer(nn.Module):
    """Bot detection model with transformer feature encoder and username branch."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ):
        super().__init__()
        config = config or TransformerConfig()
        self.config = config

        self.feature_encoder = FeatureTransformerEncoder(
            input_size=len(ALL_FEATURES),
            config=config,
        )

        if feature_mean is not None and feature_std is not None:
            with torch.no_grad():
                std = feature_std.float().clamp(min=1e-6)
                mean = feature_mean.float()
                proj = self.feature_encoder.input_proj
                proj.bias.sub_(proj.weight @ (mean / std))
                proj.weight.div_(std)

        self.name_encoder = NameTransformer(
            vocab_size=len(ALL_CHARACTERS) + 1,
            config=config,
        )

        self.name_gate = nn.Sequential(
            nn.Linear(config.feature_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        combined_size = config.feature_dim + config.name_channels

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        """Per-position logits (batch, seq_len) bundled with the padding mask.

        Position t is the prediction given snapshots 0..t only (causal mask).
        Padded positions produce garbage; mask them with .padding_mask or read
        .final_logits for the last valid position.
        """
        features_padding_mask = (
            batch["features"][:, :, OVERALL_INDEX] == FEATURE_PAD_VALUE
        )
        names_padding_mask = batch["name"] == NAME_PAD_VALUE

        feature_seq = self.feature_encoder(
            batch["features"],
            batch["elapsed_days"],
            features_padding_mask,
        )

        name_out = self.name_encoder(
            batch["name"],
            names_padding_mask,
        )

        gate = self.name_gate(feature_seq)
        name_seq = name_out.unsqueeze(1) * gate

        combined = torch.cat([feature_seq, name_seq], dim=-1)

        return ModelOutput(self.classifier(combined).squeeze(-1), features_padding_mask)
