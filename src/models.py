"""Compact baseline model and architecture factory."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class SimpleCNNConfig:
    in_channels: int = 1
    channels: tuple[int, int, int] = (32, 64, 128)
    num_classes: int = 2
    dropout: float = 0.0


class SimpleCNN(nn.Module):
    """Three-layer CNN matching the project brief's suggested baseline."""

    def __init__(self, config: SimpleCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or SimpleCNNConfig()
        c1, c2, c3 = self.config.channels
        self.features = nn.Sequential(
            nn.Conv2d(self.config.in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout),
            nn.Linear(c3, self.config.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))

    def architecture_config(self) -> dict[str, Any]:
        kwargs = asdict(self.config)
        kwargs["channels"] = list(self.config.channels)
        return {"name": "simple_cnn", "kwargs": kwargs}


def build_model(architecture: Mapping[str, Any]) -> nn.Module:
    """Build a registered model from checkpoint-safe architecture metadata."""
    if architecture.get("name") != "simple_cnn":
        raise ValueError(f"unsupported architecture: {architecture.get('name')!r}")
    kwargs = dict(architecture.get("kwargs", {}))
    if "channels" in kwargs:
        kwargs["channels"] = tuple(kwargs["channels"])
    return SimpleCNN(SimpleCNNConfig(**kwargs))
