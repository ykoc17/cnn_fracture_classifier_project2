"""Compact baseline model and architecture factory."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class SimpleCNNConfig:
    in_channels: int = 1
    channels: tuple[int, ...] = (32, 64, 128)
    num_classes: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if len(self.channels) < 2 or any(channel < 1 for channel in self.channels):
            raise ValueError("channels must contain at least two positive widths")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class SimpleCNN(nn.Module):
    """Three-layer CNN matching the project brief's suggested baseline."""

    def __init__(self, config: SimpleCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or SimpleCNNConfig()
        layers: list[nn.Module] = []
        input_channels = self.config.in_channels
        for index, output_channels in enumerate(self.config.channels):
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < len(self.config.channels) - 1:
                layers.append(nn.MaxPool2d(2))
            input_channels = output_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.channels[-1], self.config.num_classes),
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
