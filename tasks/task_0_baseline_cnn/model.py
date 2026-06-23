"""Simple baseline CNN for Task 0."""

from src.models import SimpleCNN, SimpleCNNConfig

BASELINE_CONFIG = SimpleCNNConfig(
    in_channels=1,
    channels=(16, 32, 64),
    num_classes=2,
    dropout=0.0,
)


class BaselineCNN(SimpleCNN):
    """Three convolution/ReLU stages, two max pools, and a two-class head."""

    def __init__(self) -> None:
        super().__init__(BASELINE_CONFIG)


def build_baseline_model() -> BaselineCNN:
    return BaselineCNN()
