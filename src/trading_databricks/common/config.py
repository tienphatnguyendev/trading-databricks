"""Typed configuration dataclasses for trading_databricks pipelines.

All fields use keyword-only construction.  Pass instances through pure
transform functions rather than threading raw parameters.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DollarBarConfig:
    """Configuration for dollar bar construction."""

    threshold: float  # dollars of dollar_value per bar


@dataclass(frozen=True)
class VolumeBarConfig:
    """Configuration for volume bar construction."""

    threshold: float  # units of quantity per bar


@dataclass(frozen=True)
class TickImbalanceBarConfig:
    """Configuration for tick imbalance bar construction.

    The initial expected bar size (E0_T) seeds the exponential weighted
    moving average of bar lengths used to adapt the imbalance threshold.
    """

    initial_expected_bar_size: int = 100
    ewma_span: int = 10


@dataclass(frozen=True)
class DollarImbalanceBarConfig:
    """Configuration for dollar imbalance bar construction."""

    initial_expected_bar_size: int = 100
    ewma_span: int = 10
