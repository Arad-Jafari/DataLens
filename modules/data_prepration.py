"""Backward-compatible imports for the former misspelled module name."""

from .data_preparation import (
    prepare_features_target,
    split_dataset,
    validate_dataset,
)

__all__ = [
    "prepare_features_target",
    "split_dataset",
    "validate_dataset",
]
