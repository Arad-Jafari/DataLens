from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def sha256_digest(data: bytes) -> str:
    """Return a stable SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def make_upload_id(file_name: str, file_bytes: bytes) -> tuple[str, int, str]:
    """Build a content-based identifier for an uploaded file."""
    return (file_name, len(file_bytes), sha256_digest(file_bytes))


def get_column_series(df: pd.DataFrame, column: Any) -> pd.Series:
    """Return a single Series for a column name, even if the name is duplicated."""
    data = df.loc[:, column]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def get_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate reusable dataset metrics without mutating *df*."""
    row_count, column_count = df.shape
    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    total_cells = row_count * column_count
    completeness = (
        100.0 * (1.0 - missing_count / total_cells)
        if total_cells > 0
        else 100.0
    )
    memory_bytes = int(df.memory_usage(deep=True).sum())
    duplicate_percent = (
        duplicate_count / row_count * 100.0 if row_count > 0 else 0.0
    )
    return {
        "row_count": int(row_count),
        "column_count": int(column_count),
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "total_cells": int(total_cells),
        "completeness": float(completeness),
        "memory_bytes": memory_bytes,
        "memory_kb": memory_bytes / 1024.0,
        "duplicate_percent": float(duplicate_percent),
    }


def numeric_columns(df: pd.DataFrame, *, include_bool: bool = True) -> list[Any]:
    """Column labels that pandas treats as numeric (bool included by default)."""
    columns: list[Any] = []
    for column in df.columns:
        series = get_column_series(df, column)
        if pd.api.types.is_bool_dtype(series):
            if include_bool:
                columns.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            columns.append(column)
    return columns


def categorical_columns(df: pd.DataFrame) -> list[Any]:
    """Non-numeric, non-datetime columns. Booleans follow pandas numeric typing."""
    columns: list[Any] = []
    for column in df.columns:
        series = get_column_series(df, column)
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        columns.append(column)
    return columns


def make_frequency_table(
    series: pd.Series,
    grouped: bool = False,
    bins: int = 10,
    limit: int = 100,
) -> pd.DataFrame:
    """Build a frequency table, including grouped numeric intervals when requested."""
    columns = [
        "Value / Interval",
        "Frequency",
        "Relative Frequency (%)",
        "Cumulative Frequency",
        "Cumulative Percentage (%)",
    ]
    values = series.dropna()
    if values.empty:
        return pd.DataFrame(columns=columns)

    if grouped and pd.api.types.is_numeric_dtype(values) and values.nunique() > 1:
        try:
            counts = pd.cut(
                values,
                bins=bins,
                include_lowest=True,
            ).value_counts(sort=False)
        except ValueError:
            counts = values.value_counts(dropna=False)
            if len(counts) > limit:
                counts = counts.head(limit)
    else:
        counts = values.value_counts(dropna=False)
        if len(counts) > limit:
            counts = counts.head(limit)

    table = counts.rename_axis("Value / Interval").reset_index(name="Frequency")
    table["Value / Interval"] = table["Value / Interval"].astype(str)
    total = int(table["Frequency"].sum())
    if total <= 0:
        return pd.DataFrame(columns=columns)

    table["Relative Frequency (%)"] = (table["Frequency"] / total * 100).round(2)
    table["Cumulative Frequency"] = table["Frequency"].cumsum()
    table["Cumulative Percentage (%)"] = (
        table["Cumulative Frequency"] / total * 100
    ).round(2)
    return table
