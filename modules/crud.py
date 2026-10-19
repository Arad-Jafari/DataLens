from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _is_missing_scalar(value: Any) -> bool:
    return bool(pd.api.types.is_scalar(value) and pd.isna(value))


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique before CRUD operations.")
    if len(df.columns) == 0:
        raise ValueError("Cannot manage records in a dataset with no columns.")


def _validate_record_values(
    df: pd.DataFrame,
    values: Mapping[Any, Any],
    *,
    require_all_columns: bool,
) -> dict[Any, Any]:
    if not isinstance(values, Mapping):
        raise ValueError("Record values must be provided as a mapping.")

    unknown_columns = [column for column in values if column not in df.columns]
    if unknown_columns:
        raise ValueError(f"Unknown column(s): {unknown_columns!r}")

    if require_all_columns:
        missing_columns = [column for column in df.columns if column not in values]
        if missing_columns:
            raise ValueError(f"Missing value(s) for column(s): {missing_columns!r}")
        if len(values) != len(df.columns):
            raise ValueError("A new record must contain exactly one value per column.")
    elif not values:
        raise ValueError("Provide at least one column value to update.")

    return dict(values)


def _validate_position(df: pd.DataFrame, row_position: int) -> int:
    if isinstance(row_position, bool) or not isinstance(row_position, int):
        raise ValueError("Row position must be an integer.")
    if not 0 <= row_position < len(df):
        raise ValueError(
            f"Row position must be between 0 and {len(df) - 1}."
        )
    return row_position


def create_record(
    df: pd.DataFrame,
    values: Mapping[Any, Any],
) -> pd.DataFrame:
    """Return a copy with one record appended in the DataFrame's column order."""
    _validate_dataframe(df)
    record = _validate_record_values(df, values, require_all_columns=True)
    new_row = pd.DataFrame([record], columns=df.columns)
    result = pd.concat([df, new_row], ignore_index=True)
    result.index.name = df.index.name
    return result


def read_record(df: pd.DataFrame, row_position: int) -> pd.Series:
    """Return a copy of one row selected by zero-based position, not index label."""
    _validate_dataframe(df)
    position = _validate_position(df, row_position)
    return df.iloc[position].copy()


def update_record(
    df: pd.DataFrame,
    row_position: int,
    updates: Mapping[Any, Any],
) -> pd.DataFrame:
    """Return a copy with specified fields updated at a zero-based row position."""
    _validate_dataframe(df)
    position = _validate_position(df, row_position)
    values = _validate_record_values(df, updates, require_all_columns=False)

    result = df.copy(deep=True)
    for column, value in values.items():
        series = result[column].copy()
        if _is_missing_scalar(value):
            if pd.api.types.is_integer_dtype(series.dtype):
                series = series.astype("Int64")
            elif pd.api.types.is_bool_dtype(series.dtype):
                series = series.astype("boolean")
        elif isinstance(series.dtype, pd.CategoricalDtype):
            if value not in series.cat.categories:
                series = series.cat.add_categories([value])
        else:
            try:
                value = pd.Series([value]).astype(series.dtype).iloc[0]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Value {value!r} is incompatible with column "
                    f"{column!r} ({series.dtype})."
                ) from exc
        series.iloc[position] = value
        result[column] = series
    return result


def delete_record(df: pd.DataFrame, row_position: int) -> pd.DataFrame:
    """Return a copy with one row deleted by zero-based position."""
    _validate_dataframe(df)
    position = _validate_position(df, row_position)
    keep_positions = [index for index in range(len(df)) if index != position]
    result = df.iloc[keep_positions].copy()
    result.index.name = df.index.name
    return result
