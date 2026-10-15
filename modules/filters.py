from __future__ import annotations

from typing import Any

import pandas as pd

from modules.data_utils import get_column_series

TEXT_OPERATORS = [
    "Contains",
    "Equals",
    "Not equal to",
    "Starts with",
    "Ends with",
    "Is missing",
    "Is not missing",
]

NUMERIC_OPERATORS = [
    "Between",
    "Greater than",
    "Greater than or equal to",
    "Less than",
    "Less than or equal to",
    "Equals",
    "Not equal to",
    "Is missing",
    "Is not missing",
]

DATE_OPERATORS = [
    "Between",
    "Before",
    "After",
    "On",
    "Is missing",
    "Is not missing",
]

OPERATORS_BY_KIND = {
    "text": TEXT_OPERATORS,
    "numeric": NUMERIC_OPERATORS,
    "date": DATE_OPERATORS,
}

MISSING_OPERATORS = {"Is missing", "Is not missing"}


def infer_column_type(series: pd.Series) -> str:
    """Infer numeric, date, or text type for a filter control."""
    if pd.api.types.is_bool_dtype(series):
        return "text"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    values = series.dropna()
    if values.empty:
        return "text"

    sample = values.astype(str).head(100)

    numeric = pd.to_numeric(
        sample.str.replace(",", "", regex=False),
        errors="coerce",
    )
    if numeric.notna().mean() >= 0.98:
        return "numeric"

    if sample.str.contains(r"[-/:T]", regex=True).mean() >= 0.8:
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return "date"

    return "text"


def parse_numeric(value: Any) -> float | None:
    if value is None or not str(value).strip():
        return None

    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None

    if pd.isna(result):
        return None
    return result


def validate_filter_drafts(
    drafts: list[dict[str, Any]],
    columns,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate filter drafts without applying them to a DataFrame."""
    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    column_names = list(columns)

    for position, draft in enumerate(drafts, start=1):
        column = draft.get("column")
        kind = draft.get("kind")
        operator = draft.get("operator")
        value = draft.get("value")

        if column not in column_names or not kind or not operator:
            errors.append(f"Condition {position}: select a column and operator.")
            continue

        allowed = OPERATORS_BY_KIND.get(kind)
        if allowed is None or operator not in allowed:
            errors.append(
                f"Condition {position}: unsupported {kind} operator."
            )
            continue

        if operator in MISSING_OPERATORS:
            validated.append({
                "column": column,
                "kind": kind,
                "operator": operator,
                "value": None,
            })
            continue

        if kind == "text":
            if value is None or not str(value).strip():
                errors.append(f"Condition {position}: enter a text value.")
                continue

        elif kind == "numeric":
            if operator == "Between":
                bounds = value if isinstance(value, (list, tuple)) and len(value) == 2 else (None, None)
                lower, upper = bounds
                if lower is None or upper is None:
                    errors.append(
                        f"Condition {position}: enter both numeric bounds."
                    )
                    continue
                if lower > upper:
                    errors.append(
                        f"Condition {position}: minimum exceeds maximum."
                    )
                    continue
            elif value is None:
                errors.append(
                    f"Condition {position}: enter a valid numeric value."
                )
                continue

        elif kind == "date":
            if operator == "Between":
                bounds = value if isinstance(value, (list, tuple)) and len(value) == 2 else (None, None)
                lower, upper = bounds
                if lower is None or upper is None:
                    errors.append(
                        f"Condition {position}: enter both dates."
                    )
                    continue
                if lower > upper:
                    errors.append(
                        f"Condition {position}: start date exceeds end date."
                    )
                    continue
            elif value is None:
                errors.append(f"Condition {position}: select a date.")
                continue

        validated.append({
            "column": column,
            "kind": kind,
            "operator": operator,
            "value": value,
        })

    return validated, errors


def _condition_mask(source: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    """Return a positional boolean mask aligned to *source* rows."""
    column = condition["column"]
    operator = condition["operator"]
    kind = condition["kind"]
    value = condition.get("value")
    series = get_column_series(source, column)
    index = source.index

    if operator == "Is missing":
        return pd.Series(series.isna().to_numpy(), index=index, dtype=bool)

    if operator == "Is not missing":
        return pd.Series(series.notna().to_numpy(), index=index, dtype=bool)

    if kind == "text":
        text = series.astype("string")
        if operator == "Contains":
            result = text.str.contains(
                str(value),
                case=True,
                regex=False,
                na=False,
            )
        elif operator == "Equals":
            result = text.eq(str(value)).fillna(False)
        elif operator == "Not equal to":
            result = text.ne(str(value)).fillna(False)
        elif operator == "Starts with":
            result = text.str.startswith(str(value), na=False)
        elif operator == "Ends with":
            result = text.str.endswith(str(value), na=False)
        else:
            raise ValueError(f"Unsupported text operator: {operator}")
        return pd.Series(result.to_numpy(), index=index, dtype=bool)

    if kind == "numeric":
        numeric = pd.to_numeric(
            series.astype("string").str.replace(",", "", regex=False),
            errors="coerce",
        )
        if operator == "Between":
            result = numeric.between(value[0], value[1], inclusive="both")
        elif operator == "Greater than":
            result = numeric > value
        elif operator == "Greater than or equal to":
            result = numeric >= value
        elif operator == "Less than":
            result = numeric < value
        elif operator == "Less than or equal to":
            result = numeric <= value
        elif operator == "Equals":
            result = numeric == value
        elif operator == "Not equal to":
            result = numeric != value
        else:
            raise ValueError(f"Unsupported numeric operator: {operator}")
        result = result & numeric.notna()
        return pd.Series(result.to_numpy(), index=index, dtype=bool)

    if kind == "date":
        dates = pd.to_datetime(series, errors="coerce")
        if operator == "Between":
            result = dates.between(
                pd.Timestamp(value[0]),
                pd.Timestamp(value[1]),
                inclusive="both",
            )
        elif operator == "Before":
            result = dates < pd.Timestamp(value)
        elif operator == "After":
            result = dates > pd.Timestamp(value)
        elif operator == "On":
            result = dates.dt.date == value
        else:
            raise ValueError(f"Unsupported date operator: {operator}")
        result = result & dates.notna()
        return pd.Series(result.to_numpy(), index=index, dtype=bool)

    raise ValueError(f"Unsupported filter condition: {kind} / {operator}")


def combined_filter_mask(
    source: pd.DataFrame,
    conditions: list[dict[str, Any]],
    logic: str = "AND",
) -> pd.Series:
    """Combine validated conditions into one positional boolean mask."""
    if source.empty:
        return pd.Series(dtype=bool, index=source.index)

    combined = pd.Series(True, index=range(len(source)), dtype=bool)
    positional_index = combined.index

    for condition in conditions:
        mask = _condition_mask(source, condition).fillna(False)
        mask = pd.Series(mask.to_numpy(), index=positional_index, dtype=bool)
        if logic == "OR":
            combined = combined | mask
        else:
            combined = combined & mask

    return pd.Series(combined.to_numpy(), index=source.index, dtype=bool)


def apply_filters(
    source: pd.DataFrame,
    conditions: list[dict[str, Any]],
    logic: str = "AND",
) -> pd.DataFrame:
    """Apply filters with iloc so duplicate index labels cannot duplicate rows."""
    if not conditions:
        return source.copy()

    mask = combined_filter_mask(source, conditions, logic).to_numpy(dtype=bool)
    return source.iloc[mask].copy()


def quick_search_mask(
    source: pd.DataFrame,
    query: str,
    columns: list[Any] | None = None,
) -> pd.Series:
    """Case-sensitive literal search across one or more columns."""
    if source.empty or not query:
        return pd.Series(True, index=source.index, dtype=bool)

    search_columns = list(source.columns) if columns is None else columns
    hits = pd.Series(False, index=range(len(source)), dtype=bool)

    for column in search_columns:
        if column not in source.columns:
            continue
        text = get_column_series(source, column).astype("string")
        column_hits = text.str.contains(
            query,
            case=True,
            regex=False,
            na=False,
        )
        hits = hits | pd.Series(column_hits.to_numpy(), index=hits.index, dtype=bool)

    return pd.Series(hits.to_numpy(), index=source.index, dtype=bool)
