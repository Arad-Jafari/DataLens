from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return a list of validation errors."""
    errors = []

    if not isinstance(df, pd.DataFrame):
        return ["The dataset must be a pandas DataFrame."]

    if df.empty:
        errors.append("The dataset is empty.")

    if len(df.columns) == 0:
        errors.append("The dataset has no columns.")

    if not df.columns.is_unique:
        errors.append("Column names must be unique.")

    if len(df) < 2:
        errors.append("At least two rows are required.")

    return errors


def prepare_features_target(df: pd.DataFrame, target_column):
    """Separate features X and target y."""
    errors = validate_dataset(df)

    if errors:
        raise ValueError(" ".join(errors))

    if target_column not in df.columns:
        raise ValueError("Select a valid target column.")

    if df[target_column].isna().any():
        missing = int(df[target_column].isna().sum())
        raise ValueError(
            f"The target contains {missing} missing values. "
            "Clean these before splitting."
        )

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    if X.shape[1] == 0:
        raise ValueError(
            "At least one feature column is required besides the target."
        )

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain at least two distinct values."
        )

    return X, y


def split_dataset(
    X,
    y,
    test_size=0.2,
    random_state=7,
    task_type="Auto-detect",
    stratify_classification=True,
):
    """Return train/test data and any warnings."""
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("X must be a DataFrame and y must be a Series.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if len(X) < 2:
        raise ValueError("At least two rows are required.")

    if not isinstance(test_size, (int, float)):
        raise ValueError("Test size must be a finite number between 0 and 1.")

    try:
        finite_test_size = math.isfinite(test_size)
    except OverflowError:
        finite_test_size = False
    if not finite_test_size:
        raise ValueError("Test size must be a finite number between 0 and 1.")

    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1.")

    if y.isna().any():
        raise ValueError("The target contains missing values.")

    if task_type not in ("Auto-detect", "Classification", "Regression"):
        raise ValueError(
            "Task type must be Auto-detect, Classification, or Regression."
        )

    warnings = []

    if task_type == "Classification":
        is_classification = True
    elif task_type == "Regression":
        is_classification = False
    else:
        is_classification = (
            not pd.api.types.is_numeric_dtype(y)
            or y.nunique() <= min(20, max(2, int(len(y) * 0.05)))
        )

    stratify = None

    if is_classification and stratify_classification:
        counts = y.value_counts()
        number_of_classes = len(counts)

        test_rows = math.ceil(len(y) * test_size)
        train_rows = len(y) - test_rows

        can_stratify = (
            number_of_classes >= 2
            and counts.min() >= 2
            and test_rows >= number_of_classes
            and train_rows >= number_of_classes
        )

        if can_stratify:
            stratify = y
        else:
            warnings.append(
                "Stratification is not possible with these class counts "
                "and split settings. A regular random split will be used."
            )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to split dataset: {exc}") from exc

    return X_train, X_test, y_train, y_test, warnings


def preprocess_split_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    scaling: str = "None",
    outlier_method: str = "None",
    iqr_multiplier: float = 1.5,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    numeric_imputation: str = "median",
    numeric_fill_value: float = 0.0,
    categorical_imputation: str = "most_frequent",
    categorical_fill_value: str = "(Missing)",
    categorical_encoding: str | None = None,
    categorical_columns: list[Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit feature preprocessing on training data and transform train/test.

    Numeric and categorical imputation, encoding, scaling, and outlier caps
    are fit using training data only. Test rows are never removed.
    """
    _validated_dataframe(X_train)
    _validated_dataframe(X_test)
    if X_train.empty:
        raise ValueError("Training features must contain at least one row.")
    if X_train.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test features must have identical columns.")
    if scaling not in ("None", "Standardization", "Normalization"):
        raise ValueError(
            "Scaling must be None, Standardization, or Normalization."
        )
    if outlier_method not in ("None", "IQR", "Percentile", "Z-score"):
        raise ValueError(
            "Outlier method must be None, IQR, Percentile, or Z-score."
        )
    if numeric_imputation not in ("median", "mean", "constant"):
        raise ValueError(
            "Numeric imputation must be median, mean, or constant."
        )
    if categorical_imputation not in ("most_frequent", "constant"):
        raise ValueError(
            "Categorical imputation must be most_frequent or constant."
        )
    if categorical_columns is None:
        categorical_columns = []
    if not isinstance(categorical_columns, list):
        raise ValueError("Categorical columns must be provided as a list.")
    if len(set(categorical_columns)) != len(categorical_columns):
        raise ValueError("Categorical columns must not contain duplicates.")
    non_categorical_columns = [
        column for column in categorical_columns if column not in X_train.columns
    ]
    if non_categorical_columns:
        raise ValueError(
            f"Unknown categorical column(s): {non_categorical_columns!r}"
        )
    if categorical_columns and categorical_encoding not in ("One-hot", "Ordinal"):
        raise ValueError(
            "Choose One-hot or Ordinal encoding for categorical features."
        )
    if numeric_imputation == "constant":
        try:
            valid_numeric_fill = (
                not isinstance(numeric_fill_value, bool)
                and math.isfinite(float(numeric_fill_value))
            )
        except (TypeError, ValueError, OverflowError):
            valid_numeric_fill = False
        if not valid_numeric_fill:
            raise ValueError("Numeric fill value must be finite.")
    if categorical_imputation == "constant" and not isinstance(
        categorical_fill_value, str
    ):
        raise ValueError("Categorical fill value must be text.")
    if outlier_method == "IQR":
        try:
            valid_iqr_multiplier = (
                not isinstance(iqr_multiplier, bool)
                and math.isfinite(float(iqr_multiplier))
                and float(iqr_multiplier) >= 0
            )
        except (TypeError, ValueError, OverflowError):
            valid_iqr_multiplier = False
        if not valid_iqr_multiplier:
            raise ValueError("IQR multiplier must be a finite non-negative number.")

    if outlier_method == "Percentile":
        try:
            valid_percentiles = (
                not isinstance(lower_percentile, bool)
                and not isinstance(upper_percentile, bool)
                and math.isfinite(float(lower_percentile))
                and math.isfinite(float(upper_percentile))
                and 0 <= float(lower_percentile) < float(upper_percentile) <= 100
            )
        except (TypeError, ValueError, OverflowError):
            valid_percentiles = False
        if not valid_percentiles:
            raise ValueError(
                "Percentiles must satisfy 0 <= lower < upper <= 100."
            )

    numeric_candidates = [
        column
        for column in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[column].dtype)
        and not pd.api.types.is_bool_dtype(X_train[column].dtype)
    ]
    observed_numeric_candidates = [
        column
        for column in numeric_candidates
        if pd.to_numeric(X_train[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .any()
    ]
    numeric_columns = [
        column
        for column in (
            numeric_candidates
            if numeric_imputation == "constant"
            else observed_numeric_candidates
        )
    ]
    dropped_all_missing_numeric_columns = [
        column for column in numeric_candidates if column not in numeric_columns
    ]
    categorical_candidates = [
        column for column in X_train.columns if column not in numeric_candidates
    ]
    non_categorical_selection = [
        column for column in categorical_columns
        if column not in categorical_candidates
    ]
    if non_categorical_selection:
        raise ValueError(
            "Selected categorical column(s) are numeric: "
            f"{non_categorical_selection!r}"
        )
    excluded_categorical_columns = [
        column for column in categorical_candidates
        if column not in categorical_columns
    ]
    if not numeric_columns and not categorical_columns:
        raise ValueError(
            "No features are selected for preprocessing. Select categorical "
            "columns to encode or keep at least one numeric feature."
        )

    train = X_train.copy(deep=True)
    test = X_test.copy(deep=True)
    outlier_bounds: dict[Any, tuple[float, float]] = {}

    for column in numeric_columns:
        train_values = pd.to_numeric(train[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        test_values = pd.to_numeric(test[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        finite_train = train_values.dropna()
        if finite_train.empty:
            if numeric_imputation != "constant":
                raise ValueError(
                    f"Numeric feature {column!r} has no finite training values."
                )
            bounds = (float("-inf"), float("inf"))
        elif outlier_method == "IQR":
            q1, q3 = finite_train.quantile([0.25, 0.75])
            spread = q3 - q1
            multiplier = float(iqr_multiplier)
            bounds = (
                float(q1 - multiplier * spread),
                float(q3 + multiplier * spread),
            )
        elif outlier_method == "Percentile":
            bounds = (
                float(np.percentile(finite_train, float(lower_percentile))),
                float(np.percentile(finite_train, float(upper_percentile))),
            )
        elif outlier_method == "Z-score":
            mean = float(finite_train.mean())
            std = float(finite_train.std(ddof=0))
            bounds = (mean - 3 * std, mean + 3 * std)
        elif not finite_train.empty:
            bounds = (float("-inf"), float("inf"))

        if outlier_method != "None":
            outlier_bounds[column] = bounds
            train[column] = train_values.clip(*bounds)
            test[column] = test_values.clip(*bounds)
        else:
            train[column] = train_values
            test[column] = test_values

    for column in categorical_columns:
        train[column] = train[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        ).astype(object)
        test[column] = test[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        ).astype(object)

    transformers = []
    if numeric_columns:
        if numeric_imputation == "constant":
            numeric_imputer = SimpleImputer(
                strategy="constant",
                fill_value=float(numeric_fill_value),
                keep_empty_features=True,
            )
        else:
            numeric_imputer = SimpleImputer(strategy=numeric_imputation)
        numeric_steps: list[tuple[str, Any]] = [("imputer", numeric_imputer)]
        if scaling == "Standardization":
            numeric_steps.append(("scaler", StandardScaler()))
        elif scaling == "Normalization":
            numeric_steps.append(("scaler", MinMaxScaler()))
        transformers.append((
            "numeric",
            Pipeline(numeric_steps),
            numeric_columns,
        ))

    observed_categorical_columns = [
        column
        for column in categorical_columns
        if train[column].notna().any()
    ]
    all_missing_categorical_columns = [
        column
        for column in categorical_columns
        if column not in observed_categorical_columns
    ]
    if observed_categorical_columns:
        if categorical_imputation == "constant":
            categorical_imputer = SimpleImputer(
                strategy="constant",
                fill_value=categorical_fill_value,
                keep_empty_features=True,
            )
        else:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
        if categorical_encoding == "One-hot":
            categorical_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        else:
            categorical_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        categorical_pipeline = Pipeline([
            ("imputer", categorical_imputer),
            ("encoder", categorical_encoder),
        ])
        transformers.append((
            "categorical",
            categorical_pipeline,
            observed_categorical_columns,
        ))
    if all_missing_categorical_columns:
        if categorical_encoding == "One-hot":
            empty_categorical_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        else:
            empty_categorical_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        empty_categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=categorical_fill_value,
                    keep_empty_features=True,
                ),
            ),
            ("encoder", empty_categorical_encoder),
        ])
        transformers.append((
            "all_missing_categorical",
            empty_categorical_pipeline,
            all_missing_categorical_columns,
        ))

    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    try:
        train_array = preprocessor.fit_transform(train)
        test_array = preprocessor.transform(test)
        feature_names = preprocessor.get_feature_names_out()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to preprocess train/test features: {exc}") from exc

    processed_train = pd.DataFrame(
        train_array,
        columns=feature_names,
        index=X_train.index,
    )
    processed_test = pd.DataFrame(
        test_array,
        columns=feature_names,
        index=X_test.index,
    )
    metadata = {
        "scaling": scaling,
        "outlier_method": outlier_method,
        "outlier_bounds": outlier_bounds,
        "numeric_imputation": numeric_imputation,
        "categorical_imputation": categorical_imputation,
        "categorical_encoding": categorical_encoding,
        "excluded_categorical_columns": excluded_categorical_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "all_missing_categorical_columns": all_missing_categorical_columns,
        "dropped_all_missing_numeric_columns": dropped_all_missing_numeric_columns,
        "preprocessor": preprocessor,
    }
    return processed_train, processed_test, metadata


def _validated_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")
    return df


def _validated_column(df: pd.DataFrame, column) -> pd.Series:
    _validated_dataframe(df)
    if column not in df.columns:
        raise ValueError(f"Column {column!r} does not exist.")
    return df[column]


def rename_column(
    df: pd.DataFrame,
    column,
    new_name,
) -> pd.DataFrame:
    """Return a copy with one column renamed; reject missing or duplicate names."""
    _validated_column(df, column)
    if not pd.api.types.is_scalar(new_name) or pd.isna(new_name):
        raise ValueError("The new column name must be a non-missing scalar.")
    if isinstance(new_name, str) and not new_name.strip():
        raise ValueError("The new column name must not be empty.")
    if new_name != column and new_name in df.columns:
        raise ValueError(f"Column name {new_name!r} already exists.")
    return df.rename(columns={column: new_name}).copy()


def change_column_dtype(
    df: pd.DataFrame,
    column,
    dtype,
) -> pd.DataFrame:
    """Convert a column to a pandas dtype strictly and return a copy.

    Conversion failures are raised rather than silently replaced with missing
    values. Boolean strings accept true/false, yes/no, y/n, and 1/0.
    """
    series = _validated_column(df, column)
    try:
        target_dtype = pd.api.types.pandas_dtype(dtype)
        if pd.api.types.is_bool_dtype(target_dtype):
            if pd.api.types.is_bool_dtype(series.dtype):
                converted = series.astype(target_dtype)
            else:
                normalized = series.astype("string").str.strip().str.casefold()
                true_values = {"true", "yes", "y", "1"}
                false_values = {"false", "no", "n", "0"}
                valid = normalized.isna() | normalized.isin(
                    true_values | false_values
                )
                if not valid.all():
                    bad_values = series.loc[~valid].drop_duplicates().tolist()
                    raise ValueError(
                        f"Values cannot be interpreted as booleans: {bad_values!r}"
                    )
                converted = normalized.map(
                    lambda value: (
                        pd.NA
                        if pd.isna(value)
                        else value in true_values
                    )
                ).astype(target_dtype)
        elif pd.api.types.is_datetime64_any_dtype(target_dtype):
            converted = pd.to_datetime(series, errors="raise").astype(target_dtype)
        elif pd.api.types.is_numeric_dtype(target_dtype):
            converted = pd.to_numeric(series, errors="raise").astype(target_dtype)
        else:
            converted = series.astype(target_dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Cannot convert column {column!r} to dtype {dtype!r}: {exc}"
        ) from exc

    result = df.copy()
    result[column] = converted
    return result


class DataPreparationHistory:
    """Track immutable DataFrame transformations and support undo/reset."""

    def __init__(self, dataframe: pd.DataFrame):
        self._original = _validated_dataframe(dataframe).copy(deep=True)
        self._current = self._original.copy(deep=True)
        self._undo_stack: list[pd.DataFrame] = []

    @property
    def current(self) -> pd.DataFrame:
        """Return a copy of the current prepared data."""
        return self._current.copy(deep=True)

    @property
    def original(self) -> pd.DataFrame:
        """Return a copy of the initial data."""
        return self._original.copy(deep=True)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def step_count(self) -> int:
        return len(self._undo_stack)

    def apply(
        self,
        transformation: Callable[..., pd.DataFrame],
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Apply a DataFrame transformation and save the prior state for undo."""
        if not callable(transformation):
            raise ValueError("transformation must be callable.")
        transformed = transformation(self.current, *args, **kwargs)
        if not isinstance(transformed, pd.DataFrame):
            raise ValueError("A preparation transformation must return a DataFrame.")
        _validated_dataframe(transformed)
        self._undo_stack.append(self._current.copy(deep=True))
        self._current = transformed.copy(deep=True)
        return self.current

    def undo(self) -> pd.DataFrame:
        """Restore and return the previous state, or raise if there is none."""
        if not self._undo_stack:
            raise ValueError("There is no preparation step to undo.")
        self._current = self._undo_stack.pop()
        return self.current

    def reset(self) -> pd.DataFrame:
        """Restore the original data and clear the undo history."""
        self._current = self._original.copy(deep=True)
        self._undo_stack.clear()
        return self.current


def _numeric_values_and_finite_mask(
    df: pd.DataFrame,
    column,
) -> tuple[pd.Series, np.ndarray]:
    series = _validated_column(df, column)
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError(f"Column {column!r} must have a numeric data type.")
    values = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
    return values, finite_mask


def _finite_numeric_values(df: pd.DataFrame, column) -> pd.Series:
    values, finite_mask = _numeric_values_and_finite_mask(df, column)
    finite_values = values.iloc[np.flatnonzero(finite_mask)]
    if finite_values.empty:
        raise ValueError(f"Column {column!r} has no finite numeric values.")
    return finite_values


def detect_outliers(df: pd.DataFrame, column) -> pd.DataFrame:
    """Return rows beyond the 1.5-IQR fences; missing/non-finite values are ignored."""
    values = _finite_numeric_values(df, column)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    full_values, finite_mask = _numeric_values_and_finite_mask(df, column)
    outlier_mask = np.zeros(len(df), dtype=bool)
    outlier_mask[finite_mask] = (
        (full_values.iloc[np.flatnonzero(finite_mask)] < lower_bound)
        | (full_values.iloc[np.flatnonzero(finite_mask)] > upper_bound)
    ).to_numpy()
    return df.iloc[np.flatnonzero(outlier_mask)].copy()


def remove_outliers(df: pd.DataFrame, column) -> pd.DataFrame:
    """Return a copy without IQR outliers, preserving rows with duplicate indexes."""
    values = _finite_numeric_values(df, column)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    full_values, finite_mask = _numeric_values_and_finite_mask(df, column)
    inlier_mask = ~finite_mask
    inlier_mask[finite_mask] = full_values.iloc[
        np.flatnonzero(finite_mask)
    ].between(lower_bound, upper_bound).to_numpy()
    return df.iloc[np.flatnonzero(inlier_mask)].copy()


def normalize(df: pd.DataFrame, column) -> pd.DataFrame:
    """Min-max scale a finite numeric column and return a copy."""
    values = _finite_numeric_values(df, column)
    if values.size != len(df) or values.index.size != len(df):
        raise ValueError("Cannot scale a column containing missing or non-finite values.")
    result = df.copy()
    result[column] = MinMaxScaler().fit_transform(values.to_frame()).ravel()
    return result


def standardize(df: pd.DataFrame, column) -> pd.DataFrame:
    """Standardize a finite numeric column and return a copy."""
    values = _finite_numeric_values(df, column)
    if values.size != len(df) or values.index.size != len(df):
        raise ValueError("Cannot scale a column containing missing or non-finite values.")
    result = df.copy()
    result[column] = StandardScaler().fit_transform(values.to_frame()).ravel()
    return result


def label_encode(df: pd.DataFrame, column) -> pd.DataFrame:
    """Encode a complete label column; use one-hot encoding for nominal features."""
    series = _validated_column(df, column)
    if series.isna().any():
        raise ValueError("Cannot label-encode a column containing missing values.")
    try:
        encoded = LabelEncoder().fit_transform(series)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} cannot be label-encoded: {exc}") from exc
    result = df.copy()
    result[column] = encoded
    return result


def drop_columns_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without columns that contain missing values."""
    return _validated_dataframe(df).dropna(axis=1, how="any")


def drop_rows_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without rows that contain any missing values."""
    return _validated_dataframe(df).dropna(axis=0, how="any")


def drop_selected_columns(df: pd.DataFrame, columns_to_drop) -> pd.DataFrame:
    """Return a copy without the requested columns; reject unknown column names."""
    _validated_dataframe(df)
    if isinstance(columns_to_drop, (str, bytes)):
        columns = [columns_to_drop]
    elif isinstance(columns_to_drop, Iterable):
        columns = list(columns_to_drop)
    else:
        raise ValueError("columns_to_drop must be a column name or iterable of names.")

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {missing!r}")
    return df.drop(columns=columns).copy()


def get_dummies(df: pd.DataFrame, column) -> pd.DataFrame:
    """One-hot encode a column and return a new DataFrame."""
    _validated_column(df, column)
    return pd.get_dummies(df, columns=[column], drop_first=True)


def fill_missing_values(
    df: pd.DataFrame,
    column,
    method: str = "mean",
) -> pd.DataFrame:
    """Fill missing values in one column and return a copy."""
    series = _validated_column(df, column)

    if method in ("mean", "median"):
        if (
            not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            raise ValueError(f"Method {method!r} requires a numeric column.")
        if series.notna().sum() == 0:
            raise ValueError(f"Column {column!r} has no values to calculate {method}.")
        numeric_series = series.astype("float64")
        fill_value = (
            numeric_series.mean()
            if method == "mean"
            else numeric_series.median()
        )
        filled = numeric_series.fillna(fill_value)
    elif method == "mode":
        modes = series.mode(dropna=True)
        if modes.empty:
            raise ValueError(f"Column {column!r} has no values to calculate mode.")
        filled = series.fillna(modes.iloc[0])
    elif method == "ffill":
        filled = series.ffill()
    elif method == "bfill":
        filled = series.bfill()
    else:
        raise ValueError(
            "Method must be one of: 'mean', 'median', 'mode', 'ffill', 'bfill'."
        )

    if filled.isna().any():
        raise ValueError(
            f"Method {method!r} could not fill every missing value in {column!r}."
        )
    result = df.copy()
    result[column] = filled
    return result


def fill_missing_values_with_value(
    df: pd.DataFrame,
    column,
    value,
) -> pd.DataFrame:
    """Fill missing values with an explicit value and return a copy."""
    series = _validated_column(df, column)
    if not pd.api.types.is_scalar(value) or pd.isna(value):
        raise ValueError("Fill value must not be missing.")
    try:
        filled = series.fillna(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot fill column {column!r} with {value!r}: {exc}") from exc
    result = df.copy()
    result[column] = filled
    return result


def manage_duplicates(df: pd.DataFrame, action: str = "drop") -> pd.DataFrame:
    """Return a copy with duplicate rows handled by the selected policy."""
    _validated_dataframe(df)
    if action in ("drop", "keep_first"):
        return df.drop_duplicates(keep="first").copy()
    if action == "keep_last":
        return df.drop_duplicates(keep="last").copy()
    if action == "drop_all":
        return df.drop_duplicates(keep=False).copy()
    raise ValueError(
        "Action must be one of: 'drop', 'keep_first', 'keep_last', 'drop_all'."
    )
