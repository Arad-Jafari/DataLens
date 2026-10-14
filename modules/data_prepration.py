import pandas as pd
from sklearn.model_selection import train_test_split

def machine_learning_prepration(df,y_column):
    X = df.drop(f'{y_column}', axis=1) #Data frame
    y = df[f'{y_column}'] #Series
def training_test(X,y):
    train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=7)
    return train_X,test_X,train_y,test_y




import math

import pandas as pd
from sklearn.model_selection import train_test_split


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


def prepare_features_target(df, target_column):
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
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if len(X) < 2:
        raise ValueError("At least two rows are required.")

    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1.")

    warnings = []

    # Classification detection is a heuristic.
    if task_type == "Classification":
        is_classification = True
    elif task_type == "Regression":
        is_classification = False
    else:
        is_classification = (
            not pd.api.types.is_numeric_dtype(y)
            or y.nunique()
            <= min(20, max(2, int(len(y) * 0.05)))
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
    except ValueError as exc:
        raise ValueError(f"Unable to split dataset: {exc}") from exc

    return X_train, X_test, y_train, y_test, warnings
