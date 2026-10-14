
import math

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="DataLens | ML Preparation",
    page_icon="🤖",
    layout="wide",
)

st.title("ML Preparation")
st.caption(
    "Validate your dataset, select a target, "
    "and prepare data for machine learning."
)

# ------------------------------------------------------------
# LOAD THE DATASET
# ------------------------------------------------------------

df = st.session_state.get("working_df")
dataset_name = st.session_state.get("dataset_name")

if df is None or not isinstance(df, pd.DataFrame):
    st.warning(
        "No tabular dataset is loaded. "
        "Return to Home and upload a CSV or Excel file."
    )
    st.stop()

df = df.copy()

st.caption(f"Dataset: {dataset_name or 'Current dataset'}")

# ------------------------------------------------------------
# VALIDATE THE DATASET
# ------------------------------------------------------------

errors = []

if df.empty:
    errors.append("The dataset is empty.")

if not df.columns.is_unique:
    errors.append("Column names must be unique.")

if len(df) < 2:
    errors.append("At least two rows are required.")

if errors:
    for error in errors:
        st.error(error)
    st.stop()

# ------------------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------------------

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features available", f"{len(df.columns):,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------------
# SELECT THE TARGET
# ------------------------------------------------------------

st.subheader("1. Select your target")

target_column = st.selectbox(
    "Which column do you want to predict?",
    options=list(df.columns),
    index=None,
    placeholder="Choose a target column",
)

if target_column is None:
    st.info("Choose a target column to continue.")
    st.stop()

y = df[target_column].copy()
X = df.drop(columns=[target_column]).copy()

if X.shape[1] == 0:
    st.error("You need at least one feature besides the target.")
    st.stop()

if y.isna().any():
    st.error(
        f"The target column has {int(y.isna().sum())} missing values. "
        "Clean these values before splitting."
    )
    st.stop()

if y.nunique() < 2:
    st.error("The target must contain at least two distinct values.")
    st.stop()

st.write(f"**Target:** `{target_column}`")
st.write(f"**Features:** {X.shape[1]}")

with st.expander("Preview features and target"):
    left, right = st.columns(2)
    left.write("Features (X)")
    left.dataframe(X.head(), use_container_width=True)
    right.write("Target (y)")
    right.dataframe(y.head(), use_container_width=True)

# ------------------------------------------------------------
# CONFIGURE THE TRAIN/TEST SPLIT
# ------------------------------------------------------------

st.subheader("2. Configure the split")

with st.form("split_configuration"):
    test_size = st.slider(
        "Test set size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    task_type = st.selectbox(
        "Problem type",
        ["Auto-detect", "Classification", "Regression"],
    )

    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=7,
        step=1,
    )

    stratify_classes = st.checkbox(
        "Use stratified splitting for classification",
        value=True,
    )

    submitted = st.form_submit_button(
        "Create train/test split",
        type="primary",
    )

# ------------------------------------------------------------
# SPLIT THE DATA
# ------------------------------------------------------------

if submitted:
    is_classification = (
        task_type == "Classification"
        or (
            task_type == "Auto-detect"
            and (
                not pd.api.types.is_numeric_dtype(y)
                or y.nunique()
                <= min(20, max(2, int(len(y) * 0.05)))
            )
        )
    )

    stratify = None
    messages = []

    if is_classification and stratify_classes:
        counts = y.value_counts()
        n_classes = len(counts)
        n_test = math.ceil(len(y) * test_size / 100)
        n_train = len(y) - n_test

        can_stratify = (
            counts.min() >= 2
            and n_test >= n_classes
            and n_train >= n_classes
        )

        if can_stratify:
            stratify = y
        else:
            messages.append(
                "Stratification is not possible with these class counts "
                "and split settings. A regular random split will be used."
            )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=int(random_state),
            stratify=stratify,
        )

        st.session_state["ml_preparation_result"] = {
            "dataset_name": dataset_name,
            "target_column": target_column,
            "task_type": task_type,
            "test_size": test_size,
            "random_state": int(random_state),
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        st.success("Train/test split created successfully.")

        for message in messages:
            st.warning(message)

    except ValueError as exc:
        st.error(f"Unable to split the dataset: {exc}")

# ------------------------------------------------------------
# DISPLAY THE RESULT
# ------------------------------------------------------------

result = st.session_state.get("ml_preparation_result")

if (
    result is not None
    and result["dataset_name"] == dataset_name
    and result["target_column"] == target_column
):
    st.subheader("3. Split results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", f"{len(result['X_train']):,}")
    c2.metric("Testing rows", f"{len(result['X_test']):,}")
    c3.metric("Features", f"{result['X_train'].shape[1]:,}")

    with st.expander("Preview training data"):
        st.dataframe(
            result["X_train"].assign(
                **{target_column: result["y_train"]}
            ),
            use_container_width=True,
        )

    with st.expander("Preview testing data"):
        st.dataframe(
            result["X_test"].assign(
                **{target_column: result["y_test"]}
            ),
            use_container_width=True,
        )

    train_df = result["X_train"].copy()
    train_df[target_column] = result["y_train"]

    test_df = result["X_test"].copy()
    test_df[target_column] = result["y_test"]

    d1, d2 = st.columns(2)

    d1.download_button(
        "Download training CSV",
        data=train_df.to_csv(index=False).encode("utf-8"),
        file_name="training_data.csv",
        mime="text/csv",
    )

    d2.download_button(
        "Download testing CSV",
        data=test_df.to_csv(index=False).encode("utf-8"),
        file_name="testing_data.csv",
        mime="text/csv",
    )
