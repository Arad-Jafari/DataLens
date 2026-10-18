import pandas as pd
import streamlit as st

from modules.data_preparation import (
    prepare_features_target,
    split_dataset,
    validate_dataset,
)
from modules.data_utils import get_dataset_summary
from modules.ui import (
    current_dataset,
    ensure_app_session,
    inject_base_css,
    render_sidebar,
)

st.set_page_config(
    page_title="DataLens | ML Preparation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_app_session()
inject_base_css()
render_sidebar()

st.title("ML Preparation")
st.caption(
    "Validate your dataset, select a target, "
    "and prepare data for machine learning."
)

df = current_dataset()
dataset_name = st.session_state.get("dataset_name")
dataset_id = st.session_state.get("processed_upload_id")

if df is None or not isinstance(df, pd.DataFrame):
    st.warning(
        "No tabular dataset is loaded. "
        "Return to Home and upload a CSV or Excel file."
    )
    st.stop()

df = df.copy()
st.caption(f"Dataset: {dataset_name or 'Current dataset'}")

errors = validate_dataset(df)
if errors:
    for error in errors:
        st.error(error)
    st.stop()

summary = get_dataset_summary(df)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{summary['row_count']:,}")
c2.metric("Features available", f"{summary['column_count']:,}")
c3.metric("Missing cells", f"{summary['missing_count']:,}")

st.subheader("Dataset preview")
st.dataframe(df.head(10), use_container_width=True)

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

try:
    X, y = prepare_features_target(df, target_column)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.write(f"**Target:** `{target_column}`")
st.write(f"**Features:** {X.shape[1]}")

with st.expander("Preview features and target"):
    left, right = st.columns(2)
    left.write("Features (X)")
    left.dataframe(X.head(), use_container_width=True)
    right.write("Target (y)")
    right.dataframe(y.head(), use_container_width=True)

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

if submitted:
    try:
        X_train, X_test, y_train, y_test, messages = split_dataset(
            X,
            y,
            test_size=test_size / 100,
            random_state=int(random_state),
            task_type=task_type,
            stratify_classification=stratify_classes,
        )
        st.session_state["ml_preparation_result"] = {
            "dataset_id": dataset_id,
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

result = st.session_state.get("ml_preparation_result")

if (
    result is not None
    and result.get("dataset_id") == dataset_id
    and result.get("dataset_name") == dataset_name
    and result.get("target_column") == target_column
):
    st.subheader("3. Split results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", f"{len(result['X_train']):,}")
    c2.metric("Testing rows", f"{len(result['X_test']):,}")
    c3.metric("Features", f"{result['X_train'].shape[1]:,}")

    with st.expander("Preview training data"):
        st.dataframe(
            result["X_train"].assign(**{target_column: result["y_train"]}),
            use_container_width=True,
        )

    with st.expander("Preview testing data"):
        st.dataframe(
            result["X_test"].assign(**{target_column: result["y_test"]}),
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
