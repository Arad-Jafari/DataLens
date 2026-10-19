from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.data_preparation import (
    DataPreparationHistory,
    change_column_dtype,
    drop_rows_with_missing_values,
    drop_selected_columns,
    manage_duplicates,
    prepare_features_target,
    preprocess_split_features,
    rename_column,
    split_dataset,
    validate_dataset,
)
from modules.data_utils import get_dataset_summary, numeric_columns
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
    "Edit and clean a working copy, split features and target, "
    "then optionally apply the cleaned dataset to the workspace."
)

source_df = current_dataset()
dataset_name = st.session_state.get("dataset_name") or "Current dataset"
dataset_id = st.session_state.get("processed_upload_id")

if not isinstance(source_df, pd.DataFrame):
    st.warning(
        "No tabular dataset is loaded. Return to Home and upload a CSV or Excel file."
    )
    st.stop()

if dataset_id is None:
    dataset_id = (dataset_name, source_df.shape)
history_key = (dataset_id, dataset_name)
history_key = (
    history_key,
    st.session_state.get("dataset_revision", 0),
)
if st.session_state.get("ml_prep_history_key") != history_key:
    st.session_state["ml_prep_history"] = DataPreparationHistory(source_df)
    st.session_state["ml_prep_history_key"] = history_key
    st.session_state["ml_prep_revision"] = 0
    st.session_state.pop("ml_preparation_result", None)
    st.session_state.pop("ml_preprocessing_result", None)
    st.session_state.pop("ml_split_success", None)
    st.session_state.pop("ml_split_error", None)

history: DataPreparationHistory = st.session_state["ml_prep_history"]
df = history.current


def apply_step(transformation, *args, **kwargs) -> None:
    try:
        history.apply(transformation, *args, **kwargs)
        st.session_state["ml_prep_revision"] += 1
        st.session_state.pop("ml_preparation_result", None)
        st.session_state.pop("ml_preprocessing_result", None)
        st.session_state.pop("ml_split_success", None)
        st.session_state.pop("ml_split_error", None)
        st.session_state.pop("ml_prep_target", None)
        st.rerun()
    except (TypeError, ValueError) as exc:
        st.error(f"Preparation step failed: {exc}")


def create_train_test_split() -> None:
    st.session_state.pop("ml_preparation_result", None)
    st.session_state.pop("ml_preprocessing_result", None)
    st.session_state.pop("ml_split_success", None)
    st.session_state.pop("ml_split_error", None)
    try:
        target_column = st.session_state.get("ml_prep_target")
        if target_column is None:
            raise ValueError("Choose a target column before creating the split.")
        X, y = prepare_features_target(df, target_column)
        test_size = int(st.session_state["ml_prep_test_size"])
        task_type = st.session_state["ml_prep_task_type"]
        random_state = int(st.session_state["ml_prep_random_state"])
        stratify_classes = st.session_state["ml_prep_stratify"]
        X_train, X_test, y_train, y_test, messages = split_dataset(
            X,
            y,
            test_size=test_size / 100,
            random_state=random_state,
            task_type=task_type,
            stratify_classification=stratify_classes,
        )
        st.session_state["ml_preparation_result"] = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "revision": st.session_state["ml_prep_revision"],
            "target_column": target_column,
            "task_type": task_type,
            "test_size": test_size,
            "random_state": random_state,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "messages": messages,
        }
        st.session_state.pop("ml_preprocessing_result", None)
        st.session_state.pop("ml_split_error", None)
        st.session_state["ml_split_success"] = (
            f"Train/test split created: {len(X_train):,} training rows and "
            f"{len(X_test):,} test rows. Continue with Model preprocessing."
        )
        st.session_state["ml_prep_stage_tabs"] = "4. Model preprocessing"
    except (KeyError, TypeError, ValueError) as exc:
        st.session_state["ml_split_error"] = str(exc)


def clear_split_results() -> None:
    st.session_state.pop("ml_preparation_result", None)
    st.session_state.pop("ml_preprocessing_result", None)
    st.session_state.pop("ml_split_success", None)
    st.session_state.pop("ml_split_error", None)


def change_workflow_mode() -> None:
    clear_split_results()
    if st.session_state["ml_prep_workflow_mode"].startswith("Unsupervised"):
        st.session_state["ml_prep_stage_tabs"] = "4. Model preprocessing"
    else:
        st.session_state["ml_prep_stage_tabs"] = "3. Target and split"


top_left, top_right, top_status = st.columns([1, 1, 3])
if top_left.button(
    "Undo last change",
    disabled=not history.can_undo,
    key="ml_prep_undo",
):
    try:
        history.undo()
        st.session_state["ml_prep_revision"] += 1
        st.session_state.pop("ml_preparation_result", None)
        st.session_state.pop("ml_preprocessing_result", None)
        st.session_state.pop("ml_split_success", None)
        st.session_state.pop("ml_split_error", None)
        st.session_state.pop("ml_prep_target", None)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

if top_right.button(
    "Reset preparation",
    disabled=not history.can_undo,
    key="ml_prep_reset",
):
    history.reset()
    st.session_state["ml_prep_revision"] += 1
    st.session_state.pop("ml_preparation_result", None)
    st.session_state.pop("ml_preprocessing_result", None)
    st.session_state.pop("ml_split_success", None)
    st.session_state.pop("ml_split_error", None)
    st.session_state.pop("ml_prep_target", None)
    st.rerun()

top_status.caption(
    f"Dataset: {dataset_name} · {history.step_count} preparation "
    f"{'step' if history.step_count == 1 else 'steps'}"
)

summary = get_dataset_summary(df)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{summary['row_count']:,}")
m2.metric("Columns", f"{summary['column_count']:,}")
m3.metric("Missing cells", f"{summary['missing_count']:,}")
m4.metric("Duplicate rows", f"{summary['duplicate_count']:,}")

column_stage, cleaning_stage, split_stage, preprocessing_stage, review_stage = st.tabs(
    [
        "1. Column changes",
        "2. Data cleaning",
        "3. Target and split",
        "4. Model preprocessing",
        "5. Review and apply",
    ],
    key="ml_prep_stage_tabs",
    on_change="rerun",
)

with column_stage:
    st.subheader("Column changes")
    st.caption("Edits affect this preparation copy. Use Undo or Reset to reverse them.")

    if df.columns.empty:
        st.info("There are no columns to edit.")
    else:
        rename_col, rename_name = st.columns(2)
        with rename_col:
            selected_rename_column = st.selectbox(
                "Column to rename",
                list(df.columns),
                key="ml_prep_rename_column",
            )
        with rename_name:
            new_column_name = st.text_input(
                "New column name",
                value=str(selected_rename_column),
                key="ml_prep_new_column_name",
            )
        if st.button("Rename column", key="ml_prep_rename_button"):
            apply_step(rename_column, selected_rename_column, new_column_name)

        st.divider()
        numeric_text_columns = [
            column
            for column in df.columns
            if (
                pd.api.types.is_object_dtype(df[column].dtype)
                or pd.api.types.is_string_dtype(df[column].dtype)
            )
            and not df[column].dropna().empty
            and pd.to_numeric(
                df[column].dropna(),
                errors="coerce",
            ).notna().all()
        ]
        st.markdown("#### Convert numeric-looking text to numbers")
        if numeric_text_columns:
            numeric_text_column = st.selectbox(
                "Text column containing numeric values",
                numeric_text_columns,
                key="ml_prep_numeric_text_column",
            )
            st.caption(
                "Only columns where every non-missing value is numeric are "
                "listed here. Mixed identifiers such as ticket numbers stay text."
            )
            if st.button(
                "Convert selected column to numeric",
                key="ml_prep_numeric_text_button",
            ):
                apply_step(
                    change_column_dtype,
                    numeric_text_column,
                    "Float64",
                )
        else:
            st.info(
                "No text columns containing only numeric values were found. "
                "Text categories such as Sex (male/female) are handled separately."
            )
            st.caption(
                "To convert categories like Sex to numeric features, choose "
                "One-hot encoding in Model preprocessing. It creates separate "
                "0/1 features without implying an order."
            )

        st.divider()
        dtype_col, dtype_choice = st.columns(2)
        with dtype_col:
            selected_dtype_column = st.selectbox(
                "Column to convert",
                list(df.columns),
                key="ml_prep_dtype_column",
            )
        with dtype_choice:
            requested_dtype = st.selectbox(
                "Convert to",
                [
                    "string",
                    "category",
                    "Int64",
                    "Float64",
                    "datetime64[ns]",
                    "boolean",
                ],
                key="ml_prep_dtype_choice",
            )
        st.caption(
            f"Current type: {df[selected_dtype_column].dtype}. "
            "Invalid values will be reported, not silently discarded."
        )
        if st.button("Change column type", key="ml_prep_dtype_button"):
            apply_step(change_column_dtype, selected_dtype_column, requested_dtype)

        st.divider()
        columns_to_remove = st.multiselect(
            "Columns to remove",
            list(df.columns),
            key="ml_prep_remove_columns",
        )
        if st.button(
            "Remove selected columns",
            key="ml_prep_remove_columns_button",
            disabled=not columns_to_remove,
        ):
            apply_step(drop_selected_columns, columns_to_remove)

    st.markdown("#### Current columns")
    st.dataframe(
        pd.DataFrame({
            "Column": [str(column) for column in df.columns],
            "Data type": [str(dtype) for dtype in df.dtypes],
            "Missing values": df.isna().sum().values,
        }),
        width="stretch",
        hide_index=True,
    )

with cleaning_stage:
    st.subheader("Data cleaning")
    st.caption("Preview the impact where possible. Each successful action can be undone.")

    st.markdown("#### Missing values")
    unsupervised_cleaning = st.session_state.get(
        "ml_prep_workflow_mode",
        "Supervised learning (target and train/test split)",
    ).startswith("Unsupervised")
    if unsupervised_cleaning:
        st.info(
            "For clustering, keep missing feature values here and impute them "
            "in Model preprocessing; imputation is fitted on the full dataset. "
            "There is no target column to clean."
        )
    else:
        st.info(
            "Keep missing feature values here and choose how to impute them "
            "after the split. Imputation values are learned from training "
            "features only. Missing target values must be removed before splitting."
        )
    missing_action = st.selectbox(
        "Optional structural cleanup",
        [
            "Keep missing values for model imputation",
            "Drop rows containing missing values",
        ],
        key="ml_prep_missing_action",
    )
    if missing_action == "Drop rows containing missing values":
        affected_rows = int(df.isna().any(axis=1).sum())
        st.caption(f"Rows that would be removed: {affected_rows:,}")
        if st.button(
            "Drop rows with missing values",
            key="ml_prep_drop_missing_rows",
            disabled=affected_rows == 0,
        ):
            apply_step(drop_rows_with_missing_values)
    if unsupervised_cleaning:
        st.caption(
            "To impute rather than discard values, keep this option selected "
            "and configure imputation under Model preprocessing; no split is needed."
        )
    else:
        st.caption(
            "To impute rather than discard values, keep this option selected and "
            "configure numeric and categorical imputation under Model preprocessing."
        )
    st.divider()
    st.markdown("#### Duplicate rows")
    duplicate_action = st.selectbox(
        "Duplicate handling",
        [
            ("Keep first copy", "keep_first"),
            ("Keep last copy", "keep_last"),
            ("Remove every copy of duplicated rows", "drop_all"),
        ],
        format_func=lambda option: option[0],
        key="ml_prep_duplicate_action",
    )
    duplicate_count = int(df.duplicated(keep=False).sum())
    st.caption(f"Rows involved in duplicate groups: {duplicate_count:,}")
    if st.button(
        "Apply duplicate handling",
        key="ml_prep_duplicate_button",
        disabled=duplicate_count == 0,
    ):
        apply_step(manage_duplicates, duplicate_action[1])

with split_stage:
    workflow_mode = st.selectbox(
        "Preparation workflow",
        [
            "Supervised learning (target and train/test split)",
            "Unsupervised learning (clustering, no target)",
        ],
        key="ml_prep_workflow_mode",
        on_change=change_workflow_mode,
    )
    unsupervised_mode = workflow_mode.startswith("Unsupervised")
    if unsupervised_mode:
        st.info(
            "Clustering does not need a target column or train/test split. "
            "Continue to Model preprocessing to impute, encode, scale, and "
            "optionally cap outliers using the full dataset. Remove IDs and "
            "any columns that should not influence cluster membership first."
        )
    else:
        st.subheader("Select the prediction target")
        st.info(
            "Data-dependent imputation, encoding, scaling, and outlier "
            "thresholds are learned only from training data in the next stage."
        )
        errors = validate_dataset(df)
        if errors:
            for error in errors:
                st.error(error)
        else:
            target_column = st.selectbox(
                "Target column (y)",
                options=list(df.columns),
                index=None,
                placeholder="Choose the value to predict",
                key="ml_prep_target",
                on_change=clear_split_results,
            )

            if target_column is None:
                st.info("Choose a target column to continue.")
            else:
                if st.session_state.get("ml_split_error"):
                    st.error(
                        "Unable to create the train/test split: "
                        + st.session_state.pop("ml_split_error")
                    )
                try:
                    X, y = prepare_features_target(df, target_column)
                    x_preview, y_preview = st.columns(2)
                    x_preview.markdown(f"#### Features (X) · {X.shape[1]} columns")
                    x_preview.dataframe(X.head(10), width="stretch")
                    y_preview.markdown(f"#### Target (y) · {target_column}")
                    y_preview.dataframe(
                        y.to_frame(name=target_column).head(10),
                        width="stretch",
                    )

                    with st.form("ml_prep_split_configuration"):
                        st.slider(
                            "Test set size (%)",
                            min_value=10,
                            max_value=40,
                            value=20,
                            step=5,
                            key="ml_prep_test_size",
                        )
                        st.selectbox(
                            "Problem type",
                            ["Auto-detect", "Classification", "Regression"],
                            key="ml_prep_task_type",
                        )
                        st.number_input(
                            "Random seed",
                            min_value=0,
                            max_value=999999,
                            value=7,
                            step=1,
                            key="ml_prep_random_state",
                        )
                        st.checkbox(
                            "Use stratified splitting for classification",
                            value=True,
                            key="ml_prep_stratify",
                        )
                        st.form_submit_button(
                            "Create train/test split",
                            type="primary",
                            on_click=create_train_test_split,
                        )
                except ValueError as exc:
                    st.error(str(exc))

with preprocessing_stage:
    unsupervised_mode = st.session_state.get(
        "ml_prep_workflow_mode",
        "Supervised learning (target and train/test split)",
    ).startswith("Unsupervised")
    fit_data_label = "full dataset" if unsupervised_mode else "training set"
    st.subheader(
        "Clustering feature preprocessing"
        if unsupervised_mode
        else "Leakage-safe feature preprocessing"
    )
    if st.session_state.get("ml_split_success") and not unsupervised_mode:
        st.success(st.session_state["ml_split_success"])
    if unsupervised_mode:
        st.info(
            "No target or split is needed. Transformations are fitted and "
            "applied to the full feature dataset for clustering. If you later "
            "evaluate a clustering model on held-out data, create a separate "
            "split and fit transformations on the training portion only."
        )
        st.caption(
            "Outlier values are capped using thresholds learned from the "
            "full dataset; no rows are removed."
        )
    else:
        st.caption(
            "Choose missing-value strategies here. Imputers, categorical "
            "encoders, scaling parameters, and optional outlier caps are fitted "
            "using X_train only, then applied to X_train and X_test. This "
            "prevents test data from influencing training-time decisions. "
            "Outliers are capped, not used to remove test rows."
        )
    result = st.session_state.get("ml_preparation_result")
    split_is_current = (
        result is not None
        and result.get("dataset_id") == dataset_id
        and result.get("revision") == st.session_state["ml_prep_revision"]
    )
    preprocessing_is_ready = unsupervised_mode or split_is_current
    if not preprocessing_is_ready:
        st.info("Create a train/test split before configuring preprocessing.")
    else:
        feature_reference = df if unsupervised_mode else result["X_train"]
        categorical_candidates = [
            column
            for column in feature_reference.columns
            if not (
                pd.api.types.is_numeric_dtype(feature_reference[column].dtype)
                and not pd.api.types.is_bool_dtype(feature_reference[column].dtype)
            )
        ]
        with st.form("ml_model_preprocessing_form"):
            scaling = st.selectbox(
                "Numeric feature scaling",
                ["None", "Standardization", "Normalization"],
                help=(
                    "None keeps numeric values unchanged. Standardization "
                    "uses the fitted mean and standard deviation. Normalization "
                    "uses the fitted minimum and maximum."
                ),
            )
            st.caption({
                "None": "Keep numeric feature scales unchanged.",
                "Standardization": (
                    "Centers values around 0 and scales them by standard "
                    "deviation; useful when feature scales differ."
                ),
                "Normalization": (
                    f"Scales values to 0–1 using {fit_data_label} minimums "
                    "and maximums; supervised test values may fall outside 0–1."
                ),
            }[scaling])
            categorical_columns = st.multiselect(
                "Categorical columns to encode",
                options=categorical_candidates,
                default=[],
                key=(
                    f"ml_prep_categorical_columns_"
                    f"{'unsupervised' if unsupervised_mode else 'supervised'}_"
                    f"{st.session_state['ml_prep_revision']}"
                ),
                help=(
                    "Only selected columns are encoded. Unselected categorical "
                    "columns are left out of the processed model features. "
                    "Select relevant categories such as Sex; avoid identifiers "
                    "and free-text fields."
                ),
            )
            categorical_encoding = st.radio(
                "Categorical encoding method",
                ["One-hot encoding", "Ordinal encoding"],
                horizontal=True,
                help=(
                    "Choose how selected categorical columns are converted "
                    "to numeric features. One-hot creates binary columns "
                    "without implying order. Ordinal maps categories to "
                    "numbers and can imply an order."
                ),
                key=(
                    f"ml_prep_categorical_encoding_"
                    f"{'unsupervised' if unsupervised_mode else 'supervised'}_"
                    f"{st.session_state['ml_prep_revision']}"
                ),
            )
            if categorical_columns:
                st.caption({
                    "One-hot encoding": (
                        "Creates a 0/1 feature for each category; suitable for "
                        "unordered values such as Sex."
                    ),
                    "Ordinal encoding": (
                        "Maps categories to integer codes, which can imply an "
                        "arbitrary order. Unseen test categories map to -1."
                    ),
                }[categorical_encoding])
            else:
                st.caption(
                    "Choose the encoding method above, then select the "
                    "categorical columns it should apply to. Unselected "
                    "categorical columns are excluded from processed features."
                )
            numeric_imputation = st.selectbox(
                "Numeric missing-value strategy",
                ["Median", "Mean", "Constant value"],
                key="ml_prep_numeric_imputation",
                help=(
                    "Median and Mean are estimated from the fitting data. "
                    "Constant value uses the number you enter."
                ),
            )
            st.caption({
                "Median": (
                    f"Fills missing numbers with the {fit_data_label} median; "
                    "less affected by extreme values."
                ),
                "Mean": (
                    f"Fills missing numbers with the {fit_data_label} average; "
                    "sensitive to extreme values."
                ),
                "Constant value": (
                    "Fills missing numbers with the fixed value you specify."
                ),
            }[numeric_imputation])
            numeric_fill_value = 0.0
            if numeric_imputation == "Constant value":
                numeric_fill_value = st.number_input(
                    "Numeric fill value",
                    value=0.0,
                    key="ml_prep_numeric_fill_value",
                    help="This fixed number replaces each missing numeric value.",
                )
            categorical_imputation = st.selectbox(
                "Categorical missing-value strategy",
                ["Most frequent", "Constant value"],
                key="ml_prep_categorical_imputation",
                help=(
                    "Most frequent is learned from the fitting data. "
                    "Constant value uses the text you enter."
                ),
            )
            st.caption({
                "Most frequent": (
                    "Fills missing categories with the most common category "
                    f"in the {fit_data_label}."
                ),
                "Constant value": (
                    "Fills missing categories with the text value you specify."
                ),
            }[categorical_imputation])
            categorical_fill_value = "(Missing)"
            if categorical_imputation == "Constant value":
                categorical_fill_value = st.text_input(
                    "Categorical fill value",
                    value="(Missing)",
                    key="ml_prep_categorical_fill_value",
                    help=(
                        "This text replaces missing categories before encoding."
                    ),
                )
            outlier_method = st.selectbox(
                "Outlier capping",
                ["None", "IQR", "Percentile", "Z-score"],
                help=(
                    "None leaves values unchanged. IQR caps beyond quartile "
                    "limits; Percentile caps at selected percentiles; Z-score "
                    "caps beyond three standard deviations."
                ),
            )
            st.caption({
                "None": "Leave numeric values uncapped.",
                "IQR": (
                    "Caps values outside Q1/Q3 ± the selected IQR multiplier; "
                    f"robust to extreme values when estimating limits from "
                    f"the {fit_data_label}."
                ),
                "Percentile": (
                    f"Caps values below the lower percentile and above the "
                    f"upper percentile, calculated from the {fit_data_label}. "
                    "For example, 1 and 99 use the 1st- and 99th-percentile "
                    "values as cutoffs. This caps values; it does not remove rows."
                ),
                "Z-score": (
                    f"Caps beyond three standard deviations from the "
                    f"{fit_data_label} mean; works best for roughly "
                    "bell-shaped data."
                ),
            }[outlier_method] + (
                " No rows are removed."
                if unsupervised_mode
                else " Test rows are kept; only their values are capped."
            ))
            iqr_multiplier = 1.5
            lower_percentile = 1.0
            upper_percentile = 99.0
            if outlier_method == "IQR":
                iqr_multiplier = st.number_input(
                    "IQR multiplier",
                    min_value=0.0,
                    value=1.5,
                    step=0.25,
                    help=(
                        "Controls how far beyond Q1 and Q3 values are capped. "
                        "The common default is 1.5."
                    ),
                )
            elif outlier_method == "Percentile":
                lower_percentile = st.number_input(
                    "Lower percentile",
                    min_value=0.0,
                    max_value=99.0,
                    value=1.0,
                    step=1.0,
                    help=(
                        "The lower value threshold, from 0 to 99. For example, "
                        "1 means the value at the 1st percentile of the fitting "
                        "data; lower values are capped to this threshold."
                    ),
                )
                upper_percentile = st.number_input(
                    "Upper percentile",
                    min_value=1.0,
                    max_value=100.0,
                    value=99.0,
                    step=1.0,
                    help=(
                        "The upper value threshold, from 1 to 100. For example, "
                        "99 means the value at the 99th percentile of the fitting "
                        "data; higher values are capped to this threshold."
                    ),
                )
            preprocess_clicked = st.form_submit_button(
                (
                    "Prepare full dataset for clustering"
                    if unsupervised_mode
                    else "Apply preprocessing to train and test (fit on train only)"
                ),
                type="primary",
            )

        if preprocess_clicked:
            try:
                X_fit = df if unsupervised_mode else result["X_train"]
                X_apply = df if unsupervised_mode else result["X_test"]
                X_train_processed, X_test_processed, metadata = (
                    preprocess_split_features(
                        X_fit,
                        X_apply,
                        scaling=scaling,
                        outlier_method=outlier_method,
                        iqr_multiplier=float(iqr_multiplier),
                        lower_percentile=float(lower_percentile),
                        upper_percentile=float(upper_percentile),
                        numeric_imputation={
                            "Median": "median",
                            "Mean": "mean",
                            "Constant value": "constant",
                        }[numeric_imputation],
                        numeric_fill_value=float(numeric_fill_value),
                        categorical_imputation={
                            "Most frequent": "most_frequent",
                            "Constant value": "constant",
                        }[categorical_imputation],
                        categorical_fill_value=categorical_fill_value,
                        categorical_encoding={
                            "One-hot encoding": "One-hot",
                            "Ordinal encoding": "Ordinal",
                        }.get(categorical_encoding),
                        categorical_columns=categorical_columns,
                    )
                )
                st.session_state["ml_preprocessing_result"] = {
                    "dataset_id": dataset_id,
                    "revision": st.session_state["ml_prep_revision"],
                    "mode": "unsupervised" if unsupervised_mode else "supervised",
                    "target_column": (
                        None if unsupervised_mode else result["target_column"]
                    ),
                    "X_train": X_train_processed,
                    "X_test": X_test_processed,
                    "metadata": metadata,
                }
                if unsupervised_mode:
                    st.success(
                        "Clustering features prepared using the full dataset."
                    )
                else:
                    st.success(
                        "Preprocessing fitted on training data and applied."
                    )
            except ValueError as exc:
                st.error(f"Unable to preprocess features: {exc}")

        fitted = st.session_state.get("ml_preprocessing_result")
        if (
            fitted is not None
            and fitted.get("dataset_id") == dataset_id
            and fitted.get("revision") == st.session_state["ml_prep_revision"]
        ):
            if fitted.get("mode") == "unsupervised":
                st.markdown("#### Prepared clustering features")
                st.caption(
                    f"All {len(fitted['X_train']):,} rows · "
                    f"{fitted['X_train'].shape[1]:,} numeric features"
                )
                st.dataframe(fitted["X_train"].head(10), width="stretch")
            else:
                st.markdown("#### Processed feature previews")
                x_train_col, x_test_col = st.columns(2)
                x_train_col.markdown(
                    f"##### X_train · {len(fitted['X_train']):,} rows, "
                    f"{fitted['X_train'].shape[1]:,} features"
                )
                x_train_col.dataframe(fitted["X_train"].head(10), width="stretch")
                x_test_col.markdown(
                    f"##### X_test · {len(fitted['X_test']):,} rows, "
                    f"{fitted['X_test'].shape[1]:,} features"
                )
                x_test_col.dataframe(fitted["X_test"].head(10), width="stretch")
            st.caption(
                f"Numeric imputation: {fitted['metadata']['numeric_imputation']} · "
                f"Categorical imputation: "
                f"{fitted['metadata']['categorical_imputation']} · "
                f"Categorical encoding: "
                f"{fitted['metadata']['categorical_encoding']} · "
                f"Scaling: {fitted['metadata']['scaling']} · "
                f"Outlier caps: {fitted['metadata']['outlier_method']} · "
                f"Numeric features: {len(fitted['metadata']['numeric_columns'])} · "
                f"Categorical features encoded: "
                f"{len(fitted['metadata']['categorical_columns'])}"
            )
            excluded_categorical = fitted["metadata"][
                "excluded_categorical_columns"
            ]
            if excluded_categorical:
                st.warning(
                    "Categorical features excluded because they were not "
                    "selected for encoding: "
                    + ", ".join(str(column) for column in excluded_categorical)
                )
            dropped_empty = (
                fitted["metadata"]["dropped_all_missing_numeric_columns"]
            )
            if dropped_empty:
                st.warning(
                    "Dropped numeric features with no finite training values: "
                    + ", ".join(str(column) for column in dropped_empty)
                )

with review_stage:
    st.subheader("Review prepared dataset")
    st.write(
        f"Starting data: **{history.original.shape[0]:,} rows × "
        f"{history.original.shape[1]:,} columns**"
    )
    st.write(
        f"Prepared data: **{df.shape[0]:,} rows × {df.shape[1]:,} columns**"
    )
    st.dataframe(df.head(20), width="stretch")

    result = st.session_state.get("ml_preparation_result")
    if (
        result is not None
        and result.get("dataset_id") == dataset_id
        and result.get("revision") == st.session_state["ml_prep_revision"]
    ):
        st.markdown("#### Train/test split")
        st.caption(
            f"Target: {result['target_column']} · "
            f"Problem type: {result['task_type']} · "
            f"Test size: {result['test_size']}%"
        )
        for message in result["messages"]:
            st.warning(message)

        metrics = st.columns(4)
        metrics[0].metric("X_train rows", f"{len(result['X_train']):,}")
        metrics[1].metric("y_train rows", f"{len(result['y_train']):,}")
        metrics[2].metric("X_test rows", f"{len(result['X_test']):,}")
        metrics[3].metric("y_test rows", f"{len(result['y_test']):,}")

        x_train_col, y_train_col = st.columns(2)
        with x_train_col:
            st.markdown("##### X_train")
            st.dataframe(result["X_train"].head(10), width="stretch")
            st.download_button(
                "Download X_train",
                data=result["X_train"].to_csv(index=False).encode("utf-8"),
                file_name="X_train.csv",
                mime="text/csv",
                key="ml_download_X_train",
            )
        with y_train_col:
            st.markdown("##### y_train")
            y_train = result["y_train"].to_frame(name=result["target_column"])
            st.dataframe(y_train.head(10), width="stretch")
            st.download_button(
                "Download y_train",
                data=y_train.to_csv(index=False).encode("utf-8"),
                file_name="y_train.csv",
                mime="text/csv",
                key="ml_download_y_train",
            )

        x_test_col, y_test_col = st.columns(2)
        with x_test_col:
            st.markdown("##### X_test")
            st.dataframe(result["X_test"].head(10), width="stretch")
            st.download_button(
                "Download X_test",
                data=result["X_test"].to_csv(index=False).encode("utf-8"),
                file_name="X_test.csv",
                mime="text/csv",
                key="ml_download_X_test",
            )
        with y_test_col:
            st.markdown("##### y_test")
            y_test = result["y_test"].to_frame(name=result["target_column"])
            st.dataframe(y_test.head(10), width="stretch")
            st.download_button(
                "Download y_test",
                data=y_test.to_csv(index=False).encode("utf-8"),
                file_name="y_test.csv",
                mime="text/csv",
                key="ml_download_y_test",
            )

        fitted = st.session_state.get("ml_preprocessing_result")
        if (
            fitted is not None
            and fitted.get("dataset_id") == dataset_id
            and fitted.get("revision") == st.session_state["ml_prep_revision"]
        ):
            st.markdown("#### Leakage-safe processed features")
            st.caption(
                "Imputation, encoding, scaling, and capping were fitted on "
                "training features only."
            )
            fitted_train_col, fitted_test_col = st.columns(2)
            with fitted_train_col:
                st.markdown("##### Processed X_train")
                st.dataframe(fitted["X_train"].head(10), width="stretch")
                st.download_button(
                    "Download processed X_train",
                    data=fitted["X_train"].to_csv(index=False).encode("utf-8"),
                    file_name="X_train_processed.csv",
                    mime="text/csv",
                    key="ml_download_processed_X_train",
                )
            with fitted_test_col:
                st.markdown("##### Processed X_test")
                st.dataframe(fitted["X_test"].head(10), width="stretch")
                st.download_button(
                    "Download processed X_test",
                    data=fitted["X_test"].to_csv(index=False).encode("utf-8"),
                    file_name="X_test_processed.csv",
                    mime="text/csv",
                    key="ml_download_processed_X_test",
                )
    elif unsupervised_mode:
        st.markdown("#### Clustering preparation")
        st.caption(
            "This workflow has no target or train/test split. Remove identifier "
            "columns and any fields that should not affect clustering."
        )
        fitted = st.session_state.get("ml_preprocessing_result")
        if (
            fitted is not None
            and fitted.get("mode") == "unsupervised"
            and fitted.get("dataset_id") == dataset_id
            and fitted.get("revision") == st.session_state["ml_prep_revision"]
        ):
            st.markdown(
                f"Prepared features: **{len(fitted['X_train']):,} rows × "
                f"{fitted['X_train'].shape[1]:,} columns**"
            )
            st.dataframe(fitted["X_train"].head(20), width="stretch")
            st.download_button(
                "Download prepared clustering features",
                data=fitted["X_train"].to_csv(index=False).encode("utf-8"),
                file_name="clustering_features.csv",
                mime="text/csv",
                key="ml_download_clustering_features",
            )
        else:
            st.info(
                "Go to Model preprocessing and choose the imputation, encoding, "
                "scaling, and outlier methods to prepare clustering features."
            )
    else:
        st.info("Create a train/test split in the previous stage to review it here.")

    st.divider()
    if unsupervised_mode:
        st.warning(
            "Applying this copy updates the cleaned dataset shown on Home and "
            "Data Explorer, but does not apply encoded or scaled clustering "
            "features. Download those from this review stage. The uploaded-file "
            "ID remains unchanged, and Reset Preparation restores the working copy."
        )
    else:
        st.warning(
            "Applying this copy updates the dataset shown on Home and Data Explorer. "
            "It does not replace the uploaded-file ID, so the uploader will not reload "
            "the original file. Reset Preparation restores the working copy; apply "
            "again if you also want to restore the active workspace dataset."
        )
    if st.button("Apply cleaned data to app", type="primary", key="ml_prep_apply_app"):
        st.session_state["working_df"] = df.copy(deep=True)
        st.session_state["original_df"] = df.copy(deep=True)
        st.session_state["dataset_revision"] = (
            st.session_state.get("dataset_revision", 0) + 1
        )
        st.session_state.pop("ml_prep_history_key", None)
        st.session_state.pop("ml_preparation_result", None)
        st.session_state.pop("ml_preprocessing_result", None)
        st.success("Prepared dataset is now active in the app.")
        st.rerun()
