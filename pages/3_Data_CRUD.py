from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from modules.crud import create_record, delete_record, read_record, update_record
from modules.data_utils import get_dataset_summary
from modules.ui import (
    current_dataset,
    ensure_app_session,
    inject_base_css,
    render_sidebar,
)

st.set_page_config(
    page_title="DataLens | Data Management",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_app_session()
inject_base_css()
render_sidebar()

st.title("Dataset Management")
st.caption(
    "Create, inspect, update, or delete records in a working copy. "
    "Apply your changes to the workspace when you are ready."
)

source_df = current_dataset()
dataset_name = st.session_state.get("dataset_name") or "Current dataset"
dataset_id = st.session_state.get("processed_upload_id")

if not isinstance(source_df, pd.DataFrame):
    st.info("Upload a CSV or Excel dataset on the Home page to manage records.")
    st.stop()
if source_df.columns.empty:
    st.warning("The active dataset has no columns, so record operations are unavailable.")
    st.stop()

crud_key = (
    dataset_id if dataset_id is not None else dataset_name,
    st.session_state.get("dataset_revision", 0),
)
if st.session_state.get("crud_dataset_key") != crud_key:
    st.session_state["crud_dataset_key"] = crud_key
    st.session_state["crud_original_df"] = source_df.copy(deep=True)
    st.session_state["crud_working_df"] = source_df.copy(deep=True)
    st.session_state["crud_undo_stack"] = []

working_df: pd.DataFrame = st.session_state["crud_working_df"]


def record_change(updated: pd.DataFrame) -> None:
    st.session_state["crud_undo_stack"].append(working_df.copy(deep=True))
    st.session_state["crud_working_df"] = updated.copy(deep=True)
    st.rerun()


def field_input(column: Any, value: Any, key: str):
    dtype = working_df[column].dtype
    label = str(column)
    if pd.api.types.is_bool_dtype(dtype):
        choices = ["(Missing)", "True", "False"]
        default = "(Missing)" if pd.isna(value) else str(bool(value))
        return st.selectbox(
            label,
            choices,
            index=choices.index(default),
            key=key,
        )
    if pd.api.types.is_datetime64_any_dtype(dtype):
        default = date.today() if pd.isna(value) else pd.Timestamp(value).date()
        return st.date_input(label, value=default, key=key)
    if pd.api.types.is_integer_dtype(dtype):
        default = None if pd.isna(value) else int(value)
        return st.number_input(
            label,
            value=default,
            step=1,
            key=key,
        )
    if pd.api.types.is_numeric_dtype(dtype):
        default = None if pd.isna(value) else float(value)
        return st.number_input(
            label,
            value=default,
            format="%.8g",
            key=key,
        )
    default_text = "" if pd.isna(value) else str(value)
    return st.text_input(label, value=default_text, key=key)


def parse_field_value(column: Any, value: Any) -> Any:
    dtype = working_df[column].dtype
    if pd.api.types.is_bool_dtype(dtype):
        return pd.NA if value == "(Missing)" else value == "True"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.Timestamp(value)
    if isinstance(value, str) and not value.strip():
        return pd.NA
    if pd.api.types.is_integer_dtype(dtype):
        return int(value)
    if pd.api.types.is_numeric_dtype(dtype):
        return float(value)
    return value


def is_id_column(column: Any) -> bool:
    normalized_name = (
        str(column).strip().casefold().replace("-", "_").replace(" ", "_")
    )
    return normalized_name == "id" or normalized_name.endswith("_id")


id_column = next((column for column in working_df.columns if is_id_column(column)), None)


def find_record_position(action: str) -> int | None:
    if id_column is not None:
        query = st.text_input(
            "Search by ID",
            placeholder=f"Enter part or all of {id_column}",
            key=f"crud_{action}_id_search",
        ).strip().casefold()
        if query:
            positions = [
                position
                for position, value in enumerate(working_df[id_column].tolist())
                if (
                    not (
                        pd.api.types.is_scalar(value)
                        and pd.isna(value)
                    )
                    and query in str(value).casefold()
                )
            ]
        else:
            positions = list(range(len(working_df)))
    else:
        query = st.text_input(
            "Search by record number",
            placeholder="Enter part or all of the 1-based record number",
            key=f"crud_{action}_id_search",
        ).strip()
        positions = [
            position
            for position in range(len(working_df))
            if query in str(position + 1)
        ]
        st.caption("This dataset has no ID column; searching by record number.")

    if not positions:
        st.info("No records match that search.")
        return None

    def format_position(position: int) -> str:
        preview = ", ".join(
            f"{column}={working_df.iloc[position][column]!r}"
            for column in working_df.columns[:3]
            if column != id_column
        )
        if id_column is None:
            identity = f"Record {position + 1}"
        else:
            identity = f"ID={working_df.iloc[position][id_column]!r}"
        return f"{identity} · Record {position + 1}" + (
            f" · {preview}" if preview else ""
        )

    return st.selectbox(
        "Matching records",
        options=positions,
        format_func=format_position,
        key=(
            f"crud_{action}_match_{query}_"
            f"{len(st.session_state['crud_undo_stack'])}"
        ),
    )


summary = get_dataset_summary(working_df)
metrics = st.columns(4)
metrics[0].metric("Rows", f"{summary['row_count']:,}")
metrics[1].metric("Columns", f"{summary['column_count']:,}")
metrics[2].metric("Missing cells", f"{summary['missing_count']:,}")
metrics[3].metric(
    "Unsaved CRUD edits",
    f"{len(st.session_state['crud_undo_stack']):,}",
)

undo_col, reset_col, status_col = st.columns([1, 1, 3])
if undo_col.button(
    "Undo last change",
    disabled=not st.session_state["crud_undo_stack"],
    key="crud_undo",
):
    st.session_state["crud_working_df"] = st.session_state[
        "crud_undo_stack"
    ].pop()
    st.rerun()
if reset_col.button(
    "Reset CRUD changes",
    disabled=not st.session_state["crud_undo_stack"],
    key="crud_reset",
):
    st.session_state["crud_working_df"] = st.session_state[
        "crud_original_df"
    ].copy(deep=True)
    st.session_state["crud_undo_stack"] = []
    st.rerun()
status_col.caption(f"Working copy of **{dataset_name}**")

read_tab, create_tab, update_tab, delete_tab = st.tabs(
    ["Read records", "Create record", "Update record", "Delete record"]
)

with read_tab:
    st.subheader("Read and browse records")
    page_size = st.selectbox(
        "Rows per page",
        [10, 25, 50, 100],
        key="crud_page_size",
    )
    page_count = max(1, (len(working_df) + page_size - 1) // page_size)
    page_number = st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key="crud_page_number",
    )
    start = (int(page_number) - 1) * page_size
    end = min(start + page_size, len(working_df))
    page_df = working_df.iloc[start:end].copy()
    page_df.insert(0, "Record position", range(start, end))
    st.dataframe(page_df, width="stretch", hide_index=True)
    st.caption(
        f"Showing records {start + 1:,}–{end:,} of {len(working_df):,}. "
        "Record positions are used for edits, even when index labels repeat."
    )

with create_tab:
    st.subheader("Create record")
    st.caption("Leave a field blank to create a missing value for that column.")
    with st.form("crud_create_record_form"):
        new_values = {
            column: field_input(column, pd.NA, f"crud_create_{position}")
            for position, column in enumerate(working_df.columns)
        }
        create_submitted = st.form_submit_button(
            "Create record",
            type="primary",
        )
    if create_submitted:
        try:
            parsed_values = {
                column: parse_field_value(column, value)
                for column, value in new_values.items()
            }
            record_change(create_record(working_df, parsed_values))
        except (TypeError, ValueError, OverflowError) as exc:
            st.error(f"Unable to create record: {exc}")

with update_tab:
    st.subheader("Update record")
    if working_df.empty:
        st.info("There are no records to update.")
    else:
        selected_position = find_record_position("update")
        if selected_position is not None:
            selected_record = read_record(working_df, selected_position)
            with st.form("crud_update_record_form"):
                update_values = {
                    column: field_input(
                        column,
                        selected_record[column],
                        (
                            f"crud_update_{selected_position}_"
                            f"{position}_{len(st.session_state['crud_undo_stack'])}"
                        ),
                    )
                    for position, column in enumerate(working_df.columns)
                }
                update_submitted = st.form_submit_button(
                    "Save record updates",
                    type="primary",
                )
            if update_submitted:
                try:
                    parsed_updates = {
                        column: parse_field_value(column, value)
                        for column, value in update_values.items()
                    }
                    record_change(
                        update_record(
                            working_df,
                            selected_position,
                            parsed_updates,
                        )
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    st.error(f"Unable to update record: {exc}")

with delete_tab:
    st.subheader("Delete record")
    if working_df.empty:
        st.info("There are no records to delete.")
    else:
        delete_position = find_record_position("delete")
        if delete_position is not None:
            st.dataframe(
                read_record(working_df, delete_position).to_frame().T,
                width="stretch",
                hide_index=True,
            )
            confirm_delete = st.checkbox(
                "I understand this removes the selected record from the working copy.",
                key=(
                    f"crud_confirm_delete_{delete_position}_"
                    f"{len(st.session_state['crud_undo_stack'])}"
                ),
            )
            if st.button(
                "Delete selected record",
                type="primary",
                disabled=not confirm_delete,
                key="crud_delete_button",
            ):
                record_change(delete_record(working_df, delete_position))

st.divider()
st.warning(
    "CRUD changes are not active in Home or Data Explorer until you apply them. "
    "The uploaded-file identifier remains unchanged, so the upload will not "
    "overwrite your applied dataset."
)
if st.button(
    "Apply CRUD changes to app",
    type="primary",
    disabled=not st.session_state["crud_undo_stack"],
    key="crud_apply",
):
    applied_df = st.session_state["crud_working_df"].copy(deep=True)
    st.session_state["working_df"] = applied_df.copy(deep=True)
    st.session_state["original_df"] = applied_df.copy(deep=True)
    st.session_state["dataset_revision"] = (
        st.session_state.get("dataset_revision", 0) + 1
    )
    st.session_state["crud_original_df"] = applied_df.copy(deep=True)
    st.session_state["crud_undo_stack"] = []
    st.session_state["crud_dataset_key"] = (
        dataset_id,
        st.session_state["dataset_revision"],
    )
    st.success("CRUD changes are now active in the workspace.")
    st.rerun()
