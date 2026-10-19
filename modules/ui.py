from __future__ import annotations

from copy import deepcopy
from html import escape
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import streamlit as st

from modules.data_utils import make_upload_id
from modules.file_loader import load_file

APP_SESSION_DEFAULTS: dict[str, Any] = {
    "working_df": None,
    "original_df": None,
    "dataset_name": None,
    "processed_upload_id": None,
    "dataset_revision": 0,
}

EXPLORER_FILTER_DEFAULTS: dict[str, Any] = {
    "dl_quick_search": "",
    "dl_filter_logic": "AND",
    "dl_filter_drafts": [],
    "dl_applied_conditions": [],
    "dl_applied_logic": "AND",
    "dl_exclude_missing": False,
}

_FILTER_WIDGET_PREFIXES = (
    "dl_col_",
    "dl_kind_",
    "dl_op_",
    "dl_val_",
    "dl_low_",
    "dl_high_",
    "dl_date_",
    "dl_date_low_",
    "dl_date_high_",
    "dl_remove_",
)

BASE_CSS = """
<style>
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #263449;
}

.eyebrow {
    color: #7F96B5;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-label {
    color: #94A9C5;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

div[data-testid="stMetric"] {
    background: #151F32;
    border: 1px solid #29374D;
    border-radius: 14px;
    padding: 18px 20px;
}

div[data-testid="stMetricLabel"] {
    color: #9BAEC7;
}

div[data-testid="stMetricValue"] {
    color: #F1F5F9;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #29374D;
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 2px 16px;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    background: #1D3554;
    border: 1px solid #345477;
    border-radius: 12px;
    font-size: 21px;
}

.sidebar-title {
    font-size: 1.35rem;
    font-weight: 750;
    color: #F1F5F9;
}

.sidebar-subtitle {
    color: #8198B7;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 1.4px;
}

.sidebar-section {
    color: #8198B7;
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 1.5px;
    margin: 12px 0 8px 2px;
}

.sidebar-dataset {
    background: #121E30;
    border: 1px solid #293B54;
    border-radius: 12px;
    padding: 14px;
    margin: 8px 0 12px 0;
    overflow-wrap: anywhere;
}

.dataset-status {
    color: #A9BCD4;
    font-size: 0.67rem;
    font-weight: 750;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #34D399;
    border-radius: 50%;
    margin-right: 7px;
}

.dataset-name {
    color: #F1F5F9;
    font-size: 0.9rem;
    font-weight: 650;
    margin-bottom: 5px;
}

.dataset-details {
    color: #91A5C0;
    font-size: 0.78rem;
    line-height: 1.6;
}

[data-testid="stSidebar"] hr,
hr {
    border-color: #29374D;
}
</style>
"""

HOME_CSS = """
<style>
.hero {
    background: linear-gradient(120deg, #17263D, #172D49);
    border: 1px solid #2B4362;
    border-radius: 18px;
    padding: 30px;
    margin: 12px 0 26px 0;
}

.hero h1 {
    font-size: 2.35rem;
    letter-spacing: -1px;
    color: #F8FAFC;
    margin: 8px 0;
}

.hero p {
    color: #B7C7DB;
    font-size: 1rem;
    max-width: 680px;
    margin-bottom: 0;
    line-height: 1.7;
}

div[data-testid="stFileUploader"] {
    background: #111C2D;
    border: 1px dashed #3D5677;
    border-radius: 14px;
    padding: 14px;
}

div[data-testid="stFileUploader"] section {
    background: transparent;
}

.feature-card {
    background: #151F32;
    border: 1px solid #29374D;
    border-radius: 14px;
    padding: 20px;
    min-height: 155px;
}

.feature-card h3 {
    color: #E5EDF7;
    font-size: 1rem;
    margin-bottom: 9px;
}

.feature-card p {
    color: #9BAEC7;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 0;
}
</style>
"""


def inject_base_css() -> None:
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def inject_home_css() -> None:
    st.markdown(HOME_CSS, unsafe_allow_html=True)


def ensure_app_session() -> None:
    for key, value in APP_SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_dataset() -> pd.DataFrame | None:
    original = st.session_state.get("original_df")
    if isinstance(original, pd.DataFrame):
        return original
    working = st.session_state.get("working_df")
    if isinstance(working, pd.DataFrame):
        return working
    return None


def render_sidebar() -> None:
    """Shared DataLens sidebar used by every page."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">🔎</div>
                <div>
                    <div class="sidebar-title">DataLens</div>
                    <div class="sidebar-subtitle">DATA INTELLIGENCE WORKSPACE</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            '<p class="sidebar-section">WORKSPACE</p>',
            unsafe_allow_html=True,
        )
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Data_Explorer.py", label="Data Explorer", icon="📊")
        st.page_link(
            "pages/2_Data_Preparation.py",
            label="ML Preparation",
            icon="🤖",
        )
        st.page_link(
            "pages/3_Data_CRUD.py",
            label="Dataset Management",
            icon="🗂️",
        )

        st.divider()
        st.markdown(
            '<p class="sidebar-section">DATASET</p>',
            unsafe_allow_html=True,
        )

        sidebar_upload = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx"],
            help="Upload a dataset directly from the sidebar on any page.",
            key="sidebar_dataset_uploader",
        )
        if sidebar_upload is not None:
            upload_bytes = sidebar_upload.getvalue()
            upload_id = make_upload_id(sidebar_upload.name, upload_bytes)
            if upload_id != st.session_state.get("processed_upload_id"):
                suffix = Path(sidebar_upload.name).suffix.lower()
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False,
                    ) as temp_file:
                        temp_file.write(upload_bytes)
                        temp_path = Path(temp_file.name)

                    try:
                        uploaded_df = load_file(temp_path)
                    finally:
                        temp_path.unlink(missing_ok=True)
                        temp_path = None

                    st.session_state["working_df"] = uploaded_df.copy(deep=True)
                    st.session_state["original_df"] = uploaded_df.copy(deep=True)
                    st.session_state["dataset_name"] = sidebar_upload.name
                    st.session_state["processed_upload_id"] = upload_id
                    st.session_state["dataset_revision"] = 0
                    st.session_state.pop("text_file_stats", None)
                    for key in (
                        "ml_prep_history_key",
                        "ml_preparation_result",
                        "ml_preprocessing_result",
                        "ml_split_success",
                        "ml_split_error",
                        "crud_dataset_key",
                    ):
                        st.session_state.pop(key, None)
                    st.rerun()
                except (OSError, ValueError, UnicodeDecodeError) as exc:
                    st.error(f"Unable to process this file: {exc}")
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)

        sidebar_df = current_dataset()
        sidebar_name = st.session_state.get("dataset_name")

        if isinstance(sidebar_df, pd.DataFrame):
            safe_name = escape(str(sidebar_name or "Current dataset"))
            st.markdown(
                f"""
                <div class="sidebar-dataset">
                    <div class="dataset-status">
                        <span class="status-dot"></span>
                        DATA LOADED
                    </div>
                    <div class="dataset-name">{safe_name}</div>
                    <div class="dataset-details">
                        {sidebar_df.shape[0]:,} rows ·
                        {sidebar_df.shape[1]:,} columns
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-dataset">
                    <div class="dataset-status">NO DATASET</div>
                    <div class="dataset-details">
                        Use the uploader above to add a CSV or Excel dataset.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p class="sidebar-section">SUPPORTED FILES</p>',
            unsafe_allow_html=True,
        )
        st.markdown("`CSV`  `XLSX`  `TXT`")
        st.divider()
        st.caption("DataLens · Data made clearer.")


def clear_search_and_filters() -> None:
    """Clear search and filters without changing displayed columns."""
    for key, value in EXPLORER_FILTER_DEFAULTS.items():
        st.session_state[key] = deepcopy(value)


def reset_search_and_filters() -> None:
    """Reset search, filters, and restore every display column."""
    clear_search_and_filters()
    source = current_dataset()
    if isinstance(source, pd.DataFrame):
        st.session_state["dl_visible_columns"] = list(source.columns)


def _clear_filter_widget_keys() -> None:
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(_FILTER_WIDGET_PREFIXES):
            del st.session_state[key]


def initialize_explorer_state(df: pd.DataFrame, dataset_id: Any) -> None:
    """Set Explorer defaults and reset search/filter state for a new dataset."""
    previous = st.session_state.get("dl_dataset_id")
    if previous != dataset_id:
        for key, value in EXPLORER_FILTER_DEFAULTS.items():
            st.session_state[key] = deepcopy(value)
        st.session_state["dl_visible_columns"] = list(df.columns)
        st.session_state["dl_dataset_id"] = dataset_id
        _clear_filter_widget_keys()
        return

    for key, value in EXPLORER_FILTER_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)
    if "dl_visible_columns" not in st.session_state:
        st.session_state["dl_visible_columns"] = list(df.columns)
