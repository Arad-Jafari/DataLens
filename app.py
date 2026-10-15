import tempfile
from pathlib import Path

import streamlit as st

from modules.data_utils import get_dataset_summary, make_upload_id
from modules.file_loader import load_file
from modules.ui import (
    ensure_app_session,
    inject_base_css,
    inject_home_css,
    render_sidebar,
)


st.set_page_config(
    page_title="DataLens | Workspace",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_app_session()
inject_base_css()
inject_home_css()


st.markdown(
    '<p class="section-label">01 — IMPORT DATA</p>',
    unsafe_allow_html=True,
)

st.subheader("Start with a file")
st.write("Choose a CSV, Excel workbook, or text file.")

uploaded_file = st.file_uploader(
    "Drop your file here or browse",
    type=["csv", "xlsx", "txt"],
    help=(
        "CSV and XLSX support data exploration. "
        "TXT files show basic text statistics only."
    ),
    key="home_file_uploader",
)

if uploaded_file is not None:
    upload_bytes = uploaded_file.getvalue()
    upload_id = make_upload_id(uploaded_file.name, upload_bytes)

    if upload_id != st.session_state["processed_upload_id"]:
        file_name = uploaded_file.name
        suffix = Path(file_name).suffix.lower()
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temp_file.write(upload_bytes)
                temp_path = Path(temp_file.name)

            try:
                result = load_file(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
                temp_path = None

            if suffix == ".txt":
                st.session_state["working_df"] = None
                st.session_state["original_df"] = None
                st.session_state["dataset_name"] = None
                st.session_state["text_file_stats"] = result
            else:
                st.session_state["working_df"] = result.copy()
                st.session_state["original_df"] = result.copy()
                st.session_state["dataset_name"] = file_name
                st.session_state.pop("text_file_stats", None)

            st.session_state["processed_upload_id"] = upload_id
            st.session_state.pop("ml_preparation_result", None)
            st.rerun()

        except (OSError, ValueError, UnicodeDecodeError) as exc:
            st.error(f"Unable to process this file: {exc}")

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


render_sidebar()


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">DATA WORKSPACE / OVERVIEW</div>
        <h1>Make sense of your data.</h1>
        <p>
            Upload your dataset, understand its quality, discover patterns,
            and prepare cleaner data for machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


text_stats = st.session_state.get("text_file_stats")

if text_stats is not None:
    st.markdown(
        '<p class="section-label">TEXT FILE SUMMARY</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Characters", f"{text_stats['character_count']:,}")
    c2.metric("Words", f"{text_stats['word_count']:,}")
    c3.metric("Lines", f"{text_stats['line_count']:,}")
    c4.metric(
        "File Size",
        f"{text_stats['file_size_bytes'] / 1024:.2f} KB",
    )

    st.info(
        "TXT files provide basic file information only. "
        "Dataset analysis is available for CSV and Excel."
    )


df = st.session_state.get("original_df")
if df is None:
    df = st.session_state.get("working_df")

if df is not None:
    st.divider()
    st.markdown(
        '<p class="section-label">02 — DATASET OVERVIEW</p>',
        unsafe_allow_html=True,
    )
    st.subheader(st.session_state.get("dataset_name") or "Current dataset")

    summary = get_dataset_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{summary['row_count']:,}")
    c2.metric("Columns", f"{summary['column_count']:,}")
    c3.metric("Missing Cells", f"{summary['missing_count']:,}")
    c4.metric("Memory", f"{summary['memory_kb']:,.1f} KB")

    preview_col, quality_col = st.columns([1.65, 1])

    with preview_col:
        st.markdown("#### Data preview")
        st.caption("First 10 rows of the current dataset")
        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
        )

    with quality_col:
        st.markdown("#### Data quality snapshot")
        st.metric("Data completeness", f"{summary['completeness']:.1f}%")
        st.progress(summary["completeness"] / 100)
        st.caption(
            f"{summary['missing_count']:,} missing cells out of "
            f"{summary['total_cells']:,} total cells."
        )
        st.metric("Duplicate row rate", f"{summary['duplicate_percent']:.1f}%")

    st.info(
        "Continue to Data Explorer for detailed missing-value "
        "reports, column information, descriptive statistics, "
        "visualizations, and frequency distributions."
    )

else:
    st.divider()
    st.markdown(
        '<p class="section-label">03 — YOUR TOOLKIT</p>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📊 Data Explorer</h3>
                <p>
                    Inspect columns, missing values, duplicate rows,
                    and descriptive statistics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🧹 Data Preparation</h3>
                <p>
                    Build a workflow for cleaning and transforming
                    data before modeling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🤖 ML Preparation</h3>
                <p>
                    Prepare structured datasets for future
                    machine learning experiments.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
