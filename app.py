
import tempfile
from pathlib import Path
import html

import streamlit as st

from modules.file_loader import load_file


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataLens | Workspace",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "working_df" not in st.session_state:
    st.session_state["working_df"] = None

if "original_df" not in st.session_state:
    st.session_state["original_df"] = None

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None

if "processed_upload_id" not in st.session_state:
    st.session_state["processed_upload_id"] = None


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
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

    div[data-testid="stFileUploader"] {
        background: #111C2D;
        border: 1px dashed #3D5677;
        border-radius: 14px;
        padding: 14px;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
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

    [data-testid="stSidebar"] hr {
        border-color: #29374D;
        margin-top: 14px;
        margin-bottom: 14px;
    }

    hr {
        border-color: #29374D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE UPLOADER
# ============================================================
st.markdown(
    '<p> </p>',
    unsafe_allow_html=True,
)
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


# ============================================================
# PROCESS ONLY A NEW UPLOAD
# ============================================================

if uploaded_file is not None:
    upload_bytes = uploaded_file.getvalue()
    upload_id = (
        uploaded_file.name,
        len(upload_bytes),
        hash(upload_bytes),
    )

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

            # Refresh the app so the sidebar reflects the new state.
            st.rerun()

        except Exception as e:
            st.error(f"Unable to process this file: {e}")

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size: 1.35rem; font-weight: 750;
                    color: #F1F5F9; padding: 8px 0;">
            🔎 DataLens
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("DATA INTELLIGENCE WORKSPACE")
    st.divider()

    st.markdown(
        '<p class="sidebar-section">WORKSPACE</p>',
        unsafe_allow_html=True,
    )

    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link(
        "pages/1_Data_Explorer.py",
        label="Data Explorer",
        icon="📊",
    )

    st.divider()

    st.markdown(
        '<p class="sidebar-section">DATASET</p>',
        unsafe_allow_html=True,
    )

    sidebar_df = st.session_state.get("working_df")
    sidebar_name = st.session_state.get("dataset_name")

    if sidebar_df is not None and sidebar_name:
        safe_name = html.escape(str(sidebar_name))

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
                    Upload a CSV or Excel file using the uploader
                    on the Home page.
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


# ============================================================
# DASHBOARD HEADER
# ============================================================

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


# ============================================================
# TXT FILE SUMMARY
# ============================================================

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


# ============================================================
# DATASET DASHBOARD
# ============================================================

df = st.session_state.get("working_df")

if df is not None:
    st.divider()

    st.markdown(
        '<p class="section-label">02 — DATASET OVERVIEW</p>',
        unsafe_allow_html=True,
    )

    st.subheader(
        st.session_state.get("dataset_name", "Current dataset")
    )

    row_count, column_count = df.shape
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    memory_kb = df.memory_usage(deep=True).sum() / 1024

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{row_count:,}")
    c2.metric("Columns", f"{column_count:,}")
    c3.metric("Missing Cells", f"{missing_count:,}")
    c4.metric("Memory", f"{memory_kb:,.1f} KB")

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

        total_cells = row_count * column_count

        completeness = (
            100 * (1 - missing_count / total_cells)
            if total_cells > 0
            else 100.0
        )

        duplicate_percent = (
            duplicate_count / row_count * 100
            if row_count > 0
            else 0.0
        )

        st.metric("Data completeness", f"{completeness:.1f}%")
        st.progress(completeness / 100)

        st.caption(
            f"{missing_count:,} missing cells out of "
            f"{total_cells:,} total cells."
        )

        st.metric("Duplicate row rate", f"{duplicate_percent:.1f}%")

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
