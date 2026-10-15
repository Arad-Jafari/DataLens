from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.data_utils import get_column_series, numeric_columns

PLOTLY_TEMPLATE = "plotly_dark"


def build_histogram(
    df: pd.DataFrame,
    column: Any,
    title: str,
    *,
    nbins: int | None = None,
):
    """Histogram of a numeric column. Missing and non-numeric values are ignored."""
    values = pd.to_numeric(get_column_series(df, column), errors="coerce")
    if values.notna().sum() == 0:
        return None, "The selected column has no numeric values to chart."

    plot_df = pd.DataFrame({str(column): values})
    kwargs: dict[str, Any] = {
        "x": str(column),
        "template": PLOTLY_TEMPLATE,
        "title": title,
    }
    if nbins is not None:
        kwargs["nbins"] = nbins
    return px.histogram(plot_df, **kwargs), None


def build_category_bar(
    df: pd.DataFrame,
    column: Any,
    title: str,
    *,
    limit: int = 20,
    orientation: str = "v",
    category_axis: str = "Category",
    count_axis: str = "Count",
):
    """Frequency bar chart for a categorical column."""
    counts = (
        get_column_series(df, column)
        .astype("string")
        .fillna("(Missing)")
        .value_counts()
        .head(limit)
        .rename_axis(category_axis)
        .reset_index(name=count_axis)
    )
    if counts.empty:
        return None, "No categorical values are available to chart."

    if orientation == "h":
        fig = px.bar(
            counts,
            x=count_axis,
            y=category_axis,
            orientation="h",
            template=PLOTLY_TEMPLATE,
            title=title,
        )
    else:
        fig = px.bar(
            counts,
            x=category_axis,
            y=count_axis,
            template=PLOTLY_TEMPLATE,
            title=title,
        )
    return fig, None


def build_box_plot(df: pd.DataFrame, column: Any, title: str):
    values = pd.to_numeric(get_column_series(df, column), errors="coerce")
    if values.notna().sum() == 0:
        return None, "The selected column has no numeric values to chart."

    plot_df = pd.DataFrame({str(column): values})
    fig = px.box(
        plot_df,
        y=str(column),
        template=PLOTLY_TEMPLATE,
        title=title,
    )
    return fig, None


def build_scatter_plot(
    df: pd.DataFrame,
    x: Any,
    y: Any,
    title: str,
    *,
    color: Any | None = None,
):
    plot_df = df.copy()
    plot_df[x] = pd.to_numeric(get_column_series(plot_df, x), errors="coerce")
    plot_df[y] = pd.to_numeric(get_column_series(plot_df, y), errors="coerce")
    plot_df = plot_df.dropna(subset=[x, y])
    if plot_df.empty:
        return None, "No complete X/Y pairs are available for this scatter plot."

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color,
        template=PLOTLY_TEMPLATE,
        title=title,
    )
    return fig, None


def _correlation_matrix(df: pd.DataFrame, columns: list[Any] | None = None):
    selected = columns if columns is not None else numeric_columns(df)
    if len(selected) < 2:
        return None, "At least two numeric columns are required."

    numeric_data = pd.DataFrame(
        {col: pd.to_numeric(get_column_series(df, col), errors="coerce") for col in selected}
    )
    corr = numeric_data.corr()
    valid_columns = corr.columns[corr.notna().any()]
    corr = corr.loc[valid_columns, valid_columns]
    if corr.shape[0] < 2:
        return None, "Not enough valid numeric correlations for a heatmap."
    return corr, None


def build_frequency_bar(table: pd.DataFrame, title: str):
    """Bar chart for a frequency table produced by make_frequency_table()."""
    if table.empty:
        return None, "No values are available for frequency analysis."
    fig = px.bar(
        table.head(30),
        x="Value / Interval",
        y="Frequency",
        template=PLOTLY_TEMPLATE,
        title=title,
    )
    return fig, None


def build_correlation_heatmap(df: pd.DataFrame, title: str, columns: list[Any] | None = None):
    """Plotly Express heatmap used by Search & Filter."""
    corr, error = _correlation_matrix(df, columns)
    if error:
        return None, error

    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        template=PLOTLY_TEMPLATE,
        title=title,
    )
    return fig, None


def build_correlation_heatmap_annotated(
    df: pd.DataFrame,
    title: str,
    columns: list[Any] | None = None,
):
    """Graph Objects heatmap used by Visual Reports."""
    corr, error = _correlation_matrix(df, columns)
    if error:
        return None, error

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.astype(str),
            y=corr.index.astype(str),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            zmid=0,
            colorbar_title="Correlation",
            text=[
                [
                    f"{value:.2f}" if pd.notna(value) else "N/A"
                    for value in row
                ]
                for row in corr.values
            ],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title=title)
    return fig, None
