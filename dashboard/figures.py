"""Plotly figure builders shared between the static layout and the
callbacks that will make them interactive.

Color is assigned by job, not by eye: Legit/Fraud is a good/bad status
(reserved green/red), the two-run comparison is nominal identity
(categorical blue/orange), and single-series magnitude charts
(feature importance, fraud-by-hour, confusion matrix) use one
sequential blue hue. Palette values are the validated defaults from
the project's dataviz skill.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import confusion_matrix

METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]

PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "red": "#e34948",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
}

# Fraud/Legit reads as good/bad, so this would normally reach for the
# status green/red pair — but that pair fails CVD separation hard
# (deuteranopia ΔE 4.1, checked with the dataviz skill's validator).
# blue/red is the palette's documented diverging pair and passes clean
# (ΔE 21.6), so that's the mapping used everywhere in this app.
CLASS_COLOR_MAP = {"Fraud": PALETTE["red"], "Legit": PALETTE["blue"]}

_axis_style = dict(gridcolor=PALETTE["gridline"], zerolinecolor=PALETTE["baseline"], linecolor=PALETTE["muted"])

pio.templates["fraud_theme"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=PALETTE["ink"], size=13),
        title=dict(font=dict(size=16)),
        colorway=[PALETTE["blue"], PALETTE["orange"]],
        xaxis=_axis_style,
        yaxis=_axis_style,
        margin=dict(t=50, l=60, r=30, b=40),
    )
)
pio.templates.default = "fraud_theme"


def comparison_bar_figure(results_df, run_a, run_b):
    """Diverging bar chart comparing two runs across metrics — the
    same idea as the reference app's two-company bigram comparison.
    Nominal identity (arbitrary runs, not good/bad), so categorical
    blue/orange rather than the status colors."""
    row_a = results_df[results_df["run"] == run_a].iloc[0]
    row_b = results_df[results_df["run"] == run_b].iloc[0]

    fig = go.Figure()
    fig.add_bar(name=run_a, x=METRICS, y=[row_a[m] for m in METRICS], marker_color=PALETTE["blue"])
    fig.add_bar(name=run_b, x=METRICS, y=[-row_b[m] for m in METRICS], marker_color=PALETTE["orange"])
    fig.update_layout(
        barmode="relative",
        title=f"{run_a}  vs  {run_b}",
        yaxis_title="metric value",
        yaxis_tickformat=".2f",
    )
    return fig


def fraud_by_hour_figure(df):
    counts = df[df["Class"] == 1].groupby("hour_of_day").size().reindex(range(24), fill_value=0)
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Hour of day", "y": "Fraud count"},
        title="Fraud count by hour of day",
    )
    fig.update_traces(marker_color=PALETTE["blue"])
    return fig


def embedding_scatter_figure(cache, method):
    df = pd.DataFrame(
        {
            "x": cache["coords"][:, 0],
            "y": cache["coords"][:, 1],
            "class": np.where(cache["y"] == 1, "Fraud", "Legit"),
        }
    )
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="class",
        color_discrete_map=CLASS_COLOR_MAP,
        opacity=0.6,
        title=f"{method.upper()} projection of transactions",
    )
    return fig


def feature_importance_figure(shap_cache, chart_type="bar"):
    importance = np.abs(shap_cache["shap_values"]).mean(axis=0)
    imp_df = pd.DataFrame(
        {"feature": shap_cache["feature_names"], "importance": importance}
    ).sort_values("importance", ascending=True)

    if chart_type == "treemap":
        fig = px.treemap(imp_df, path=["feature"], values="importance", title="Feature importance")
        fig.update_traces(marker_colorscale=[[0, PALETTE["surface"]], [1, PALETTE["blue"]]])
        return fig

    fig = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature importance (mean |SHAP value|)",
    )
    fig.update_traces(marker_color=PALETTE["blue"])
    return fig


def shap_waterfall_figure(shap_cache, row_idx=0, top_n=15):
    """Red = pushes the prediction toward fraud, blue = pushes toward
    legit — the same Fraud/Legit color mapping used everywhere else."""
    values = shap_cache["shap_values"][row_idx]
    feature_names = shap_cache["feature_names"]
    order = np.argsort(np.abs(values))[::-1][:top_n]

    fig = go.Figure(
        go.Bar(
            x=values[order],
            y=[feature_names[i] for i in order],
            orientation="h",
            marker_color=[PALETTE["red"] if v > 0 else PALETTE["blue"] for v in values[order]],
        )
    )
    fig.update_layout(
        title=f"SHAP contributions toward fraud (base value = {shap_cache['base_value']:.3f})",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def confusion_matrix_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Pred: Legit", "Pred: Fraud"],
        y=["True: Legit", "True: Fraud"],
        color_continuous_scale=[[0, PALETTE["surface"]], [1, PALETTE["blue"]]],
        title="Confusion matrix",
    )
    return fig
