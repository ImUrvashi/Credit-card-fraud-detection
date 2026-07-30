"""Plotly figure builders shared between the static layout and the
callbacks that will make them interactive.

Color is assigned by job, not by eye: Legit/Fraud reads as good/bad,
but the status green/red pair fails CVD separation hard (see
CLASS_COLOR_MAP below), so it uses the palette's blue/red diverging
pair instead. The two-run comparison is nominal identity (categorical
blue/orange), and single-series magnitude charts (feature importance,
fraud-by-hour, confusion matrix) use one sequential blue hue. Palette
values are the validated defaults from the project's dataviz skill.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

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
    """A single real trace (not px.scatter's per-category split) so a
    click's pointIndex maps directly to a position in cache['coords']/
    cache['row_index'] — no reverse-engineering which category-filtered
    subset a click landed in. Two invisible dummy traces supply the
    Fraud/Legit legend swatches."""
    point_colors = np.where(cache["y"] == 1, CLASS_COLOR_MAP["Fraud"], CLASS_COLOR_MAP["Legit"])
    labels = np.where(cache["y"] == 1, "Fraud", "Legit")

    fig = go.Figure(
        go.Scatter(
            x=cache["coords"][:, 0],
            y=cache["coords"][:, 1],
            mode="markers",
            marker=dict(color=point_colors, opacity=0.6, size=7),
            text=labels,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    for class_name in ["Legit", "Fraud"]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=CLASS_COLOR_MAP[class_name], size=9),
                name=class_name,
            )
        )
    fig.update_layout(title=f"{method.upper()} projection of transactions", xaxis_title="x", yaxis_title="y")
    return fig


def feature_importance_figure(shap_cache, chart_type="bar"):
    importance = np.abs(shap_cache["shap_values"]).mean(axis=0)
    imp_df = pd.DataFrame(
        {"feature": shap_cache["feature_names"], "importance": importance}
    ).sort_values("importance", ascending=True)

    if chart_type == "treemap":
        # color must be passed here, not via update_traces after the
        # fact — without it px.treemap has no numeric values bound to
        # a colorscale, so box color would default to an arbitrary
        # qualitative palette instead of encoding importance.
        fig = px.treemap(
            imp_df,
            path=["feature"],
            values="importance",
            color="importance",
            color_continuous_scale=[[0, PALETTE["surface"]], [1, PALETTE["blue"]]],
            title="Feature importance",
        )
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


def threshold_curve_figure(y_true, y_scores, threshold, curve_type="pr"):
    """The ROC or PR curve across every possible threshold, with a marker
    showing where the current slider position actually sits on it —
    the scalar metrics only show one point; this shows the whole
    tradeoff. PR is the default since it's this project's primary
    metric (see UNDERSTANDING.md §6/§9).

    The marker's position is computed directly from `y_scores >=
    threshold` — the same comparison the confusion matrix panel next
    to it uses — rather than searching the curve's own threshold grid
    for the "nearest" value. Model scores are often sparse/discretized
    (e.g. tree-based leaf probabilities), so the nearest *actually
    observed* threshold to an arbitrary slider value like 0.51 can sit
    far away (e.g. 0.31), which looked like the marker jumping
    non-monotonically as the slider moved. Recomputing directly from
    the query threshold keeps the marker exactly consistent with the
    adjacent confusion matrix and metrics text."""
    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    recall_point = tp / (tp + fn) if (tp + fn) else 0.0
    precision_point = tp / (tp + fp) if (tp + fp) else 1.0
    fpr_point = fp / (fp + tn) if (fp + tn) else 0.0

    if curve_type == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores)

        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=PALETTE["blue"]), name="ROC curve")
        fig.add_scatter(
            x=[0, 1], y=[0, 1], mode="lines", line=dict(color=PALETTE["muted"], dash="dot"), name="No-skill"
        )
        fig.add_scatter(
            x=[fpr_point],
            y=[recall_point],
            mode="markers",
            marker=dict(color=PALETTE["red"], size=12, line=dict(color=PALETTE["surface"], width=2)),
            name=f"threshold = {threshold:.2f}",
        )
        fig.update_layout(
            title=f"ROC curve (AUC = {auc_val:.3f})",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
        )
        return fig

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_val = average_precision_score(y_true, y_scores)
    no_skill = float(np.mean(y_true))

    fig = go.Figure()
    fig.add_scatter(x=recall, y=precision, mode="lines", line=dict(color=PALETTE["blue"]), name="PR curve")
    fig.add_hline(y=no_skill, line_dash="dot", line_color=PALETTE["muted"], annotation_text="no-skill baseline")
    fig.add_scatter(
        x=[recall_point],
        y=[precision_point],
        mode="markers",
        marker=dict(color=PALETTE["red"], size=12, line=dict(color=PALETTE["surface"], width=2)),
        name=f"threshold = {threshold:.2f}",
    )
    fig.update_layout(
        title=f"Precision-Recall curve (PR-AUC = {auc_val:.3f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
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
