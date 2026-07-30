"""Plotly figure builders shared between the static layout and the
callbacks that will make them interactive."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def comparison_bar_figure(results_df, run_a, run_b):
    """Diverging bar chart comparing two runs across metrics — the
    same idea as the reference app's two-company bigram comparison."""
    row_a = results_df[results_df["run"] == run_a].iloc[0]
    row_b = results_df[results_df["run"] == run_b].iloc[0]

    fig = go.Figure()
    fig.add_bar(name=run_a, x=METRICS, y=[row_a[m] for m in METRICS])
    fig.add_bar(name=run_b, x=METRICS, y=[-row_b[m] for m in METRICS])
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
        color_discrete_map={"Fraud": "crimson", "Legit": "steelblue"},
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
        return px.treemap(imp_df, path=["feature"], values="importance", title="Feature importance")

    fig = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature importance (mean |SHAP value|)",
    )
    return fig


def shap_waterfall_figure(shap_cache, row_idx=0, top_n=15):
    values = shap_cache["shap_values"][row_idx]
    feature_names = shap_cache["feature_names"]
    order = np.argsort(np.abs(values))[::-1][:top_n]

    fig = go.Figure(
        go.Bar(
            x=values[order],
            y=[feature_names[i] for i in order],
            orientation="h",
            marker_color=["crimson" if v > 0 else "steelblue" for v in values[order]],
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
        color_continuous_scale="Blues",
        title="Confusion matrix",
    )
    return fig
