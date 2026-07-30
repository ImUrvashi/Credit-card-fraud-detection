"""Wires the static cards built in layout.py to live user interaction.

One-time setup (loading the best model, scoring the validation set)
happens once here at import time so callbacks stay cheap — no model
reloading or re-inference on every slider tick.
"""

import numpy as np
from dash import Input, Output, State, callback, html
from sklearn.metrics import f1_score, precision_score, recall_score

from dashboard import data_loaders as dl
from dashboard import figures as fig
from src.train import FEATURE_COLS

_results_df = dl.load_results()
_val_df = dl.load_val_df()
_best_run = dl.best_run_name(_results_df)
_best_model = dl.load_model(_best_run)

_X_val = _val_df[FEATURE_COLS]
_y_val = _val_df["Class"]
_val_scores = (
    _best_model.predict_proba(_X_val)[:, 1]
    if hasattr(_best_model, "predict_proba")
    else _best_model.decision_function(_X_val)
)

_live_sim_sequence = dl.load_live_sim_sequence()

# pca/tsne/umap embedding caches are the same sample, just projected
# differently, so any one of them's row_index maps a scatter click
# back to an original val_df row regardless of which is on screen.
_embedding_row_index = dl.load_embedding_cache("pca")["row_index"]


@callback(
    Output("comparison-graph", "figure"),
    Input("compare-run-a", "value"),
    Input("compare-run-b", "value"),
)
def update_comparison(run_a, run_b):
    return fig.comparison_bar_figure(_results_df, run_a, run_b)


@callback(Output("embedding-graph", "figure"), Input("embedding-method", "value"))
def update_embedding(method):
    cache = dl.load_embedding_cache(method)
    return fig.embedding_scatter_figure(cache, method)


@callback(
    Output("importance-graph", "figure"),
    Input("importance-model", "value"),
    Input("importance-chart-type", "value"),
)
def update_importance(model_name, chart_type):
    shap_cache = dl.load_shap_cache(model_name)
    return fig.feature_importance_figure(shap_cache, chart_type)


@callback(
    Output("shap-graph", "figure"),
    Input("importance-model", "value"),
    Input("embedding-graph", "clickData"),
)
def update_transaction_detail(model_name, click_data):
    """Shares the Feature Importance card's model dropdown — both cards
    are about "the currently selected model". A click on the embedding
    scatter picks which transaction; its pointIndex maps straight into
    cache['row_index'] since that figure uses one real trace (see
    figures.embedding_scatter_figure)."""
    shap_cache = dl.load_shap_cache(model_name)

    if click_data is None:
        row_position = dl.default_fraud_row_position(shap_cache, _val_df)
    else:
        point = click_data["points"][0]
        # Plotly.js documents "pointNumber" for cartesian traces;
        # "pointIndex" is a legacy alias not guaranteed on every trace
        # type, so prefer pointNumber and fall back defensively.
        point_index = point.get("pointNumber", point.get("pointIndex"))
        clicked_original_index = _embedding_row_index[point_index]
        matches = np.where(shap_cache["row_index"] == clicked_original_index)[0]
        # The clicked point may not be in a KernelExplainer model's
        # smaller sample (knn/naive_bayes) — fall back to a fraud case.
        row_position = int(matches[0]) if len(matches) else dl.default_fraud_row_position(shap_cache, _val_df)

    return fig.shap_waterfall_figure(shap_cache, row_position)


@callback(
    Output("threshold-cm-graph", "figure"),
    Output("threshold-metrics", "children"),
    Input("threshold-slider", "value"),
)
def update_threshold(threshold):
    y_pred = (_val_scores >= threshold).astype(int)
    precision = precision_score(_y_val, y_pred, zero_division=0)
    recall = recall_score(_y_val, y_pred, zero_division=0)
    f1 = f1_score(_y_val, y_pred, zero_division=0)
    flagged = int(y_pred.sum())

    metrics_text = html.Div(
        f"Precision {precision:.3f}  ·  Recall {recall:.3f}  ·  F1 {f1:.3f}  ·  "
        f"{flagged:,} transactions flagged as fraud"
    )
    return fig.confusion_matrix_figure(_y_val, y_pred), metrics_text


@callback(
    Output("threshold-curve-graph", "figure"),
    Input("threshold-slider", "value"),
    Input("threshold-curve-type", "value"),
)
def update_threshold_curve(threshold, curve_type):
    return fig.threshold_curve_figure(_y_val, _val_scores, threshold, curve_type)


@callback(
    Output("live-sim-interval", "disabled"),
    Output("live-sim-play", "children"),
    Input("live-sim-play", "n_clicks"),
    State("live-sim-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_live_sim(n_clicks, is_disabled):
    about_to_run = is_disabled
    return not is_disabled, ("⏸ Pause" if about_to_run else "▶ Play")


@callback(
    Output("live-sim-feed-store", "data"),
    Output("live-sim-index", "data"),
    Input("live-sim-interval", "n_intervals"),
    State("live-sim-feed-store", "data"),
    State("live-sim-index", "data"),
    prevent_initial_call=True,
)
def advance_live_sim(n_intervals, feed, index):
    row_position = index % len(_live_sim_sequence)
    row = _live_sim_sequence.iloc[row_position]

    score = float(_best_model.predict_proba(row[FEATURE_COLS].to_frame().T)[:, 1][0])
    entry = {
        "time_s": float(row["Time"]),
        "amount": float(row["Amount"]),
        "actual": "Fraud" if row["Class"] == 1 else "Legit",
        "score": score,
        "flagged": score >= 0.5,
    }

    feed = ([entry] + (feed or []))[:15]
    return feed, row_position + 1


@callback(Output("live-sim-feed", "children"), Input("live-sim-feed-store", "data"))
def render_live_sim_feed(feed):
    if not feed:
        return "Live simulation not started yet."

    header = html.Thead(html.Tr([html.Th(c) for c in ["Time", "Amount", "Actual", "Score", "Flagged"]]))
    rows = [
        html.Tr(
            [
                html.Td(f"{entry['time_s']:.0f}s"),
                html.Td(f"${entry['amount']:.2f}"),
                html.Td(entry["actual"]),
                html.Td(f"{entry['score']:.3f}"),
                html.Td("🚩 Fraud" if entry["flagged"] else "OK"),
            ],
            className="table-danger" if entry["flagged"] else "",
        )
        for entry in feed
    ]
    return html.Table([header, html.Tbody(rows)], className="table table-sm mb-0")
