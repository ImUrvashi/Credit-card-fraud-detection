"""Static card layout for the fraud detection dashboard. This is the
visual structure with sensible defaults loaded from disk — interactive
callbacks get wired on top of these same components next."""

import dash_bootstrap_components as dbc
import numpy as np
from dash import dash_table, dcc, html

from dashboard import data_loaders as dl
from dashboard import figures as fig
from src.train import FEATURE_COLS

LEADERBOARD_COLUMNS = ["model", "strategy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def navbar():
    return dbc.Navbar(
        dbc.Container(dbc.NavbarBrand("💳 Credit Card Fraud Detection", class_name="fw-bold fs-4")),
        color="dark",
        dark=True,
        class_name="mb-4",
    )


def card(title, body):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(body)], class_name="mb-4 shadow-sm")


def leaderboard_card(results_df):
    df = results_df.sort_values("pr_auc", ascending=False)
    table = dash_table.DataTable(
        id="leaderboard-table",
        columns=[{"name": c, "id": c} for c in LEADERBOARD_COLUMNS],
        data=df[LEADERBOARD_COLUMNS].round(4).to_dict("records"),
        sort_action="native",
        page_size=10,
        style_as_list_view=True,
        style_cell={"padding": "6px"},
        style_header={"fontWeight": "bold"},
    )
    return card("Leaderboard — every algorithm × imbalance strategy", table)


def comparison_card(results_df):
    run_options = [{"label": r, "value": r} for r in results_df["run"]]
    top_two = results_df.sort_values("pr_auc", ascending=False)["run"].iloc[:2].tolist()
    body = [
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id="compare-run-a", options=run_options, value=top_two[0]), width=6),
                dbc.Col(dcc.Dropdown(id="compare-run-b", options=run_options, value=top_two[1]), width=6),
            ],
            class_name="mb-3",
        ),
        dcc.Graph(id="comparison-graph", figure=fig.comparison_bar_figure(results_df, top_two[0], top_two[1])),
    ]
    return card("Compare two runs across metrics", body)


def timeline_card(train_df):
    return card("Fraud volume over time", dcc.Graph(id="timeline-graph", figure=fig.fraud_by_hour_figure(train_df)))


def embedding_card(pca_cache):
    body = [
        dcc.RadioItems(
            id="embedding-method",
            options=[{"label": m.upper(), "value": m} for m in ["pca", "tsne", "umap"]],
            value="pca",
            inline=True,
            inputClassName="me-1",
            labelClassName="me-3",
            className="mb-3",
        ),
        dcc.Graph(id="embedding-graph", figure=fig.embedding_scatter_figure(pca_cache, "pca")),
    ]
    return card("2D transaction embedding (click a point to inspect it)", body)


def feature_importance_card(results_df, default_shap_cache):
    model_options = [{"label": m, "value": m} for m in sorted(results_df["model"].unique())]
    body = [
        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id="importance-model", options=model_options, value="catboost"), width=6),
                dbc.Col(
                    dcc.RadioItems(
                        id="importance-chart-type",
                        options=[{"label": "Bar", "value": "bar"}, {"label": "Treemap", "value": "treemap"}],
                        value="bar",
                        inline=True,
                        inputClassName="me-1",
                        labelClassName="me-3",
                    ),
                    width=6,
                ),
            ],
            class_name="mb-3",
        ),
        dcc.Graph(id="importance-graph", figure=fig.feature_importance_figure(default_shap_cache, "bar")),
    ]
    return card("Feature importance", body)


def shap_detail_card(default_shap_cache, val_df):
    class_labels = val_df.loc[default_shap_cache["row_index"], "Class"].to_numpy()
    fraud_positions = np.where(class_labels == 1)[0]
    default_row = int(fraud_positions[0]) if len(fraud_positions) else 0

    graph = dcc.Graph(id="shap-graph", figure=fig.shap_waterfall_figure(default_shap_cache, default_row))
    return card("Transaction detail — SHAP contributions", graph)


def threshold_card(results_df, val_df):
    best_run = dl.best_run_name(results_df)
    model = dl.load_model(best_run)
    X_val, y_val = val_df[FEATURE_COLS], val_df["Class"]
    y_scores = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
    y_pred = (y_scores >= 0.5).astype(int)

    body = [
        dcc.Slider(id="threshold-slider", min=0, max=1, step=0.01, value=0.5, marks={0: "0", 0.5: "0.5", 1: "1"}),
        html.Div(id="threshold-metrics", className="my-3"),
        dcc.Graph(id="threshold-cm-graph", figure=fig.confusion_matrix_figure(y_val, y_pred)),
    ]
    return card(f"Decision threshold tuner ({best_run})", body)


def live_sim_card():
    body = [
        dbc.Button("▶ Play", id="live-sim-play", color="primary", class_name="mb-3"),
        html.Div(id="live-sim-feed", children="Live simulation not started yet."),
    ]
    return card("Simulated live transaction stream", body)


def build_layout():
    results_df = dl.load_results()
    train_df = dl.load_train_df()
    val_df = dl.load_val_df()
    pca_cache = dl.load_embedding_cache("pca")
    default_shap_cache = dl.load_shap_cache("catboost")

    return html.Div(
        [
            navbar(),
            dbc.Container(
                [
                    leaderboard_card(results_df),
                    comparison_card(results_df),
                    timeline_card(train_df),
                    embedding_card(pca_cache),
                    feature_importance_card(results_df, default_shap_cache),
                    shap_detail_card(default_shap_cache, val_df),
                    threshold_card(results_df, val_df),
                    live_sim_card(),
                ]
            ),
        ]
    )
