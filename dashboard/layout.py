"""Static card layout for the fraud detection dashboard. This is the
visual structure with sensible defaults loaded from disk — interactive
callbacks get wired on top of these same components next."""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from dashboard import data_loaders as dl
from dashboard import figures as fig
from src.train import FEATURE_COLS

LEADERBOARD_COLUMNS = ["model", "strategy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def navbar():
    brand = html.Div(
        [
            html.Div("💳 Credit Card Fraud Detection", className="fw-bold fs-4 text-white"),
            html.Div(
                "9 algorithms × 7 imbalance strategies, compared and explained",
                className="navbar-subtitle",
            ),
        ]
    )
    return dbc.Navbar(dbc.Container(brand), color="dark", dark=True, class_name="mb-4 py-3")


def card(title, body):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(body)], class_name="mb-4 shadow-sm")


def stat_tile(label, value):
    return dbc.Col(
        html.Div(
            [html.Div(value, className="stat-tile-value"), html.Div(label, className="stat-tile-label")],
            className="stat-tile",
        ),
        md=3,
        xs=6,
    )


def dataset_overview_card(train_df, val_df, test_df):
    total = len(train_df) + len(val_df) + len(test_df)
    fraud_total = int(train_df["Class"].sum() + val_df["Class"].sum() + test_df["Class"].sum())
    fraud_rate = fraud_total / total * 100

    stats = dbc.Row(
        [
            stat_tile("Transactions", f"{total:,}"),
            stat_tile("Fraud cases", f"{fraud_total:,}"),
            stat_tile("Fraud rate", f"{fraud_rate:.3f}%"),
            stat_tile("Features", "30"),
        ],
        class_name="mb-4 g-3",
    )

    description = html.P(
        [
            "Transactions made by European cardholders in September 2013, over two days "
            "(source: ",
            html.A(
                "Kaggle — mlg-ulb/creditcardfraud",
                href="https://www.kaggle.com/mlg-ulb/creditcardfraud",
                target="_blank",
            ),
            "). To protect cardholder identity, 28 of the 30 features (V1–V28) are "
            "anonymized components from a PCA transform — only Time (seconds since the "
            "first transaction) and Amount are original. Fraud makes up just "
            f"{fraud_rate:.3f}% of transactions, so this project compares 9 classification "
            "algorithms against 7 imbalance-handling strategies (class weighting, "
            "over/undersampling, SMOTE, ADASYN, SMOTETomek) and ranks them primarily by "
            "PR-AUC (average precision), since accuracy — and even ROC-AUC — are both "
            "misleading at this level of class imbalance.",
        ],
        className="mb-0 text-secondary",
    )

    return card("Dataset overview", [stats, description])


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
    default_row = dl.default_fraud_row_position(default_shap_cache, val_df)
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
        dcc.RadioItems(
            id="threshold-curve-type",
            options=[{"label": "PR curve", "value": "pr"}, {"label": "ROC curve", "value": "roc"}],
            value="pr",
            inline=True,
            inputClassName="me-1",
            labelClassName="me-3",
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="threshold-curve-graph", figure=fig.threshold_curve_figure(y_val, y_scores, 0.5, "pr")),
                    width=6,
                ),
                dbc.Col(dcc.Graph(id="threshold-cm-graph", figure=fig.confusion_matrix_figure(y_val, y_pred)), width=6),
            ]
        ),
    ]
    return card(f"Decision threshold tuner ({best_run})", body)


def live_sim_card():
    body = [
        dbc.Button("▶ Play", id="live-sim-play", color="primary", class_name="mb-3"),
        dcc.Interval(id="live-sim-interval", interval=1200, n_intervals=0, disabled=True),
        dcc.Store(id="live-sim-feed-store", data=[]),
        dcc.Store(id="live-sim-index", data=0),
        html.Div(id="live-sim-feed", children="Live simulation not started yet."),
    ]
    return card("Simulated live transaction stream (held-out test data, replayed in time order)", body)


def build_layout():
    results_df = dl.load_results()
    train_df = dl.load_train_df()
    val_df = dl.load_val_df()
    test_df = dl.load_test_df()
    pca_cache = dl.load_embedding_cache("pca")
    default_shap_cache = dl.load_shap_cache("catboost")

    return html.Div(
        [
            navbar(),
            dbc.Container(
                [
                    dataset_overview_card(train_df, val_df, test_df),
                    timeline_card(train_df),
                    leaderboard_card(results_df),
                    comparison_card(results_df),
                    feature_importance_card(results_df, default_shap_cache),
                    shap_detail_card(default_shap_cache, val_df),
                    embedding_card(pca_cache),
                    threshold_card(results_df, val_df),
                    live_sim_card(),
                ]
            ),
        ]
    )
