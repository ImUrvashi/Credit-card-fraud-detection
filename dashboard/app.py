"""Entry point for the Dash dashboard. Run with:

    python -m dashboard.app
"""

import dash
import dash_bootstrap_components as dbc

from dashboard.layout import build_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Credit Card Fraud Detection")
app.layout = build_layout()

# Imported after `app` exists: dash's @callback decorator registers
# against the Dash app created in this process.
from dashboard import callbacks  # noqa: E402,F401

if __name__ == "__main__":
    app.run(debug=True)
