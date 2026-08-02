"""Entry point for the Dash dashboard.

Local dev:  python -m dashboard.app
Production: gunicorn dashboard.app:server
"""

import os

import dash
import dash_bootstrap_components as dbc

from dashboard.layout import build_layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Credit Card Fraud Detection")
app.layout = build_layout()
server = app.server  # the underlying Flask app - what gunicorn/a WSGI host needs

# Imported after `app` exists: dash's @callback decorator registers
# against the Dash app created in this process.
from dashboard import callbacks  # noqa: E402,F401

if __name__ == "__main__":
    # Render (and most hosts) set PORT and expect the app to bind
    # 0.0.0.0 - Dash's own defaults (127.0.0.1:8050) are local-only and
    # won't be detected as a listening port there. No PORT env var set
    # is a reasonable signal we're running locally, so default debug
    # mode (hot reload, etc.) on in that case and off otherwise.
    port = int(os.environ.get("PORT", 8050))
    is_deployed = "PORT" in os.environ
    app.run(host="0.0.0.0", port=port, debug=not is_deployed)
