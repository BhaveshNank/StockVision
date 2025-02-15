import dash
from dash import html
import dash_bootstrap_components as dbc

# This app is for developers only
app_debug = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY]
)

app_debug.layout = dbc.Container(
    [
        html.H1("Developer Debug Panel", className="text-center text-light mb-4"),
        dbc.Card(
            dbc.CardBody("Server logs, callbacks and error details will appear here..."),
            className="bg-dark text-light"
        )
    ],
    fluid=True,
    className="bg-secondary p-4"
)

if __name__ == '__main__':
    app_debug.run_server(debug=True)
