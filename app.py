import dash
from dash import dcc
from dash import html
from datetime import datetime as dt
import yfinance as yf
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd 
import plotly.graph_objs as go
import plotly.express as px
# model
from model import prediction
from sklearn.svm import SVR


def get_stock_price_fig(df):
    # Check if the 'Close' and 'Open' columns exist
    if 'Close' not in df.columns or 'Open' not in df.columns:
        return None
    
    # Flatten 'Close' and 'Open' columns
    df['Close'] = df['Close'].values.flatten() if len(df['Close'].values.shape) > 1 else df['Close']
    df['Open'] = df['Open'].values.flatten() if len(df['Open'].values.shape) > 1 else df['Open']
    
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
    return fig


def get_more(df):
    # Ensure 'Close' is in the DataFrame
    if 'Close' not in df.columns:
        return None
    
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.line(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    return fig



app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Roboto&display=swap"
    ])
server = app.server
# html layout of site
app.layout = html.Div(
    [
        html.Div(
            [
                # Navigation
                html.P("Welcome to the Stock Dash App!", className="start"),
                html.Div([
                    html.P("Input stock code: "),
                    html.Div([
                        dcc.Input(id="dropdown_tickers", type="text"),
                        html.Button("Submit", id='submit'),
                    ],
                             className="form")
                ],
                         className="input-place"),
                html.Div([
                    dcc.DatePickerRange(id='my-date-picker-range',
                                        min_date_allowed=dt(1995, 8, 5),
                                        max_date_allowed=dt.now(),
                                        initial_visible_month=dt.now(),
                                        end_date=dt.now().date()),
                ],
                         className="date"),
                html.Div([
                    html.Button(
                        "Stock Price", className="stock-btn", id="stock"),
                    html.Button("Indicators",
                                className="indicators-btn",
                                id="indicators"),
                    dcc.Input(id="n_days",
                              type="text",
                              placeholder="number of days"),
                    html.Button(
                        "Forecast", className="forecast-btn", id="forecast")
                ],
                         className="buttons"),
                # here
            ],
            className="nav"),

        # content
        html.Div(
            [
                html.Div(
                    [  # header
                        html.Img(id="logo"),
                        html.P(id="ticker")
                    ],
                    className="header"),
                html.Div(id="description", className="decription_ticker"),
                html.Div([], id="graphs-content"),
                html.Div([], id="main-content"),
                html.Div([], id="forecast-content")
            ],
            className="content"),
    ],
    className="container")


# callback for company info
@app.callback([
    Output("description", "children"),
    Output("logo", "src"),
    Output("ticker", "children"),
    Output("stock", "n_clicks"),
    Output("indicators", "n_clicks"),
    Output("forecast", "n_clicks")
], [Input("submit", "n_clicks")], 
   [State("dropdown_tickers", "value")]
)
def update_data(n, val):  # input parameter(s)
    if n is None:
        return (
            "Hey there! Please enter a legitimate stock code to get details.",
            "https://melmagazine.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-31-at-5.47.12-PM.png",
            "Stonks",
            None, None, None
        )
    else:
        if val is None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T

            # Check if each field exists, provide default values if not
            logo_url = df['logo_url'].values[0] if 'logo_url' in df.columns else "https://melmagazine.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-31-at-5.47.12-PM.png"
            short_name = df['shortName'].values[0] if 'shortName' in df.columns else "Unknown Stock"
            long_business_summary = df['longBusinessSummary'].values[0] if 'longBusinessSummary' in df.columns else "No description available."

            return long_business_summary, logo_url, short_name, None, None, None

@app.callback([
    Output("graphs-content", "children"),
], [
    Input("stock", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def stock_price(n, start_date, end_date, val):
    if n is None:
        return [""]

    if val is None:
        return ["Please enter a stock code."]

    try:
        # Fetch the stock data
        df = yf.download(val, start=start_date, end=end_date) if start_date else yf.download(val)
        
        # Flatten the MultiIndex if it exists
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Extracts the price names, like 'Close', 'Open'
        
        # Check if required columns are present
        if 'Close' not in df.columns or 'Open' not in df.columns:
            return ["No data available for the selected stock code or date range."]
        
        df.reset_index(inplace=True)
        fig = get_stock_price_fig(df)
        if fig is None:
            return ["Could not generate stock price figure due to data issues."]
    except Exception as e:
        return [f"Error fetching stock price data: {e}"]

    return [dcc.Graph(figure=fig)]



@app.callback([Output("main-content", "children")], [
    Input("indicators", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def indicators(n, start_date, end_date, val):
    if n is None:
        return [""]
    
    if val is None:
        return ["Please enter a stock code."]
    
    try:
        df_more = yf.download(val, start=start_date, end=end_date) if start_date else yf.download(val)
        
        # Flatten MultiIndex columns
        if isinstance(df_more.columns, pd.MultiIndex):
            df_more.columns = df_more.columns.get_level_values(0)
        
        # Check if 'Close' is in the DataFrame
        if 'Close' not in df_more.columns:
            return ["No indicator data available for the selected date range."]
        
        df_more.reset_index(inplace=True)
        fig = get_more(df_more)
        if fig is None:
            return ["Could not compute indicators. Ensure the stock code is correct."]
    except Exception as e:
        return [f"Error generating indicators: {e}"]

    return [dcc.Graph(figure=fig)]





@app.callback([Output("forecast-content", "children")],
              [Input("forecast", "n_clicks")],
              [State("n_days", "value"),
               State("dropdown_tickers", "value")])
def forecast(n, n_days, val):
    if n is None:
        return [""]

    if val is None:
        return ["Please enter a stock code."]
    
    try:
        n_days = int(n_days) if n_days else 7  # Default to 7 if not provided
    except ValueError:
        return ["Invalid number of days entered."]
    
    try:
        # Fetch data for forecasting with a valid period
        print("Fetching data for forecast with period '1y'")
        df = yf.download(val, period="1y")  # Set period to '1y' to ensure at least one year of data

        # Debugging output to verify data structure and period used
        print("Data fetched for forecast:", df.head())

        # Validate data structure and length
        if df.empty or 'Close' not in df.columns:
            return ["Insufficient data to generate a forecast. Please choose a different stock or try a longer period."]
        
        # Check if there are enough samples for forecasting
        if len(df) < n_days:
            return ["Not enough data points for the requested forecast period. Please reduce the number of forecast days or choose a different stock."]

        # Call the prediction function with the ticker and days
        fig = prediction(val, n_days)
    except Exception as e:
        return [f"Error generating forecast: {e}"]

    return [dcc.Graph(figure=fig)]



if __name__ == '__main__':
    app.run_server(debug=True)