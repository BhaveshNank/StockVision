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
from dash import callback_context
from dash import no_update


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
    html.Button("Stock Price", className="stock-btn", id="stock"),
    html.Button("Indicators", className="indicators-btn", id="indicators"),
    dcc.Input(id="n_days", type="text", placeholder="number of days"),
    html.Button("Forecast", className="forecast-btn", id="forecast"),
    # Dropdown for Technical Indicators
    dcc.Dropdown(
        id="indicator-dropdown",
        options=[
            {"label": "MACD", "value": "MACD"},
            {"label": "RSI", "value": "RSI"},
            {"label": "Bollinger Bands", "value": "Bollinger"}
        ],
        placeholder="Select an Indicator"
    )
], className="buttons"),

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


# New unified indicator callback
# Unified indicator callback


def calculate_bollinger_bands(df):
    df['MA'] = df['Close'].rolling(window=20).mean()
    df['STD'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA'] + (df['STD'] * 2)
    df['Lower'] = df['MA'] - (df['STD'] * 2)
    return df

def create_bollinger_figure(df, val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA'], mode='lines', name='Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Lower Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', opacity=0.5))
    fig.update_layout(title=f"Bollinger Bands for {val}", xaxis_title="Date", yaxis_title="Price")
    return fig

def handle_indicator_button(n_clicks, val, start_date, end_date):
    # Placeholder function to simulate handling of the indicators button
    if not val:
        return "Please enter a stock code to get indicators."
    try:
        df = yf.download(val, start=start_date, end=end_date)
        if df.empty:
            return "No data available for this ticker in the selected date range."
        # Example of what might be done; really depends on your application needs
        df = calculate_macd(df)  # Assuming calculate_macd uses just df
        fig = create_macd_figure(df, val)
        return [dcc.Graph(figure=fig)]
    except Exception as e:
        return f"Error processing indicators: {e}"


@app.callback(
    Output("main-content", "children"),
    [Input("indicator-dropdown", "value"), Input("indicators", "n_clicks")],
    [State("dropdown_tickers", "value"), State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')]
)
def update_main_content(indicator, n_clicks, val, start_date, end_date):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered == "indicator-dropdown" and indicator:
        if not val:
            return "Please enter a valid stock symbol."

        try:
            df = yf.download(val, start=start_date, end=end_date)
            if df.empty:
                return "No data available for this ticker in the selected date range."
        except Exception as e:
            return f"Error downloading data: {e}"

        if 'Close' in df.columns:
            if indicator == "MACD":
                df = calculate_macd(df)
                return [dcc.Graph(figure=create_macd_figure(df, val))]
            elif indicator == "RSI":
                df = calculate_rsi(df)
                return [dcc.Graph(figure=create_rsi_figure(df, val))]
            elif indicator == "Bollinger":
                df = calculate_bollinger_bands(df)
                return [dcc.Graph(figure=create_bollinger_figure(df, val))]
        else:
            return "Required data columns missing."
    elif triggered == "indicators" and n_clicks:
        # This function should be defined to handle what happens when 'indicators' button is clicked
        return handle_indicator_button(n_clicks, val, start_date, end_date)

    return no_update





# Example helper functions
def calculate_macd(df):
    df['12_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_EMA'] - df['26_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def create_macd_figure(df, val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line'))
    fig.update_layout(title=f"MACD for {val}", xaxis_title="Date", yaxis_title="MACD Value")
    return fig

def calculate_rsi(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def create_rsi_figure(df, val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.update_layout(title=f"RSI for {val}", xaxis_title="Date", yaxis_title="RSI Value", yaxis=dict(range=[0, 100]))
    return fig


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