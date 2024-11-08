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
import requests
from datetime import datetime, timedelta
from dash import Dash, html, dcc, Input, Output
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Dash(__name__)

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

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    return [(headline, analyzer.polarity_scores(headline)['compound']) for headline in headlines]

# html layout of site
app.layout = html.Div(
    [
        # Left Sidebar (for input controls)
        html.Div(
            [
                html.P("Welcome to the Stock Dash App!", className="start"),

                # Input stock code and Fetch News button
                html.Div([
                    html.Label("Input stock code:"),
                    dcc.Input(id="stock-input", type="text", placeholder="Enter a stock symbol..."),
                    html.Button("Fetch News", id="fetch-news", n_clicks=0, style={'margin-left': '5px'}),
                ], className="form", style={'margin-bottom': '20px'}),

                # Date Picker
                html.Label("Select Date Range:"),
                dcc.DatePickerRange(
                    id='my-date-picker-range',
                    min_date_allowed=dt(1995, 8, 5),
                    max_date_allowed=dt.now(),
                    initial_visible_month=dt.now(),
                    end_date=dt.now().date(),
                    style={'margin-bottom': '20px'}
                ),

                # Stock Data and Indicators Buttons
                html.Div([
                    html.Button("Stock Price", className="stock-btn", id="stock"),
                    html.Button("Indicators", className="indicators-btn", id="indicators", style={'margin-left': '10px'}),
                ], style={'display': 'flex', 'margin-bottom': '20px'}),

                # Forecast section
                html.Div([
                    dcc.Input(id="n_days", type="text", placeholder="Number of forecast days", style={'width': '150px'}),
                    html.Button("Forecast", className="forecast-btn", id="forecast", style={'margin-left': '10px'}),
                ], style={'margin-bottom': '20px'}),

                # Technical Indicator Dropdown
                dcc.Dropdown(
                    id="indicator-dropdown",
                    options=[
                        {"label": "MACD", "value": "MACD"},
                        {"label": "RSI", "value": "RSI"},
                        {"label": "Bollinger Bands", "value": "Bollinger"}
                    ],
                    placeholder="Select an Indicator",
                    style={'margin-bottom': '20px'}
                ),
            ],
            className="sidebar",  # Apply CSS class for styling
            style={'width': '25%', 'padding': '20px', 'background-color': '#f7f7f7'}
        ),

        # Main Content Area (for displaying outputs)
        html.Div(
            [
                # Header with logo and ticker display
                html.Div([
                    html.Img(id="logo", style={'height': '50px'}),
                    html.P(id="ticker", style={'font-size': '24px', 'margin-top': '10px'}),
                ], className="header", style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),

                # Description section
                html.Div(id="description", className="description", style={'margin-bottom': '20px'}),

                # Graphs for stock price and indicators
                html.Div(id="graphs-content", className="graph-section", style={'margin-bottom': '20px'}),

                # Technical indicators or forecast data
                html.Div(id="main-content", className="indicator-section", style={'margin-bottom': '20px'}),
                html.Div(id="forecast-content", className="forecast-section", style={'margin-bottom': '20px'}),

                # News and sentiment analysis
                html.Div(id='news-container', children=[
                    html.H2('Latest News'),
                    html.Ul(id='news-list')
                ], className="news-section", style={'margin-bottom': '20px'}),

                html.Div(id='sentiment-output', children=[
                    html.H2('News Sentiment Analysis'),
                    html.Div(id='sentiment-details')
                ], className="sentiment-section")
            ],
            className="main-content",  # Apply CSS class for styling
            style={'width': '70%', 'padding': '20px'}
        ),
    ],
    className="container",
    style={'display': 'flex', 'justify-content': 'space-between'}
)






# callback for company info
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("ticker", "children"),
        Output("stock", "n_clicks"),
        Output("indicators", "n_clicks"),
        Output("forecast", "n_clicks")
    ],
    [Input("stock-input", "value")]
)
def update_data(stock_code):
    if stock_code is None:
        return (
            "Please enter a legitimate stock code to get details.",
            "https://melmagazine.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-31-at-5.47.12-PM.png",
            "Stonks",
            None, None, None
        )
    else:
        try:
            ticker = yf.Ticker(stock_code)
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T

            logo_url = df['logo_url'].values[0] if 'logo_url' in df.columns else "https://melmagazine.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-31-at-5.47.12-PM.png"
            short_name = df['shortName'].values[0] if 'shortName' in df.columns else "Unknown Stock"
            long_business_summary = df['longBusinessSummary'].values[0] if 'longBusinessSummary' in df.columns else "No description available."

            return long_business_summary, logo_url, short_name, None, None, None

        except yf.YFException as e:
            return (
                "Stock data not found. Please check the stock symbol.",
                "https://melmagazine.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-31-at-5.47.12-PM.png",
                "Unknown Stock",
                None, None, None
            )
        except Exception as e:
            return (
                f"An error occurred: {e}",
                "https://melmagazine.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-31-at-5.47.12-PM.png",
                "Unknown Stock",
                None, None, None
            )


@app.callback(
    Output("graphs-content", "children"),
    [Input("stock", "n_clicks"),
     Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date')],
    [State("stock-input", "value")]
)
def stock_price(n, start_date, end_date, val):
    if n is None:
        return [""]

    if val is None:
        return ["Please enter a stock code."]
    
    # Fetch stock data logic here


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
    [State("stock-input", "value"), State('my-date-picker-range', 'start_date'), State('my-date-picker-range', 'end_date')]
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


@app.callback(
    [Output("forecast-content", "children")],
    [Input("forecast", "n_clicks")],
    [State("n_days", "value"), State("stock-input", "value")]
)
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

def fetch_news(stock_symbol):
    API_KEY = 'csmkjthr01qn12jeuc5gcsmkjthr01qn12jeuc60'  # Replace 'YOUR_API_KEY' with the actual key
    yesterday = datetime.now() - timedelta(days=1)
    today = datetime.now()
    endpoint = f'https://finnhub.io/api/v1/company-news?symbol={stock_symbol}&from={yesterday:%Y-%m-%d}&to={today:%Y-%m-%d}&token={API_KEY}'

    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()  # Returns a list of news articles
    else:
        return []
    
# Combined Callback for News and Sentiment Analysis (MERGED CALLBACK)
@app.callback(
    [Output('news-list', 'children'),
     Output('sentiment-details', 'children')],  # Added output for sentiment details
    [Input('fetch-news', 'n_clicks')],
    [State('stock-input', 'value')],
    prevent_initial_call=True
)
def update_news_combined(n_clicks, stock_symbol):
    ticker = stock_symbol

    if not ticker:
        return [html.Li("Please enter a stock symbol first.")], ""

    # Fetch news from the API
    news_items = fetch_news(ticker)
    if not news_items:
        return [html.Li('No news found for this stock symbol.')], "No sentiment data available."

    # Perform sentiment analysis
    analyzed_headlines = analyze_sentiment([news['headline'] for news in news_items])
    
    # Generate list items with sentiment for news display
    news_elements = [
        html.Li(f"{headline}: {'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'} (Score: {score})")
        for headline, score in analyzed_headlines
    ]

    # Calculate overall sentiment summary
    positive = sum(1 for _, score in analyzed_headlines if score > 0)
    neutral = sum(1 for _, score in analyzed_headlines if score == 0)
    negative = sum(1 for _, score in analyzed_headlines if score < 0)
    average_score = sum(score for _, score in analyzed_headlines) / len(analyzed_headlines)

    sentiment_summary = html.Div([
        html.P(f"Overall Sentiment Summary:"),
        html.P(f"Positive Articles: {positive}"),
        html.P(f"Neutral Articles: {neutral}"),
        html.P(f"Negative Articles: {negative}"),
        html.P(f"Average Sentiment Score: {average_score:.2f}")
    ])

    return news_elements, sentiment_summary


if __name__ == '__main__':
    app.run_server(debug=True)