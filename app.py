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
import dash_bootstrap_components as dbc


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
    external_stylesheets=[dbc.themes.DARKLY],  # Replace 'DARKLY' with your preferred theme
)
server = app.server

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score

def create_sentiment_gauge(score):
    # Ensure the score is valid
    if score is None or not isinstance(score, (int, float)) or not (-1 <= score <= 1):
        return go.Figure(layout={"xaxis": {"visible": False}, "yaxis": {"visible": False}})

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={'text': "Sentiment Analysis", 'font': {'size': 24, 'color': "white"}},
        delta={'reference': 0, 'increasing': {'color': "limegreen"}, 'decreasing': {'color': "red"}},
        number={'font': {'size': 36, 'color': "white"}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': "royalblue", 'thickness': 0.4},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.6], 'color': "#ff4d4d"},
                {'range': [-0.6, -0.3], 'color': "#ff8080"},
                {'range': [-0.3, 0.3], 'color': "#ffff80"},
                {'range': [0.3, 0.6], 'color': "#80ff80"},
                {'range': [0.6, 1], 'color': "#33cc33"}
            ],
            'threshold': {
                'line': {'color': "gold", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return gauge



# Callback for News and Sentiment Analysis with style update
@app.callback(
    [Output('news-list', 'children'),
     Output('sentiment-gauge', 'figure'),
     Output('sentiment-gauge', 'style')],
    [Input('fetch-news', 'n_clicks')],
    [State('stock-input', 'value')],
    prevent_initial_call=True
)
def update_news_and_sentiment_gauge(n_clicks, stock_symbol):
    if not stock_symbol:
        return [html.Li("Please enter a stock symbol first.")], go.Figure(), {'display': 'none'}

    # Fetch news from the API
    news_items = fetch_news(stock_symbol)
    if not news_items:
        return [html.Li('No news found for this stock symbol.')], go.Figure(), {'display': 'none'}

    # Perform sentiment analysis on headlines
    headlines = [news['headline'] for news in news_items if news.get('headline')]
    average_sentiment = analyze_sentiment(headlines)

    # Create list of news items with clickable links
    news_elements = [
        html.Li([
            html.A(news['headline'], href=news['url'], target="_blank")
        ]) for news in news_items if news.get('headline') and news.get('url')
    ]

    # Generate the sentiment gauge
    fig = create_sentiment_gauge(average_sentiment)

    # Toggle visibility based on sentiment score
    if average_sentiment != 0:
        return news_elements, fig, {'display': 'block'}
    else:
        return news_elements, go.Figure(), {'display': 'none'}



# html layout of site

app.layout = dbc.Container(
    [
        # Title Row

            dbc.Row(
                dbc.Col(
                    html.H1(
                        "Stock Dashboard App",
                        className="text-center mb-4",
                        style={"color": "white"}  # Change text color to white
                    ),
                    width=12
                )
            ),

        
        # Input Controls (Sidebar)
        dbc.Row(
            [
                # Left Sidebar
                dbc.Col(
                    [
                        html.P("Welcome to the Stock Dash App!", className="text-light mb-4"),
                        
                        # Input Stock Code and Fetch News Button
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Input(
                                        id="stock-input",
                                        type="text",
                                        placeholder="Enter a stock symbol...",
                                        className="form-control"
                                    ),
                                    width=True,
                                ),
                                dbc.Col(
                                    html.Button("Fetch News", id="fetch-news", n_clicks=0, className="btn btn-info"),
                                    width="auto",
                                ),
                            ],
                            className="mb-3",
                        ),
                        
                        # Date Picker and Stock Price Button
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.DatePickerRange(
                                        id='my-date-picker-range',
                                        min_date_allowed=dt(1995, 8, 5),
                                        max_date_allowed=dt.now(),
                                        initial_visible_month=dt.now(),
                                        end_date=dt.now().date(),
                                        className="form-control",
                                    ),
                                    width=True,
                                ),
                                dbc.Col(
                                    html.Button("Stock Price", id="stock", n_clicks=0, className="btn btn-primary"),
                                    width="auto",
                                ),
                            ],
                            className="mb-3",
                        ),

                        # Forecast Section (Aligned with Input)
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Input(
                                        id="n_days",
                                        type="text",
                                        placeholder="Enter number of forecast days",
                                        className="form-control",
                                    ),
                                    width=True,
                                ),
                                dbc.Col(
                                    html.Button("Forecast", id="forecast", n_clicks=0, className="btn btn-success"),
                                    width="auto",
                                ),
                            ],
                            className="mb-3",
                        ),

                        # Indicators Section (Aligned with Dropdown)
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="indicator-dropdown",
                                        options=[
                                            {"label": "MACD", "value": "MACD"},
                                            {"label": "RSI", "value": "RSI"},
                                            {"label": "Bollinger Bands", "value": "Bollinger"},
                                        ],
                                        placeholder="Select an Indicator",
                                        className="form-control",
                                    ),
                                    width=True,
                                ),
                                dbc.Col(
                                    html.Button("Indicators", id="indicators", n_clicks=0, className="btn btn-warning"),
                                    width="auto",
                                ),
                            ],
                            className="mb-3",
                        ),
                    ],
                    width=4,
                    className="bg-dark p-4 rounded"
                ),
                
                # Main Content Area
                dbc.Col(
                    [
                        # Header (Ticker and Description)
                        dbc.Row(
                            [
                                dbc.Col(html.P(id="ticker", className="text-light h3"), width=10),
                            ],
                            className="mb-4"
                        ),
                        
                        # Description Section
                        html.Div(id="description", className="text-light mb-4"),
                        
                        # Graphs Content
                        html.Div(id="graphs-content", className="mb-4"),
                        
                        # Indicators or Forecast Data
                        html.Div(id="main-content", className="mb-4"),
                        html.Div(id="forecast-content", className="mb-4"),
                        
                        # News Section
                        html.Div(
                            [
                                html.H2("Latest News", className="text-light"),
                                html.Ul(id="news-list", className="list-unstyled"),
                            ],
                            className="news-section mb-4",
                        ),
                        
                        # Sentiment Analysis Gauge
                        html.Div(
                            [
                                html.H2("News Sentiment Analysis", className="text-light"),
                                dcc.Graph(id="sentiment-gauge", style={"display": "none"}),  # Initially hidden
                            ],
                            className="sentiment-section",
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
    className="bg-secondary text-light p-4",
)














# callback for company info
@app.callback(
    [
        Output("description", "children"),
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
            "Stonks",
            None, None, None
        )
    else:
        try:
            ticker = yf.Ticker(stock_code)
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T

            
            short_name = df['shortName'].values[0] if 'shortName' in df.columns else "Unknown Stock"
            long_business_summary = df['longBusinessSummary'].values[0] if 'longBusinessSummary' in df.columns else "No description available."

            return long_business_summary, short_name, None, None, None

        except Exception as e:
            # Handling any exception and providing feedback to the user
            return (
                "Stock data not found. Please check the stock symbol.",
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
    

if __name__ == '__main__':
    app.run_server(debug=True)