def prediction(stock, n_days):
    # import dash
    # import dash_core_components as dcc
    # import dash_html_components as html
    # from datetime import datetime as dt
    import yfinance as yf
    # from dash.dependencies import Input, Output, State
    # from dash.exceptions import PreventUpdate
    import pandas as pd
    import plotly.graph_objs as go
    import plotly.express as px
    # model
    # from model import prediction
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    from sklearn.svm import SVR
    from datetime import date, timedelta

    # Fetch the data with a valid period, ensuring we have enough data points
    print(f"Fetching data for {stock} with period '1y' in prediction function")
    df = yf.download(stock, period='1y')  # Using '1y' for at least one year of data
    df.reset_index(inplace=True)

    # Debugging output to verify data structure and period used
    print("Data fetched in prediction function:", df.head())

    # Check if the data is sufficient
    if df.empty or 'Close' not in df.columns:
        raise ValueError("Insufficient data for forecasting. Please choose a different stock or a longer data period.")

    # Ensure there are enough samples for the forecast
    if len(df) < n_days:
        raise ValueError("Not enough data points for the requested forecast period. Try reducing the forecast days or selecting a different stock.")

    df['Day'] = df.index

    days = list()
    for i in range(len(df.Day)):
        days.append([i])

    # Splitting the dataset
    X = days
    Y = df[['Close']]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 100, 1000],
            'epsilon': [
                0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,
                50, 100, 150, 1000
            ],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
        },
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )

    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf',
                   C=best_params["C"],
                   epsilon=best_params["epsilon"],
                   gamma=best_params["gamma"],
                   max_iter=-1)

    # Support Vector Regression Model
    rbf_svr = best_svr
    rbf_svr.fit(x_train, y_train)

    output_days = list()
    for i in range(1, n_days):
        output_days.append([i + x_test[-1][0]])

    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days=1)
        dates.append(current)

    # Plot Results
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,  # np.array(ten_days).flatten(), 
            y=rbf_svr.predict(output_days),
            mode='lines+markers',
            name='data'
        )
    )
    fig.update_layout(
        title="Predicted Close Price of next " + str(n_days - 1) + " days",
        xaxis_title="Date",
        yaxis_title="Closed Price",
    )

    return fig
