import datetime as dt
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
import numpy as np
# Import the required libraries
from statsmodels.tsa.ar_model import AutoReg


def fetch_stocks():
    # Load the data
    file_path = r"data\nifty50_companies.csv"
    df = pd.read_csv(file_path)
    # Filter the data
    df = df[["Security ID","Issuer Name"]]
    # Create a dictionary
    stock_dict = dict(zip(df["Issuer Name"],df["Security ID"]))
    # Return the dictionary
    return stock_dict

def fetch_periods_intervals():
    # Create dictionary for periods and intervals
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }
    # Return the dictionary
    return periods

def fetch_stock_info(stock_ticker):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)
    # Extract full of the stock
    stock_data_info = stock_data.info
    # Function to safely get value from dictionary or return "N/A"
    def safe_get(data_dict, key):
        return data_dict.get(key, "N/A")
    stock_data_info = {
        "Basic Information" : {
            "symbol" : safe_get(stock_data_info, "symbol"),
            "longName": safe_get(stock_data_info, "longName"),
            "currency": safe_get(stock_data_info, "currency"),
            "exchange": safe_get(stock_data_info, "exchange"),
        },
        "Market Data" : {
            "currentPrice" : safe_get(stock_data_info, "currentPrice"),
            "previousClose" : safe_get(stock_data_info, "previousClose"),
            "open" : safe_get(stock_data_info, "open"),
            "dayLow" : safe_get(stock_data_info, "dayLow"),
            "dayHigh" : safe_get(stock_data_info, "dayHigh"),
            "regularMarketPreviousClose" : safe_get(stock_data_info, "regularMarketPreviousClose"),
            "regularMarketOpen": safe_get(stock_data_info, "regularMarketOpen"),
            "regularMarketDayLow": safe_get(stock_data_info, "regularMarketDayLow"),
            "regularMarketDayHigh": safe_get(stock_data_info, "regularMarketDayHigh"),
        },
        "Volume and Shares": {
            "volume": safe_get(stock_data_info, "volume"),
            "regularMarketVolume": safe_get(stock_data_info, "regularMarketVolume"),
            "sharesOutstanding": safe_get(stock_data_info, "sharesOutstanding"),
        }    
    } 
    return stock_data_info

# Function to fetch the stock history
def fetch_stock_history(stock_ticker, period, interval):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    # Return the stock data
    return stock_data_history



# Function to generate the stock prediction
def generate_stock_prediction(stock_ticker):
    # Try to generate the predictions
    try:
        # Pull the data for the first security
        stock_data = yf.Ticker(stock_ticker)

        # Extract the data for last 1yr with 1d interval
        stock_data_hist = stock_data.history(period="2y", interval="1d")

        # Clean the data for to keep only the required columns
        stock_data_close = stock_data_hist[["Close"]]

        # Change frequency to day
        stock_data_close = stock_data_close.asfreq("D", method="ffill")

        # Fill missing values
        stock_data_close = stock_data_close.ffill()

        # Define training and testing area
        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]  # 90%
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]  # 10%

        # Define training model
        model = AutoReg(train_df["Close"], 250).fit(cov_type="HC0")

        # Predict data for test data
        predictions = model.predict(
            start=test_df.index[0], end=test_df.index[-1], dynamic=True
        )

        # Predict 90 days into the future
        forecast = model.predict(
            start=test_df.index[0],
            end=test_df.index[-1] + dt.timedelta(days=90),
            dynamic=True,
        )
        print("train")
        print(train_df)
        print("test")
        print(test_df)
        print("predictions")
        print(predictions)
        print("forecast")
        print(forecast)
        # Return the required data
        return train_df, test_df, forecast, predictions

    # If error occurs
    except:
        # Return None
        return None, None, None, None




