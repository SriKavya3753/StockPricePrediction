import streamlit as st
import sys
import os

from helper import *

st.set_page_config(
    page_title="Stock Info",
    page_icon="🏛",
)

#####Sidebar Start#####

#Add a sidebar
st.sidebar.markdown("## User Input Features")

# Fetch and store the stock data
stock_dict = fetch_stocks()

# Add a dropdown for selecting the stock
st.sidebar.markdown("### Select stock")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

# Build the stock ticker
stock_ticker = f"{stock_dict[stock]}"


# Add a disabled input for stock ticker
st.sidebar.markdown("### Stock ticker")
st.sidebar.text_input(
    label="Stock ticker code", placeholder=stock_ticker, disabled=True
)


#####Sidebar End#####

# Fetch the info of the stock
try:
    stock_data_info = fetch_stock_info(stock_ticker)
except:
    st.error("Error: Unable to fetch the stock data. Please try again later.")
    st.stop()


#####Title#####

# Add title to the app
st.markdown("# Stock Info Plus")

# Add a subtitle to the app
st.markdown("##### Enhancing Your Stock Market Insights")

#####Title End#####

#####Basic Information#####

# Add a heading
st.markdown("## Basic Information")

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame({"Issuer Name": [stock_data_info["Basic Information"]["longName"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Symbol": [stock_ticker]}),
    hide_index=True,
    width=500,
)

# Row 2
col1.dataframe(
    pd.DataFrame({"Currency": [stock_data_info["Basic Information"]["currency"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(pd.DataFrame({"Exchange": ["NSE"]}), hide_index=True, width=500)

#####Basic Information End#####

#####Market Data#####

# Add a heading
st.markdown("## Market Data")

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame({"Current Price": [stock_data_info["Market Data"]["currentPrice"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Previous Close": [stock_data_info["Market Data"]["previousClose"]]}),
    hide_index=True,
    width=500,
)

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Row 1
col1.dataframe(
    pd.DataFrame({"Open": [stock_data_info["Market Data"]["open"]]}),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame({"Day Low": [stock_data_info["Market Data"]["dayLow"]]}),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame({"High": [stock_data_info["Market Data"]["dayHigh"]]}),
    hide_index=True,
    width=300,
)

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame(
        {
            "Regular Market Previous Close": [
                stock_data_info["Market Data"]["regularMarketPreviousClose"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Regular Market Open": [stock_data_info["Market Data"]["regularMarketOpen"]]}
    ),
    hide_index=True,
    width=500,
)

# Row 2
col1.dataframe(
    pd.DataFrame(
        {
            "Regular Market Day Low": [
                stock_data_info["Market Data"]["regularMarketDayLow"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Regular Market Day High": [
                stock_data_info["Market Data"]["regularMarketDayHigh"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)


#####Market Data End#####

#####Volume and Shares#####

# Add a heading
st.markdown("## Volume")

# Create 3 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame({"Volume": [stock_data_info["Volume and Shares"]["volume"]]}),
    hide_index=True,
    width=500,
)
# Row 2
col2.dataframe(
    pd.DataFrame(
        {
            "Regular Market Volume": [
                stock_data_info["Volume and Shares"]["regularMarketVolume"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)


#####Volume and Shares End#####