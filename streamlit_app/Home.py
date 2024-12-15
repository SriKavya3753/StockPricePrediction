import streamlit as st

st.set_page_config(
    page_title="Stock Prediction App"
)

st.markdown(
    """# 📈 **Stock Prediction App**
### **Predicting Stocks with ML**

**It is an ML-powered stock price prediction app built with Python and Streamlit. It utilizes machine learning model to forecast stock prices and help investors make data-driven decisions.**

## 🏗️ **How It's Built**

This is built with these core frameworks and modules:

- **Streamlit** - To create the web app UI and interactivity 
- **YFinance** - To fetch financial data from Yahoo Finance API
- **StatsModels** - To build the AR time series forecasting model
- **Plotly** - To create interactive financial charts

The app workflow is:

1. User selects a stock ticker
2. Historical data is fetched with YFinance
3. AR model is trained on the data 
4. Model makes multi-day price forecasts
5. Results are plotted with Plotly

## 🎯 **Key Features**

- **Real-time data** - Fetch latest prices and fundamentals 
- **Financial charts** - Interactive historical and forecast charts
- **AR forecasting** - Make statistically robust predictions
- **Backtesting** - Evaluate model performance
- **Responsive design** - Works on all devices """
)