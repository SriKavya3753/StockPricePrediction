import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load the pretrained LSTM model
model = load_model("lstm.h5")

# Load and preprocess the dataset
df = pd.read_csv('SBI_20Y_Stock_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

target = 'Close'
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[[target]])

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# Sequence length
seq_length = 30
X, y = create_sequences(scaled_data, seq_length)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Make predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Calculate MSE
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

# Forecast the next 60 days
current_input = X_test[-1]  # Use the last test sequence as the starting input
current_input = current_input.reshape(1, seq_length, 1)  # Reshape to (1, seq_length, 1)

predictions = []
for _ in range(60):
    next_prediction = model.predict(current_input, verbose=0)
    predictions.append(next_prediction[0, 0])  # Append the predicted value

    # Update `current_input` with the new prediction
    next_input = next_prediction.reshape(1, 1, 1)  # Shape: (1, 1, 1)
    current_input = np.append(current_input[:, 1:, :], next_input, axis=1)  # Slide the window

# Convert predictions back to the original scale
forecast_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Create dates for the forecast
forecast_dates = pd.date_range(start=df.index[-1], periods=61, freq='B')[1:]

# Convert predictions back to original scale
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Streamlit App
st.title("Stock Price Prediction and Forecasting")
st.write("This app predicts and forecasts stock prices using a pre-trained LSTM model.")

# Date range selection
st.sidebar.subheader("Select Date Range")
start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2003-02-17"), min_value=pd.Timestamp("2003-02-17"), max_value=pd.Timestamp("2023-02-15"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp("2023-02-15"), min_value=pd.Timestamp("2003-02-17"), max_value=pd.Timestamp("2023-02-15"))

# Ensure valid date range
if start_date > end_date:
    st.error("Error: Start date must be before end date.")
else:
    # Filter the data based on the selected date range
    filtered_data = df.loc[start_date:end_date]

    # Display company details
    st.subheader("Company Details")
    st.write("Showing High, Low, Open, Close details for the selected date range:")
    st.dataframe(filtered_data[['Open', 'High', 'Low', 'Close']])

    # Plot graph
    st.subheader("Stock Prices Between Selected Dates")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_data.index, filtered_data['Close'], label='Close Price', color='blue')
    ax.set_title('Stock Prices Between Selected Dates')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

# Display MSE metrics
st.subheader("Model Performance")
st.write(f"Train MSE: {train_mse}")
st.write(f"Test MSE: {test_mse}")

# Historical Prices Visualization
st.subheader("Historical Stock Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Close'], label='Close Prices', color='blue')
ax.set_title('Historical Stock Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.legend()
st.pyplot(fig)

# Predicted vs Actual and Forecast Visualization
st.subheader("Predicted vs Actual Prices and Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index[train_size + seq_length:], y_test_actual, label='Actual Prices', color='blue')
ax.plot(df.index[train_size + seq_length:], test_pred, label='Predicted Prices', color='orange', linestyle='--')
ax.plot(forecast_dates, forecast_values, label='Forecast Prices (60 Days)', color='green', linestyle='--')
ax.set_title('Predicted, Actual, and Forecast Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend()
st.pyplot(fig)

# Forecast Values
st.subheader("Forecasted Prices for the Next 60 Days")
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted Price': forecast_values.flatten()
})
st.dataframe(forecast_df)
