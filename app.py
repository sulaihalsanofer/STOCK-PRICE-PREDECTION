# apple_lstm_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

st.set_page_config(page_title="Apple Stock Prediction", layout="wide")
st.title("Apple (AAPL) Stock Price Prediction using LSTM")

# -------------------- DATA LOADING --------------------
st.sidebar.header("Data Parameters")

# Fixed start and dynamic end date
start_date = datetime(2024, 1, 1)
end_date = datetime.now()

st.sidebar.write(f"**Start Date:** {start_date.strftime('%Y-%m-%d')}")
st.sidebar.write(f"**End Date:** {end_date.strftime('%Y-%m-%d')}")

st.write("### Loading Data...")
df = yf.download("AAPL", start=start_date, end=end_date)
st.write(f"Data from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**:")
st.dataframe(df.tail())

# -------------------- PREPROCESSING --------------------
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]  # overlap for sequence

# Create sequences
def create_sequences(dataset, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(dataset)):
        X.append(dataset[i - seq_length:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# -------------------- MODEL BUILDING --------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping
early_stop = EarlyStopping(monitor='loss', patience=5)

# -------------------- TRAIN MODEL --------------------
with st.spinner("Training LSTM model... (this might take a minute)"):
    model.fit(X_train, y_train, batch_size=32, epochs=20, callbacks=[early_stop])

# -------------------- PREDICT --------------------
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------- VISUALIZE --------------------
st.subheader("📊 Actual vs Predicted Stock Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_prices, label="Actual Price", color='blue')
ax.plot(predicted_prices, label="Predicted Price", color='red')
ax.set_title("AAPL Stock Price Prediction (Test Data)")
ax.set_xlabel("Days")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# -------------------- PREDICT NEXT 5 DAYS --------------------
st.subheader("🔮 Predict Next 5 Days")

last_60_days = scaled_data[-60:]
future_predictions = []

for _ in range(5):
    X_future = np.reshape(last_60_days, (1, 60, 1))
    pred_price = model.predict(X_future)
    future_predictions.append(pred_price[0, 0])
    last_60_days = np.append(last_60_days[1:], pred_price)
    last_60_days = np.reshape(last_60_days, (60, 1))

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(5)]
future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_predictions.flatten()
})

st.write("### 📅 Next 5-Day Price Forecast")
st.dataframe(future_df)

# -------------------- FINAL CHART --------------------
st.subheader("📈 Historical + Future Predictions")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(df.index[-100:], df['Close'].values[-100:], label="Actual Price (Last 100 Days)", color='blue')
ax2.plot(future_df["Date"], future_df["Predicted Price"], label="Predicted Future Price", color='orange', marker='o')
ax2.set_title("AAPL: Actual vs 5-Day Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)

st.success("✅ Prediction complete!")