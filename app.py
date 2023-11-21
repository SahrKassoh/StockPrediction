import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained model
model_path = "/Users/queensia/Documents/StockPrediction/StcokImages/Stock_Predictions_Model.h5"
model = load_model(model_path)

# Streamlit UI components
st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'AAPL')
start = '2013-11-12'
end = '2023-11-12'

# Download stock data
data = yf.download(stock, start, end)

# Ensure the index is set to datetime
data.index = pd.to_datetime(data.index)

# Calculate and trim moving averages
ma_50_days = data['Close'].rolling(50).mean().loc[start:end]
ma_100_days = data['Close'].rolling(100).mean().loc[start:end]
ma_200_days = data['Close'].rolling(200).mean().loc[start:end]

st.subheader('Stock Data')
st.write(data)

# Prepare training and test data
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Concatenate the past 100 days of training data with the test data
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

# Scale the test data
data_test_scale = scaler.fit_transform(data_test)

# 50 day moving average
fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Actual Price')
plt.title('Moving Average 50 Days vs Actual Price')
plt.legend(loc='upper left')
st.pyplot(fig1)


# 100 day moving average
fig2 = plt.figure(figsize=(10, 8))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(data.Close, 'g', label='Actual Price')
plt.title('Moving Average 100 Days vs Actual Price')
plt.legend(loc='upper left')
st.pyplot(fig2)


# 200 day moving average
fig3 = plt.figure(figsize=(10, 8))
plt.plot(ma_200_days, 'r', label='MA200')
plt.plot(data.Close, 'g', label='Actual Price')
plt.title('Moving Average 200 Days vs Actual Price')
plt.legend(loc='upper left')
st.pyplot(fig3)



# Generate sequences for test data
x_test = []
y_test = []

# Loop through the scaled test data to create sequences
for i in range(100, len(data_test_scale)):
    x_test.append(data_test_scale[i - 100:i, 0])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape the test data to match model input
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predict = model.predict(x_test)
scale = 1 / scaler.scale_
predict = predict * scale
y_test = y_test * scale

# Original price vs Predicted Price
fig4 = plt.figure(figsize=(10, 8))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y_test, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original Price vs Predicted Price')
plt.legend(loc='upper left')
st.pyplot(fig4)


