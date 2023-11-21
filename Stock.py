import numpy as np  # Used for numerical operations
import pandas as pd  # Used for data manipulation and analysis
import matplotlib.pyplot as plt  # Used for creating static, interactive, and animated visualizations
import yfinance as yf  # Used for downloading market data from Yahoo Finance
from sklearn.preprocessing import MinMaxScaler  # Used for scaling data
from keras.src.engine.sequential import Sequential  # Allows creating a linear stack of neural network layers
from keras.layers import Dense, Dropout, LSTM  # Dense layers are typically used for output layers

# Dropout is used to prevent overfitting, and LSTM is used for time-series predictions


# Define the start and end dates for the historical data
start = '2013-11-12'
end = '2023-11-12'
# Define the ticker symbol for the asset we want to analyze (SPY ETF)
stock = 'AAPL'

# Download the historical data for the specified stock between the start and end dates
data = yf.download(stock, start, end)
# Reset the index of the DataFrame to turn the Date index into a column
data.reset_index(inplace=True)
# Calculate the 100-day moving average of the closing prices
ma_100_days = data.Close.rolling(100).mean()
# Calculate the 200-day moving average of the closing prices
ma_200_days = data.Close.rolling(200).mean()

# Create a plot with a specified figure size
plt.figure(figsize=(8, 6))
# Plot the 100-day moving average in red
plt.plot(ma_100_days, 'r', label='100-day MA')
# Plot the 200-day moving average in blue
plt.plot(ma_200_days, 'b', label='200-day MA')
# Plot the closing prices in green
plt.plot(data.Close, 'g', label='Closing Prices')
# Add a legend to the plot
plt.legend()
# Display the plot
plt.show()

# Drop rows with missing values that may have been created when calculating the moving averages
data.dropna(inplace=True)
# Split the data into a training set (80% of the data)
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
# Split the data into a test set (remaining 20% of the data)
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Initialize a MinMaxScaler that will scale features to a range between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
# Scale the training data to be within the range of 0 and 1
data_train_scale = scaler.fit_transform(data_train)

# our 2 data arrays for the for loop
x_train = []
y_train = []
# Loop through the data of the first 100 days to calculate or predict the 101th day
for i in range(100, data_train_scale.shape[0]):
    x_train.append(data_train_scale[i - 100:i])
    y_train.append(data_train_scale[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Initialize the Sequential model - a linear stack of neural network layers
model = Sequential()

# Add the first LSTM layer with 50 units. The 'relu' activation function is used for non-linearity.
# 'return_sequences' is True, as we will add more LSTM layers after this one.
# The 'input_shape' is required to let the model know what input it should expect.
model.add(LSTM(units=50, activation='relu', return_sequences=True,
               input_shape=(x_train.shape[1], 1)))

# Add a Dropout layer after the LSTM layer to prevent overfitting, dropping 20% of the units
model.add(Dropout(0.2))

# Add additional LSTM and Dropout layers, increasing the complexity of the model
# to capture more intricate patterns in the data. Return sequences is kept true to allow
# the stacking of LSTM layers.
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(Dropout(0.2))  # 20% of the units are dropped

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))  # 30% of the units are dropped

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))  # 40% of the units are dropped

# The last LSTM layer does not return sequences, as this is the final LSTM output that will be
# fed to the dense output layer.
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))  # 50% of the units are dropped

# Add a Dense output layer with one unit. This unit will predict the scaled closing price.
model.add(Dense(units=1))

# Compile the model using the 'adam' optimizer and mean squared error loss function,
# which is a common choice for regression problems.
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model to the data. 'x' is the input data, 'y' is the target data.
# The model will train for 50 epochs, which means it will go through the data 50 times.
# The batch size of 32 means that 32 samples will be used to update the model per gradient descent.

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
past_100_days = data_train.tail(100)

# Prepare the test data for training
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Generate sequences for test data
x_test = []
y_test = []

for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i - 100:i])
    y_test.append(data_test_scale[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict future stock prices

# Predict future stock prices
y_predict = model.predict(x_test)
y_predict = scaler.inverse_transform(y_predict)  # Correctly inverse scaling predictions
y_test_reshaped = y_test.reshape(-1, 1)  # Reshaping y_test to have the same shape as y_predict for inverse transform
y = scaler.inverse_transform(y_test_reshaped)  # Correctly inverse scaling actual values

# Plot the predicted vs actual prices
plt.figure(figsize=(10, 8))
plt.plot(y_predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

# Save the figure
plt.savefig('Predicted_vs_Original_Price.png', format='png')

# Now display the plot window (this will block the code execution until the window is closed)
plt.show()

model.save('Stock_Predictions_Model.h5')
