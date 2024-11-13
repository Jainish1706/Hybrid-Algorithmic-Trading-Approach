import talib
import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from mplfinance.original_flavor import candlestick_ohlc
import talib
import shutil
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from scipy.optimize import root
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def xnpv(rate, cashflows, dates):
    """
    Calculate the net present value of a series of cashflows at irregular intervals.

    Parameters:
    - rate: The discount rate.
    - cashflows: List of cashflows (positive and negative).
    - dates: List of corresponding dates for each cashflow.

    Returns:
    - XNPV value.
    """
    try:
        return np.nansum([cf / (1 + rate)**((date - dates[0]).days / 365.0) for cf, date in zip(cashflows, dates)])
    except (ZeroDivisionError, RuntimeWarning):
        return np.nan  # Return NaN for any issues



#LSTM
def train_and_predict(file_path, output_folder):
    # Load the dataset
    dataset = pd.read_csv(file_path)

    # Extracting date information
    dates = dataset['Date']

    # Assuming the dataset has a column named 'Value' for the time series data
    training_data = dataset['Residuals'].values.reshape(-1, 1)

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_data_scaled = sc.fit_transform(training_data)

    # Splitting the data into training and testing sets
    train_size = int(len(training_data_scaled) * 0.8)  # 80% for training, 20% for testing
    train_data = training_data_scaled[:train_size]
    test_data = training_data_scaled[train_size:]

    # Prepare training data
    X_train = []
    y_train = []
    for i in range(7, len(train_data)):
        X_train.append(train_data[i-7:i, 0])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Define the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=200, batch_size=32)

    # Prepare test data
    X_test = []
    y_test = []
    for i in range(7, len(test_data)):
        X_test.append(test_data[i-7:i, 0])
        y_test.append(test_data[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions
    predicted_stock_price = model.predict(X_test)

    # Inverse transform predictions
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Save the forecasted values to a CSV file with date column
    output_file_path = os.path.join(output_folder, os.path.basename(file_path)[:-4] + '_forecast.csv')
    pd.DataFrame({'Date': dates[-len(predicted_stock_price):], 'Forecasted': predicted_stock_price.flatten()}).to_csv(output_file_path, index=False)

    # Plot the graph for test data vs predicted data
    plt.plot(dates[-len(y_test):], sc.inverse_transform(test_data[-len(y_test):]), color='red', label='Real Data')
    plt.plot(dates[-len(y_test):], predicted_stock_price, color='blue', label='Predicted Data')
    plt.title('Test Data vs Predicted Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Directory containing the dataset files
folder_path = '/content/drive/MyDrive/Major projects_Gautam_Jan2024_June2024/Dataset_final/Residulas_test'

# Output folder to store forecasted values
output_folder = '/content/drive/MyDrive/Major projects_Gautam_Jan2024_June2024/Dataset_final/Lstm_prediction'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the folder
file_names = os.listdir(folder_path)

# Train and predict for each file
for file_name in file_names:
    if file_name.endswith('.csv'):  # Check if the file is a CSV file
        file_path = os.path.join(folder_path, file_name)
        train_and_predict(file_path, output_folder)