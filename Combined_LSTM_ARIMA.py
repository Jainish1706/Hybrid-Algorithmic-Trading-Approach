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


# Folder containing ARIMA predictions
arima_folder = '/content/drive/MyDrive/Major projects_Gautam_Jan2024_June2024/Dataset_final/Forecasted_test_split'

# Folder containing LSTM predictions
lstm_folder = '/content/drive/MyDrive/Major projects_Gautam_Jan2024_June2024/Dataset_final/Lstm_prediction'

# Output folder for combined predictions
output_folder = '/content/drive/MyDrive/Major projects_Gautam_Jan2024_June2024/Dataset_final/final_comb copy'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# Function to extract base name before the first underscore
def extract_base_name(file):
    return os.path.splitext(os.path.basename(file))[0].split('_')[0]


# List all files in ARIMA and LSTM prediction folders
arima_files = [os.path.join(arima_folder, file) for file in os.listdir(arima_folder) if file.endswith('.csv')]
lstm_files = [os.path.join(lstm_folder, file) for file in os.listdir(lstm_folder) if file.endswith('.csv')]


# Dictionary to store combined predictions
combined_predictions = {}

# Combine predictions with the same base name
for arima_file in arima_files:
    arima_base_name = extract_base_name(arima_file)

    for lstm_file in lstm_files:
        lstm_base_name = extract_base_name(lstm_file)

        if arima_base_name == lstm_base_name:
            print(f"Combining predictions from {os.path.basename(arima_file)} and {os.path.basename(lstm_file)}...")

            # Load ARIMA and LSTM predictions
            arima_predictions = pd.read_csv(arima_file)
            lstm_predictions = pd.read_csv(lstm_file)

            # Check if the lengths of both dataframes are the same
            if len(arima_predictions) != len(lstm_predictions):
                print(f"Length mismatch between ARIMA and LSTM predictions for files: {arima_file}, {lstm_file}")
                continue  # Skip this pair of files

            # Check if the structure of both dataframes are the same
            if arima_predictions.columns.tolist() != lstm_predictions.columns.tolist():
                print(f"Column structure mismatch between ARIMA and LSTM predictions for files: {arima_file}, {lstm_file}")
                continue  # Skip this pair of files

            # Combine predictions assuming the same index or time range
            combined_predictions[arima_base_name] = pd.DataFrame({
                'Date': arima_predictions['Date'],  # Assuming date column exists in both
                'Combined_Forecasted': arima_predictions['Forecasted'] + lstm_predictions['Forecasted']
            })

# Save combined predictions to output folder
for base_name, predictions in combined_predictions.items():
    output_file = os.path.join(output_folder, f"combined_{base_name}.csv")
    predictions.to_csv(output_file, index=False)
