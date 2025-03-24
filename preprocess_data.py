#!/usr/bin/env python3
"""
preprocess_data.py

This script processes a single CSV file that may have extra header rows,
and then splits the data year by year, saving each year’s training and test
sets as separate .npy files. Each set is scaled using Min-Max normalization.

Usage:
    python preprocess_data.py \
        --input apple.csv \
        --start_year 2010 \
        --end_year 2023 \
        --sequence_length 20 \
        --test_ratio 0.2

For each year in [start_year, end_year], it will create:
    X_train_<year>.npy
    y_train_<year>.npy
    X_test_<year>.npy
    y_test_<year>.npy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import csv

def load_and_clean_data(filename):
    """
    Loads data from a CSV file that might contain extra header rows.
    - Skips the first 3 rows
    - Manually assigns column names
    - Parses the Date column, sorts by Date
    - Fills missing values
    """
    # Read the CSV, skipping first 3 lines, no header
    df = pd.read_csv(filename, skiprows=3, header=None)

    # Manually assign column names (adjust if your columns differ)
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # Parse Date and sort
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.sort_values("Date", inplace=True)

    # Fill missing values (forward then backward)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

def create_sequences(data, sequence_length):
    """
    Creates sequences of length 'sequence_length' and the subsequent target.
    Returns arrays X, y.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def process_year_data(df, year, sequence_length, test_ratio):
    """
    Filters the DataFrame for a single year, scales the 'Close' column,
    creates sequences, splits into train/test, and returns (X_train, y_train, X_test, y_test).
    """
    # Filter rows for this specific year
    df_year = df[df["Date"].dt.year == year].copy()
    if len(df_year) == 0:
        return None  # No data for this year

    # Scale the Close prices
    close_prices = df_year["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    # Create sequences
    X, y = create_sequences(scaled_prices, sequence_length)

    # Train/Test split
    split_index = int((1 - test_ratio) * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, y_train, X_test, y_test

def main():
    parser = argparse.ArgumentParser(description="Preprocess stock data year by year.")
    parser.add_argument("--input", type=str, default="apple.csv",
                        help="Input CSV file (default: apple.csv)")
    parser.add_argument("--start_year", type=int, default=2010,
                        help="First year to process (default: 2010)")
    parser.add_argument("--end_year", type=int, default=2023,
                        help="Last year to process (default: 2023)")
    parser.add_argument("--sequence_length", type=int, default=20,
                        help="Length of the input sequence (default: 20)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Proportion of data to use for testing (default: 0.2)")
    args = parser.parse_args()

    # Load and clean the data
    df = load_and_clean_data(args.input)
    print(f"Data loaded. Total records: {len(df)}")

    # Ensure 'Close' column exists
    if "Close" not in df.columns:
        raise ValueError("The CSV must contain a 'Close' column after re-labeling.")

    # Iterate through each year
    for year in range(args.start_year, args.end_year + 1):
        result = process_year_data(df, year, args.sequence_length, args.test_ratio)
        if result is None:
            print(f"No data found for year {year}. Skipping.")
            continue

        X_train, y_train, X_test, y_test = result
        print(f"Year {year}: Train size={len(X_train)}, Test size={len(X_test)}")

        # Save arrays if there's enough data
        if len(X_train) == 0:
            print(f"Warning: no training data for {year}.")
            continue

        # Save .npy files
        np.save(f"X_train_{year}.npy", X_train)
        np.save(f"y_train_{year}.npy", y_train)
        np.save(f"X_test_{year}.npy", X_test)
        np.save(f"y_test_{year}.npy", y_test)
        print(f"Saved X_train_{year}.npy, y_train_{year}.npy, X_test_{year}.npy, y_test_{year}.npy")

if __name__ == "__main__":
    main()
