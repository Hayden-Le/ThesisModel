#!/usr/bin/env python3
"""
Client for Federated Learning on Stock Data.
Each client is assigned a company (Microsoft, Intel, or NVIDIA) and trains on its stock data.
Usage: python client.py [Company]
Default company: Microsoft
Ensure the corresponding CSV (e.g. "microsoft.csv") is available.
"""

import sys
import pandas as pd
import requests
import tensorflow as tf
import numpy as np

# Configuration
SEQUENCE_LENGTH = 20
# Uncomment the appropriate server URL:
# SERVER_URL = "http://100.98.121.3:5000"  # Laptop WSL
SERVER_URL = "http://100.107.145.118:5000"  # PC WSL

# Set company from command line argument; default is Microsoft.
if len(sys.argv) > 1:
    COMPANY = sys.argv[1]
else:
    COMPANY = "Microsoft"

def create_lstm_model():
    """Creates a simple LSTM model for stock prediction."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(SEQUENCE_LENGTH, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

def load_stock_data(company):
    """Loads stock data for the given company from a CSV file."""
    filename = f"{company.lower()}.csv"
    df = pd.read_csv(filename)
    # Assume the CSV has a 'Close' column with stock closing prices.
    return df['Close'].values

def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
    """Converts a time series into sequences and labels."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def serialize_weights(weights):
    """Converts model weights (list of NumPy arrays) into JSON-serializable lists."""
    return [w.tolist() for w in weights]

def deserialize_weights(serialized_weights):
    """Converts JSON-serialized weights back to a list of NumPy arrays."""
    return [np.array(w) for w in serialized_weights]

def get_global_model():
    """Fetches the current global model weights and version from the server."""
    response = requests.get(f"{SERVER_URL}/get_model")
    data = response.json()
    weights = deserialize_weights(data["weights"])
    version = data["version"]
    return weights, version

def send_update(updated_weights):
    """Sends the client's updated model weights to the server."""
    payload = {"weights": serialize_weights(updated_weights)}
    response = requests.post(f"{SERVER_URL}/post_update", json=payload)
    return response.json()

def wait_for_new_model(current_version):
    """Waits until the server updates the global model version and returns the new model."""
    params = {"version": current_version}
    response = requests.get(f"{SERVER_URL}/wait_for_update", params=params)
    data = response.json()
    new_weights = deserialize_weights(data["weights"])
    new_version = data["version"]
    return new_weights, new_version

def local_training(global_weights, dataset):
    """Loads the global model, trains on local data, and returns updated weights."""
    model = create_lstm_model()
    model.set_weights(global_weights)
    model.fit(dataset, epochs=1, verbose=1)
    return model.get_weights()

def main():
    print(f"Loading data for {COMPANY}...")
    data = load_stock_data(COMPANY)
    X, y = create_sequences(data, SEQUENCE_LENGTH)
    X = X.reshape(-1, SEQUENCE_LENGTH, 1)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(20)
    
    # Step 1: Get the current global model and version from the server
    global_weights, version = get_global_model()
    print(f"Initial global model version: {version}")
    
    # Step 2: Perform local training using the global model
    updated_weights = local_training(global_weights, dataset)
    
    # Step 3: Send the updated weights to the server
    send_response = send_update(updated_weights)
    print("Local update sent to server:", send_response)
    
    # Step 4: Wait for the server to update the global model
    print("Waiting for updated global model from server...")
    new_weights, new_version = wait_for_new_model(version)
    print(f"Received updated global model version: {new_version}")
    
    # Optionally update your local model with the new global weights
    local_model = create_lstm_model()
    local_model.set_weights(new_weights)
    print("Local model updated with the new global model.")

if __name__ == "__main__":
    main()
