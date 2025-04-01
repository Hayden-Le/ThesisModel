#!/usr/bin/env python3
import argparse
import requests
import tensorflow as tf
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
sequence_length = 20
d_model = 64
server_url = "http://<server-tailscale-ip>:5000"  # Replace with your server's Tailscale IP
mu = 0.1  # Regularization coefficient for FedDyn

# --- Model Definition ---
def create_transformer_model():
    """Creates a Transformer-based model (LLM-style) for time-series forecasting."""
    inputs = tf.keras.layers.Input(shape=(sequence_length, 1))
    x = tf.keras.layers.Dense(d_model)(inputs)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model//4)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ff_output = tf.keras.layers.Dense(d_model, activation='relu')(x)
    ff_output = tf.keras.layers.Dense(d_model)(ff_output)
    x = tf.keras.layers.Add()([x, ff_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    return model

# --- Data Download and Preprocessing for a Specific Year ---
def download_apple_data_for_year(year, start_date=None, end_date=None):
    """
    Downloads Apple stock data via yfinance for a given year.
    If start_date and end_date are not provided, they default to the full year.
    """
    if start_date is None:
        start_date = f"{year}-01-01"
    if end_date is None:
        end_date = f"{year+1}-01-01"
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.sort_values("Date", inplace=True)
    if df.empty:
        raise ValueError(f"No data found for year {year}")
    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    return scaled_prices, scaler

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def partition_data_cyclically(X, y, client_id, num_clients, start_offset=11):
    """
    Partitions data cyclically starting at a given offset.
    With start_offset=11 and num_clients=3:
      - Client 0 gets indices: 11, 14, 17, 20, …
      - Client 1 gets indices: 12, 15, 18, 21, …
      - Client 2 gets indices: 13, 16, 19, 22, …
    """
    return X[start_offset + client_id::num_clients], y[start_offset + client_id::num_clients]

# --- Serialization Functions ---
def serialize_weights(weights):
    return [w.tolist() for w in weights]

def deserialize_weights(serialized_weights):
    return [np.array(w) for w in serialized_weights]

# --- Server Communication Functions ---
def get_global_model():
    response = requests.get(f"{server_url}/get_model")
    data = response.json()
    weights = deserialize_weights(data["weights"])
    version = data["version"]
    return weights, version

def send_update(updated_weights):
    payload = {"weights": serialize_weights(updated_weights)}
    response = requests.post(f"{server_url}/post_update", json=payload)
    return response.json()

def wait_for_new_model(current_version):
    params = {"version": current_version}
    response = requests.get(f"{server_url}/wait_for_update", params=params)
    data = response.json()
    new_weights = deserialize_weights(data["weights"])
    new_version = data["version"]
    return new_weights, new_version

# --- FedDyn Functions ---
def initialize_d(global_weights):
    """Initializes the dynamic correction vector d_local as zeros in float32."""
    return [tf.zeros_like(tf.cast(w, tf.float32)) for w in global_weights]

def local_training_feddyn(global_weights, dataset, mu, d_local):
    """
    Performs one epoch of local training using FedDyn.
    Loss = MSE(y_true, y_pred) + (mu/2)*sum(||w - w_global||^2) + sum(<d, (w - w_global)>)
    Then, update d_local as: d <- d + mu * (w - w_global)
    """
    model = create_transformer_model()
    model.set_weights(global_weights)
    optimizer = tf.keras.optimizers.Adam()
    mse = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    for X_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            base_loss = mse(y_batch, y_pred)
            prox_loss = 0.0
            dynamic_loss = 0.0
            for w, w_global, d in zip(model.trainable_weights, global_weights, d_local):
                # Cast to float32 to ensure consistency
                diff = tf.cast(w, tf.float32) - tf.cast(w_global, tf.float32)
                prox_loss += tf.reduce_sum(tf.square(diff))
                dynamic_loss += tf.reduce_sum(tf.cast(d, tf.float32) * diff)
            loss = base_loss + (mu / 2.0) * prox_loss + dynamic_loss
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_loss(loss)
        # Update d_local: d <- d + mu * (w - w_global)
        new_d_local = []
        for w, w_global, d in zip(model.trainable_weights, global_weights, d_local):
            diff = tf.cast(w, tf.float32) - tf.cast(w_global, tf.float32)
            new_d = tf.cast(d, tf.float32) + mu * diff
            new_d_local.append(new_d)
        d_local = new_d_local
    print("FedDyn Training Loss:", train_loss.result().numpy())
    return model.get_weights(), d_local

# --- Main Client Routine ---
def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client with FedDyn for Apple Data")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID (0-indexed)")
    parser.add_argument("--num_clients", type=int, required=True, help="Total number of clients")
    parser.add_argument("--start_year", type=int, default=2010, help="Start year for training (default: 2010)")
    parser.add_argument("--end_year", type=int, default=2022, help="End year for training (default: 2022)")
    args = parser.parse_args()
    
    print(f"Client {args.client_id} of {args.num_clients} processing training years {args.start_year} to {args.end_year}")
    
    # Get initial global model from server
    global_weights, version = get_global_model()
    print(f"Initial global model version: {version}")
    
    # Initialize the dynamic correction vector for FedDyn
    d_local = initialize_d(global_weights)
    
    # Process each training year (2010 to 2022)
    for year in range(args.start_year, args.end_year + 1):
        print(f"\nProcessing year {year}...")
        # Download full-year data for the year
        scaled_prices, _ = download_apple_data_for_year(year)
        # Create sequences from the data
        X, y = create_sequences(scaled_prices, sequence_length)
        print(f"Full-year data shape: X={X.shape}, y={y.shape}")
        
        # Partition data cyclically among clients starting at offset 11
        X_local, y_local = partition_data_cyclically(X, y, args.client_id, args.num_clients, start_offset=11)
        print(f"Client {args.client_id} data shape for year {year}: X={X_local.shape}, y={y_local.shape}")
        
        # Perform a local chronological train/validation split (80/20 split)
        split_index = int(0.8 * len(y_local))
        if split_index == 0:
            print(f"Not enough data for training in year {year}. Skipping.")
            continue
        X_train, X_val = X_local[:split_index], X_local[split_index:]
        y_train, y_val = y_local[:split_index], y_local[split_index:]
        
        # Create TensorFlow dataset for training
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(20)
        
        print(f"Local training for year {year} on {len(y_train)} samples using FedDyn...")
        updated_weights, d_local = local_training_feddyn(global_weights, dataset, mu, d_local)
        
        # Send the updated weights to the server
        response = send_update(updated_weights)
        print(f"Update sent for year {year}: {response}")
        
        # Wait for aggregated global model update from the server
        print("Waiting for aggregated global model update...")
        new_weights, new_version = wait_for_new_model(version)
        print(f"Received new global model version: {new_version}")
        global_weights, version = new_weights, new_version
    
    print("\nTraining complete for all years. Client exiting.")

if __name__ == "__main__":
    main()
