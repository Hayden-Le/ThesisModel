#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import time, csv, os
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# --- Configuration ---
NUM_CLIENTS = 3           # Total number of client updates expected per round
client_updates = []       # To store client weight updates temporarily
global_model_version = 0  # Global version counter
sequence_length = 20      # Length of input sequence for the model
d_model = 64              # Embedding dimension for the Transformer model

# --- Model Definition ---
def create_transformer_model():
    """Creates a Transformer-based model (LLM-style) for time-series forecasting."""
    inputs = tf.keras.layers.Input(shape=(sequence_length, 1))
    # Project inputs to d_model dimensions
    x = tf.keras.layers.Dense(d_model)(inputs)
    # Self-attention layer
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model//4)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    # Feed-forward network
    ff_output = tf.keras.layers.Dense(d_model, activation='relu')(x)
    ff_output = tf.keras.layers.Dense(d_model)(ff_output)
    x = tf.keras.layers.Add()([x, ff_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    # Global pooling and final regression output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    return model

global_model = create_transformer_model()

# --- Data Download and Preprocessing for Global Test Set ---
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def load_apple_data_for_test(start_date="2023-01-01", end_date="2024-01-01"):
    """
    Downloads Apple stock data from yfinance for the test set.
    This data (from 2023) is reserved solely for evaluating the aggregated model.
    """
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start=start_date, end=end_date)
    if df.empty:
        raise ValueError("No test data fetched from yfinance. Check the date range.")
    df.reset_index(inplace=True)
    df.sort_values("Date", inplace=True)
    close_prices = df["Close"].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    
    X_test, y_test = create_sequences(scaled_prices, sequence_length)
    return X_test, y_test, scaler

# --- Metrics Computation and Logging ---
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
    return rmse, mae, r2

def log_metrics(version, rmse, mae, r2):
    file_name = "metrics.csv"
    header = ["version", "rmse", "mae", "r2", "timestamp"]
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        writer.writerow([version, rmse, mae, r2, timestamp])
    print(f"Logged metrics: version={version}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

def test_global_model():
    """Evaluates the global model on the reserved 2023 test set, logs metrics, and exports the model."""
    X_test, y_test, _ = load_apple_data_for_test()
    y_pred = global_model.predict(X_test)
    rmse, mae, r2 = compute_metrics(y_test, y_pred)
    print(f"Test metrics on global (2023) data: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    log_metrics(global_model_version, rmse, mae, r2)
    global_model.save("aggregated_model.h5")
    print("Aggregated model exported to aggregated_model.h5")

# --- Flask Endpoints ---
@app.route('/get_model', methods=['GET'])
def get_model():
    """Clients fetch the current global model weights and version."""
    weights = global_model.get_weights()
    serialized_weights = [w.tolist() for w in weights]
    return jsonify({
        "weights": serialized_weights,
        "version": global_model_version
    })

@app.route('/post_update', methods=['POST'])
def post_update():
    """
    Clients send their local model updates here.
    Once all client updates are received, the server aggregates them,
    updates the global model, tests on the reserved 2023 data, and logs metrics.
    """
    global client_updates, global_model, global_model_version
    data = request.get_json()
    update = data.get("weights", [])
    client_update = [np.array(w) for w in update]
    client_updates.append(client_update)
    print(f"Received update #{len(client_updates)} from a client.")
    
    if len(client_updates) == NUM_CLIENTS:
        print("All client updates received. Aggregating...")
        # Basic FedAvg: average the client updates
        aggregated_weights = []
        for weights_tuple in zip(*client_updates):
            aggregated_weights.append(np.mean(np.array(weights_tuple), axis=0))
        global_model.set_weights(aggregated_weights)
        client_updates = []  # reset for next round
        global_model_version += 1
        print(f"Global model updated. New version: {global_model_version}")
        test_global_model()
    
    return jsonify({"status": "update received"})

@app.route('/wait_for_update', methods=['GET'])
def wait_for_update():
    """Clients poll this endpoint until a new global model version is available."""
    client_version = int(request.args.get('version', 0))
    timeout = 30
    poll_interval = 1
    waited = 0
    while waited < timeout:
        if global_model_version > client_version:
            weights = global_model.get_weights()
            serialized_weights = [w.tolist() for w in weights]
            return jsonify({
                "weights": serialized_weights,
                "version": global_model_version
            })
        time.sleep(poll_interval)
        waited += poll_interval
    weights = global_model.get_weights()
    serialized_weights = [w.tolist() for w in weights]
    return jsonify({
        "weights": serialized_weights,
        "version": global_model_version,
        "timeout": True
    })

if __name__ == '__main__':
    # Run the server on all interfaces so it's accessible over Tailscale
    app.run(host="0.0.0.0", port=5000)
