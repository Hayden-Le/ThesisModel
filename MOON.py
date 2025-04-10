#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import time, os, csv
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

### Configuration ###
NUM_CLIENTS = 3         # Expected number of client updates per round
client_updates = []     # List to store received client updates
global_model_version = 0  # Global model version (increments on update)
sequence_length = 20    # Number of time steps per sample
d_model = 64            # Embedding dimension of the model
lambda_moon = 0.1       # MOON hyperparameter (contrastive loss weight)

### Model Definition ###
def create_transformer_model():
    inputs = tf.keras.layers.Input(shape=(sequence_length, 1))
    x = tf.keras.layers.Dense(d_model)(inputs)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model//4)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ff_output = tf.keras.layers.Dense(d_model, activation='relu')(x)
    ff_output = tf.keras.layers.Dense(d_model)(ff_output)
    x = tf.keras.layers.Add()([x, ff_output])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    # GlobalAveragePooling1D will serve as our embedding layer.
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    return model

global_model = create_transformer_model()

### Embedding Model Helper ###
def create_embedding_model(model):
    """
    Given a model, returns a new model whose output is the penultimate layer.
    We assume the last two layers are [GlobalAveragePooling1D, Dense].
    """
    return tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

### Data Preprocessing for Global Test Set ###
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def load_test_data(start_date="2023-01-01", end_date="2024-01-01"):
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.sort_values("Date", inplace=True)
    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_prices = scaler.fit_transform(close_prices)
    X_test, y_test = create_sequences(scaled_prices, sequence_length)
    return X_test, y_test, scaler

### Metrics Calculation and Logging ###
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
    exists = os.path.isfile(file_name)
    with open(file_name, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not exists:
            writer.writerow(header)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        writer.writerow([version, rmse, mae, r2, timestamp])
    print(f"Logged metrics: version={version}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

def test_global_model():
    X_test, y_test, _ = load_test_data()
    y_pred = global_model.predict(X_test)
    rmse, mae, r2 = compute_metrics(y_test, y_pred)
    print(f"Test metrics on global (2023) data: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    log_metrics(global_model_version, rmse, mae, r2)
    global_model.save("aggregated_model.h5")
    print("Aggregated model saved as aggregated_model.h5")

### Server Endpoints ###
@app.route("/get_model", methods=["GET"])
def get_model():
    weights = global_model.get_weights()
    serialized = [w.tolist() for w in weights]
    return jsonify({"weights": serialized, "version": global_model_version})

@app.route("/post_update", methods=["POST"])
def post_update():
    """
    Receives raw model updates from clients.
    Implements server-side MOON by:
    1. Computing the simple average (FedAvg) to obtain w_avg.
    2. Creating an embedding model for both the current global model and a model built from w_avg.
    3. Using a fixed test set (2023 data) to compute average cosine similarity s between embeddings:
          s = mean_{sample}(cosine_similarity(h_global, h_avg))
    4. Defining a correction factor:
          δ = lambda_moon * (1 - s)
    5. Updating global weights as:
          w_new = w_avg - δ * (w_avg - w_global)
    """
    global client_updates, global_model, global_model_version, lambda_moon
    data = request.get_json()
    update_list = data.get("weights", [])
    client_update = [np.array(w) for w in update_list]
    client_updates.append(client_update)
    print(f"Received update #{len(client_updates)} from a client.")
    
    if len(client_updates) == NUM_CLIENTS:
        print("All client updates received. Aggregating with server-side MOON...")
        # 1. Compute simple average of client updates: w_avg
        aggregated_weights = []
        for weights_tuple in zip(*client_updates):
            aggregated_weights.append(np.mean(np.array(weights_tuple), axis=0))
        
        # 2. Get current global weights (w_global)
        w_global = global_model.get_weights()
        
        # 3. Build a temporary model from w_avg and create embedding models from both
        temp_model = create_transformer_model()
        temp_model.set_weights(aggregated_weights)
        global_embedding_model = create_embedding_model(global_model)
        temp_embedding_model = create_embedding_model(temp_model)
        
        # 4. Use the reserved test data to compute embeddings and cosine similarity
        X_test, _, _ = load_test_data()
        h_global = global_embedding_model.predict(X_test)
        h_avg = temp_embedding_model.predict(X_test)
        # Compute cosine similarity for each sample (vectorized)
        eps = 1e-8
        dot_product = np.sum(h_global * h_avg, axis=1)
        norm_global = np.linalg.norm(h_global, axis=1)
        norm_avg = np.linalg.norm(h_avg, axis=1)
        cos_sim = dot_product / (norm_global * norm_avg + eps)
        s = np.mean(cos_sim)
        print(f"Average cosine similarity between global and aggregated embeddings: {s:.4f}")
        
        # 5. Define correction factor δ
        delta = lambda_moon * (1 - s)
        print(f"Correction factor δ = {delta:.4f}")
        
        # 6. Compute new global weights: w_new = w_avg - δ*(w_avg - w_global)
        new_global_weights = []
        for w_avg_tensor, w_global_tensor in zip(aggregated_weights, w_global):
            new_w = w_avg_tensor - delta * (w_avg_tensor - w_global_tensor)
            new_global_weights.append(new_w)
        
        global_model.set_weights(new_global_weights)
        client_updates = []
        global_model_version += 1
        print(f"Global model updated. New version: {global_model_version}")
        test_global_model()
    
    return jsonify({"status": "update received"})

@app.route("/wait_for_update", methods=["GET"])
def wait_for_update():
    client_version = int(request.args.get("version", 0))
    timeout = 30  # seconds maximum to wait
    poll_interval = 1
    waited = 0
    while waited < timeout:
        if global_model_version > client_version:
            weights = global_model.get_weights()
            serialized = [w.tolist() for w in weights]
            return jsonify({"weights": serialized, "version": global_model_version})
        time.sleep(poll_interval)
        waited += poll_interval
    weights = global_model.get_weights()
    serialized = [w.tolist() for w in weights]
    return jsonify({"weights": serialized, "version": global_model_version, "timeout": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
