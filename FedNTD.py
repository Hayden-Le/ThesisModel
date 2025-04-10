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
NUM_CLIENTS = 3         # Number of expected client updates per round
client_updates = []     # To store received client updates
global_model_version = 0  # Global model version
sequence_length = 20    # Number of time steps per sample
d_model = 64            # Embedding dimension
lambda_ntd = 0.1        # Hyperparameter for distillation weight

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
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    return model

global_model = create_transformer_model()

### Data Preprocessing for Public (Test) Data ###
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data)-sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def load_public_data(start_date="2023-01-01", end_date="2024-01-01"):
    ticker = yf.Ticker("AAPL")
    df = ticker.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.sort_values("Date", inplace=True)
    close_prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_prices = scaler.fit_transform(close_prices)
    X_pub, y_pub = create_sequences(scaled_prices, sequence_length)
    return X_pub, y_pub, scaler

### Metrics and Logging ###
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
    X_pub, y_pub, _ = load_public_data()
    y_pred = global_model.predict(X_pub)
    rmse, mae, r2 = compute_metrics(y_pub, y_pred)
    print(f"Test metrics on public (2023) data: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    log_metrics(global_model_version, rmse, mae, r2)
    global_model.save("aggregated_model.h5")
    print("Aggregated model saved as aggregated_model.h5")

### Distillation Loss Helper ###
def distillation_loss(teacher_logits, student_logits, temperature=1.0):
    """
    Computes Kullback-Leibler divergence between teacher and student logits.
    In this context, logits are the outputs (or soft predictions).
    We use temperature scaling for softness.
    """
    # Scale logits
    teacher_logits_scaled = teacher_logits / temperature
    student_logits_scaled = student_logits / temperature
    # Compute soft targets using softmax
    teacher_soft = tf.nn.softmax(teacher_logits_scaled, axis=1)
    student_soft = tf.nn.softmax(student_logits_scaled, axis=1)
    # Compute KL divergence
    kl = tf.keras.losses.KLDivergence()(teacher_soft, student_soft)
    return kl

### Server Endpoints ###
@app.route("/get_model", methods=["GET"])
def get_model():
    weights = global_model.get_weights()
    serialized = [w.tolist() for w in weights]
    return jsonify({"weights": serialized, "version": global_model_version})

@app.route("/post_update", methods=["POST"])
def post_update():
    """
    Receives raw client updates and aggregates them using FedNTD.
    Process:
      1. Compute FedAvg to get w_avg.
      2. Use a public dataset to obtain teacher signals from the current global model (w_global).
      3. Initialize a temporary student model with w_avg.
      4. Train the student model on the public dataset with a distillation loss that minimizes
         the difference between its predictions and the teacher’s (global model’s) soft labels.
      5. The updated student model becomes the new global model.
    """
    global client_updates, global_model, global_model_version, lambda_ntd
    data = request.get_json()
    update_list = data.get("weights", [])
    client_update = [np.array(w) for w in update_list]
    client_updates.append(client_update)
    print(f"Received update #{len(client_updates)} from a client.")
    
    if len(client_updates) == NUM_CLIENTS:
        print("All client updates received. Aggregating with FedNTD...")
        # 1. Compute simple average: w_avg
        aggregated_weights = []
        for weight_tuple in zip(*client_updates):
            aggregated_weights.append(np.mean(np.array(weight_tuple), axis=0))
        
        # 2. Get current global weights (teacher: w_global)
        w_global = global_model.get_weights()
        
        # 3. Initialize a temporary student model with w_avg
        student_model = create_transformer_model()
        student_model.set_weights(aggregated_weights)
        
        # 4. Use a public dataset (e.g., public data from 2023) for distillation
        X_pub, y_pub, _ = load_public_data(start_date="2023-01-01", end_date="2024-01-01")
        # Obtain teacher soft labels from the current global model
        teacher_preds = global_model.predict(X_pub)
        # Distill: Train the student model for a few epochs using teacher_preds as soft targets
        # Here we use a simple MSE loss between the student's prediction and the teacher's prediction,
        # scaled by lambda_ntd.
        student_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        student_model.fit(X_pub, teacher_preds, epochs=1, verbose=0)
        
        # 5. Set the updated student weights as the new global model weights.
        new_weights = student_model.get_weights()
        global_model.set_weights(new_weights)
        client_updates = []
        global_model_version += 1
        print(f"Global model updated. New version: {global_model_version}")
        test_global_model()
        
    return jsonify({"status": "update received"})

@app.route("/wait_for_update", methods=["GET"])
def wait_for_update():
    client_version = int(request.args.get("version", 0))
    timeout = 30
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
