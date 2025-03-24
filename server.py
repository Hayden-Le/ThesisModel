#!/usr/bin/env python3
"""
server.py

Multi-year federated learning server. Each "round" corresponds to one year of data.
The server:
  1. Waits for NUM_REMOTE_CLIENTS updates for the current year.
  2. Trains locally on Apple data for that year.
  3. Aggregates all updates, updates the global model, and increments the year.
  4. Computes RMSE on Apple’s test data for that year, exports the model.
  5. Repeats until no more years remain.

Requires:
  - X_train_<year>.npy, y_train_<year>.npy, X_test_<year>.npy, y_test_<year>.npy
    for Apple (the server) in the same directory.
"""

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import time
import logging
import math

app = Flask(__name__)

# ----- Configuration -----
NUM_REMOTE_CLIENTS = 3
YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 
         2017, 2018, 2019, 2020, 2021, 2022, 2023]
SEQUENCE_LENGTH = 20
BATCH_SIZE = 20

# Logging
logging.basicConfig(level=logging.INFO)

# ----- Global State -----
client_updates = []
global_model_version = 0
current_year_index = 0
training_in_progress = False  # Prevent double-training if updates arrive simultaneously

def create_lstm_model():
    """Creates a simple LSTM model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(SEQUENCE_LENGTH, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

global_model = create_lstm_model()

def server_local_training_for_year(global_weights, year):
    """
    Loads Apple’s training data for the given year, performs local training,
    and returns the updated weights.
    """
    model = create_lstm_model()
    model.set_weights(global_weights)

    X_train = np.load(f"X_train_{year}.npy")
    y_train = np.load(f"y_train_{year}.npy")

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    logging.info(f"Server local training on Apple data for year {year}...")
    model.fit(dataset, epochs=1, verbose=1)

    return model.get_weights()

def aggregate_weights(updates):
    """Averages each layer’s weights across all updates."""
    new_weights = []
    for weights_tuple in zip(*updates):
        new_weights.append(np.mean(np.array(weights_tuple), axis=0))
    return new_weights

def compute_rmse_on_year_data(model, year):
    """
    Loads Apple’s test data for the given year, evaluates MSE, returns RMSE.
    """
    X_test = np.load(f"data/X_test_{year}.npy")
    y_test = np.load(f"data/y_test_{year}.npy")
    mse = model.evaluate(X_test, y_test, verbose=0)
    rmse = math.sqrt(mse)
    return rmse

@app.route('/get_model', methods=['GET'])
def get_model():
    """Clients call this to get the current global model weights + version."""
    weights = global_model.get_weights()
    serialized_weights = [w.tolist() for w in weights]
    return jsonify({
        "weights": serialized_weights,
        "version": global_model_version
    })

@app.route('/post_update', methods=['POST'])
def post_update():
    """
    Clients post their local model update here.
    Once NUM_REMOTE_CLIENTS updates have arrived, the server trains locally,
    aggregates, increments the version, and moves on to the next year.
    """
    global client_updates, global_model_version, training_in_progress, current_year_index

    if current_year_index >= len(YEARS):
        # No more years left to train
        return jsonify({"status": "no more years"}), 200

    data = request.get_json()
    update = data.get("weights", [])
    client_update = [np.array(w) for w in update]
    client_updates.append(client_update)
    logging.info(f"Received remote update #{len(client_updates)} from a client (year {YEARS[current_year_index]}).")

    # Once all remote updates arrive, do local training + aggregation
    if len(client_updates) == NUM_REMOTE_CLIENTS and not training_in_progress:
        training_in_progress = True

        year = YEARS[current_year_index]
        server_update = server_local_training_for_year(global_model.get_weights(), year)

        # Combine the server’s update with remote updates
        client_updates.append(server_update)
        new_weights = aggregate_weights(client_updates)

        # Update global model
        global_model.set_weights(new_weights)
        client_updates.clear()
        global_model_version += 1

        logging.info(f"Global model updated for year {year}. New version: {global_model_version}")

        # Evaluate on Apple’s test data (RMSE)
        rmse = compute_rmse_on_year_data(global_model, year)
        logging.info(f"Year {year} - RMSE on Apple’s test data: {rmse:.4f}")

        # Export model
        export_path = f"global_model_{year}.h5"
        try:
            global_model.save(export_path)
            logging.info(f"Global model exported to {export_path}")
        except Exception as e:
            logging.error(f"Error exporting model: {str(e)}")

        # Move on to next year
        current_year_index += 1
        training_in_progress = False

    return jsonify({"status": "update received"})

@app.route('/wait_for_update', methods=['GET'])
def wait_for_update():
    """
    Clients call this to check if the global model version has changed (new round).
    We also return how many years remain to be processed, so clients can stop
    once no more years are left.
    """
    global current_year_index

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
                "version": global_model_version,
                "remaining_years": max(0, len(YEARS) - current_year_index)
            })
        time.sleep(poll_interval)
        waited += poll_interval

    # Timed out, return current state
    weights = global_model.get_weights()
    serialized_weights = [w.tolist() for w in weights]
    return jsonify({
        "weights": serialized_weights,
        "version": global_model_version,
        "timeout": True,
        "remaining_years": max(0, len(YEARS) - current_year_index)
    })

@app.route('/export_model', methods=['POST'])
def export_model():
    """Manual export endpoint, if needed."""
    data = request.get_json() or {}
    export_path = data.get("export_path", "global_model.h5")
    try:
        global_model.save(export_path)
        return jsonify({"status": "success", "message": f"Model exported to {export_path}"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
