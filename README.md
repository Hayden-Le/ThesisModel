# Federated Stock Prediction with LSTM

This project demonstrates a federated learning system for stock price prediction using an LSTM model. The system includes a central server and multiple clients, each running on separate virtual machines (VMs). The server aggregates updates from clients (Microsoft, Intel, NVIDIA) and tests performance using Apple stock data.

---

## Project Files

```
Thesis/
├── server.py
├── client.py
├── preprocess_data.py
├── README.md
├── data/
│   ├── X_train_<year>.npy
│   ├── y_train_<year>.npy
│   ├── X_test_<year>.npy
│   └── y_test_<year>.npy
```

---

## Virtual Environment Setup

### Linux/Mac

1. **Create environment:**

```bash
python3 -m venv venv
```

2. **Activate environment:**

```bash
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install tensorflow flask numpy pandas scikit-learn yfinance
```

### Windows

1. **Create environment:**

```bash
python -m venv venv
```

2. **Activate environment:**

- CMD:

```cmd
venv\Scripts\activate.bat
```

- PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

3. **Install dependencies:**

```bash
pip install tensorflow flask numpy pandas scikit-learn yfinance
```

---

## Data Preprocessing

Run this on each VM using its own CSV file (e.g., `apple.csv`, `microsoft.csv`):

```bash
python preprocess_data.py --input apple.csv --start_year 2010 --end_year 2023 --sequence_length 20 --test_ratio 0.2
```

Generated `.npy` files will be saved under the `data/` folder.

---

## Running the Server

On your server VM:

```bash
source venv/bin/activate
python server.py
```

The server listens on port `5000`. Make sure this port is accessible.

---

## Running the Clients

On each client VM:

```bash
source venv/bin/activate
python client.py --company Microsoft --start_year 2010 --end_year 2023
```

Change `Microsoft` to `Intel` or `NVIDIA` accordingly.

---

## Important Notes

- **Server URL:** Ensure each `client.py` file correctly references the server IP address (e.g., `http://<server-ip>:5000`).
- **Firewall/Networking:** Port `5000` must be open for communication between clients and server.
- **Environment:** Always activate your virtual environment before running scripts.
- **Troubleshooting:**
  - If encountering "development server" warnings from Flask, consider using a production-grade server like Gunicorn.
  - The `.h5` model-saving warnings can be ignored or changed to `.keras`.

---

Now your federated learning setup is ready to run and train across multiple rounds automatically.

# Federated Learning System for Stock Price Prediction

This document describes in detail the federated learning system implemented to predict stock prices using an LSTM-based neural network.

---

## Overview

Federated learning enables decentralized model training where multiple client nodes independently train models on local data, periodically sending updates to a central server that aggregates them to produce a unified global model.

This system predicts future stock prices using historical stock market data. The server aggregates updates from several clients, each holding data from different companies (Microsoft, Intel, NVIDIA). The server evaluates the global model on Apple's stock data.

---

## Model Architecture

The model employs an LSTM (Long Short-Term Memory) neural network:

- **Input Layer:** Time series sequences (length = 20).
- **Hidden Layer:** LSTM layer with 32 units.
- **Output Layer:** Single neuron predicting the next-day stock price.

### Structure:
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(20, 1)),
    tf.keras.layers.Dense(1)
])
```

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

---

## Variables and Data

- **Sequence length:** 20 days (previous 20 days' prices to predict the next day's price).
- **Data Features:** Stock price (closing price normalized).
- **Normalization:** Min-Max scaling to [0,1].
- **Data Split:** 80% training and 20% testing.

---

## Federated Learning Algorithm

### Client Side

For each year:

1. Download the current global model from the server.
2. Train the model locally on company-specific stock data for that year.
3. Send the updated weights to the server.
4. Wait for the server to provide an updated global model before proceeding to the next year.

### Server Side

For each year:

1. Receive model updates from all clients.
2. Aggregate the weights (average).
3. Train additionally on Apple's local data.
4. Update the global model.
5. Evaluate global model performance (RMSE) using Apple's test data.
6. Export and save the global model.

---

## Model Aggregation

Server aggregates client model updates by averaging:

```python
def aggregate_weights(updates):
    aggregated_weights = []
    for weights in zip(*updates):
        aggregated_weights.append(np.mean(np.array(weights), axis=0))
    return aggregated_weights
```

---

## Evaluation Metric

The global model is evaluated yearly using:

- **Root Mean Squared Error (RMSE)**:

\[ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} \]

Lower RMSE indicates better prediction accuracy.

---

## Workflow Summary

- **Data Preprocessing:**
  - Normalization and sequence generation.

- **Model Initialization:**
  - Initial LSTM model is generated on the server.

- **Federated Training (yearly loop):**
  - Clients train locally and send updates.
  - Server aggregates weights, performs local training on Apple data, evaluates the model, and exports it.

---

## Advantages

- **Data Privacy:** Clients' data remain local.
- **Scalability:** Easily add more clients.
- **Decentralization:** Less computational load on the central server.

---

## Conclusion

This federated learning system leverages decentralized LSTM training to predict stock prices, ensuring data privacy while improving global predictive performance over time.

