from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

NUM_CLIENTS = 3   # Number of expected clients per round
TOTAL_ROUNDS = 5  # Total number of federated rounds
current_round = 1
client_updates = []  # List to store client updates for the current round

# Define the model architecture
class StockPredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=16, output_size=1):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

global_model = StockPredictor()

def serialize_state_dict(state_dict):
    """Convert state dict to a JSON-serializable format."""
    serializable_dict = {}
    for key, value in state_dict.items():
        serializable_dict[key] = value.tolist()
    return serializable_dict

def deserialize_state_dict(serialized_state_dict):
    """Reconstruct a state dict from its serialized version."""
    state_dict = {}
    for key, value in serialized_state_dict.items():
        state_dict[key] = torch.tensor(value)
    return state_dict

def aggregate_updates(updates):
    """Aggregate client updates by averaging corresponding parameters."""
    aggregated_state_dict = {}
    for key in updates[0].keys():
        avg_param = torch.stack([upd[key] for upd in updates], dim=0).mean(dim=0)
        aggregated_state_dict[key] = avg_param
    return aggregated_state_dict

def run_test_on_fourth_stock():
    """
    Test the updated global model on synthetic test data that represents the 'fourth stock'.
    For consistency, we use a fixed seed so the test data remains the same across rounds.
    """
    torch.manual_seed(400)
    X_test = torch.randn(100, 10)
    # Here, the target is defined as the sum of features. In a real scenario,
    # you would replace this with the actual testing data for the fourth stock.
    y_test = X_test.sum(dim=1, keepdim=True)
    predictions = global_model(X_test)
    loss = F.mse_loss(predictions, y_test)
    return loss.item()

@app.route('/get_model', methods=['GET'])
def get_model():
    """Endpoint for clients to download the current global model."""
    state = serialize_state_dict(global_model.state_dict())
    return jsonify({'round': current_round, 'model': state})

@app.route('/submit_update', methods=['POST'])
def submit_update():
    """
    Endpoint for clients to submit their locally trained model updates.
    Once all updates for the round are received, the server aggregates them,
    updates the global model, performs testing, and then moves to the next round.
    """
    global current_round, global_model, client_updates
    data = request.get_json()
    client_id = data.get('client_id')
    round_submitted = data.get('round')
    model_update_serialized = data.get('model_update')
    
    if round_submitted != current_round:
        return jsonify({'message': f'Update round mismatch. Current round is {current_round}.'}), 400
    
    model_update = deserialize_state_dict(model_update_serialized)
    client_updates.append(model_update)
    
    if len(client_updates) == NUM_CLIENTS:
        # Aggregate all client updates
        aggregated_state = aggregate_updates(client_updates)
        global_model.load_state_dict(aggregated_state)
        client_updates = []  # Reset for next round
        
        # Run testing on the "fourth stock" after updating the global model
        test_loss = run_test_on_fourth_stock()
        
        response_message = {
            'message': 'Round complete. Global model updated.',
            'test_loss': test_loss,
            'round': current_round
        }
        current_round += 1
        return jsonify(response_message)
    else:
        return jsonify({'message': f'Update received from client {client_id}. Waiting for other clients.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)