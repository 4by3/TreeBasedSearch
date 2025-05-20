import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.astar import astar

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 25),
            nn.ReLU(),
            nn.Linear(25, output_size)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 25),
            nn.ReLU(),
            nn.Linear(25, output_size)
        )
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
    
class FNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(FNNRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, X_train, y_train, X_test, y_test, scaler, locations, scats_numbers, epochs=50, batch_size=32, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            outputs = model(batch_X)
            mse_loss = criterion(outputs, batch_y)
            
            predicted_flows = scaler.inverse_transform(outputs.detach().numpy())
            predicted_flows_avg = predicted_flows.mean(axis=0)
            from parse_train import create_traffic_network
            G = create_traffic_network(locations, scats_numbers, predicted_flows_avg, scats_numbers)
            nodes = {int(scats): (lat, lon) for scats, lat, lon in locations}

            graph_dict = {int(node): [(int(neighbor), G[node][neighbor]['weight']) for neighbor in G.neighbors(node)] for node in G.nodes}
            _, _, _, path_cost = astar(graph_dict, nodes, int(scats_numbers[0]), [int(scats_numbers[-1])], heuristic="E")
            path_loss = torch.tensor(float(path_cost) if path_cost is not None else 0.0, requires_grad=True)

            loss = mse_loss + 0.1 * path_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= (len(X_train) // batch_size)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss.item():.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        
        model.train()
    
    return best_loss