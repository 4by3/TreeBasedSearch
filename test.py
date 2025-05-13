import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import networkx as nx
import math
from datetime import datetime
import heapq
from pathlib import Path
import sys


# Import A* and heuristic functions
from algorithms.astar import astar

# Data Processing
def load_and_process_data(file_path, sheet_name='Data', header=1):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    
    file_ext = Path(file_path).suffix.lower()
    if file_ext == '.xls':
        engine = 'xlrd'
    elif file_ext in ['.xlsx', '.xlsm']:
        engine = 'openpyxl'
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}. Use .xls or .xlsx.")
    
    try:
        df = pd.read_excel(file_path, engine=engine, sheet_name=sheet_name, header=header)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        raise
    
    print("Column names in the dataset:", df.columns.tolist())
    
    required_columns = ['SCATS Number', 'Date', 'NB_LATITUDE', 'NB_LONGITUDE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}")
    
    date_col = None
    for col in df.columns:
        if col.lower().strip() == 'date':
            date_col = col
            break
    
    if date_col is None:
        raise KeyError("No 'Date' column found in the dataset. Available columns: " + str(df.columns.tolist()))
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if df[date_col].isna().any():
                raise ValueError("Some dates could not be parsed.")
        except Exception as e:
            print(f"Error converting Date column: {e}")
            raise
    
    poor_quality_sites = [970, 2000, 2820, 3001, 3002, 3127, 3682, 3685, 4057, 4262, 4263, 4264, 4272, 4812]
    df = df[~df['SCATS Number'].isin(poor_quality_sites)]
    
    df = df.dropna(subset=['NB_LATITUDE', 'NB_LONGITUDE'])
    df = df[df['NB_LATITUDE'].between(-90, 90) & df['NB_LONGITUDE'].between(-180, 180)]
    
    volume_columns = [col for col in df.columns if col.startswith('V')]
    if not volume_columns:
        raise ValueError("No traffic volume columns (V00-V95) found in the dataset.")
    traffic_data = df[volume_columns].values
    
    # Convert SCATS Number to native Python int
    df['SCATS Number'] = df['SCATS Number'].astype(int)
    locations = df[['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].values
    scats_numbers = df['SCATS Number'].values
    
    return traffic_data, locations, scats_numbers

# Prepare Time Series Data
def prepare_time_series_data(data, look_back=4):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i+look_back])
        y.append(scaled_data[i+look_back])
    
    X = np.array(X)
    y = np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_test, y_test, scaler

# PyTorch LSTM Model
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

# PyTorch GRU Model
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

# Training Function with A* Feedback
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
            G = create_traffic_network(locations, scats_numbers, predicted_flows, scats_numbers)
            nodes = {int(scats): (lat, lon) for scats, lat, lon in locations}
            # Convert NetworkX graph to dictionary format
            graph_dict = {int(node): [(int(neighbor), G[node][neighbor]['weight']) for neighbor in G.neighbors(node)] for node in G.nodes}
            _, _, _, path_cost = astar(graph_dict, nodes, int(scats_numbers[0]), [int(scats_numbers[-1])], heuristic="E")
            path_loss = torch.tensor(path_cost if path_cost is not None else 0.0, requires_grad=True)
            
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

# Traffic Flow to Speed Conversion
def flow_to_speed(flow):
    a = -1.4648375
    b = 93.75
    c = -flow
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return 60.0
    
    speed1 = (-b + math.sqrt(discriminant)) / (2*a)
    speed2 = (-b - math.sqrt(discriminant)) / (2*a)
    
    if flow <= 351:
        return min(60.0, max(speed1, speed2))
    elif flow <= 1500:
        return max(speed1, speed2)
    else:
        return min(speed1, speed2)

# Traffic Network Creation
def create_traffic_network(locations, site_ids, predicted_flows, all_site_ids):
    graph = nx.Graph()
    unique_site_ids = np.unique([int(site_id) for site_id in site_ids])
    
    # Add nodes
    for site_id, latitude, longitude in locations:
        site_id = int(site_id)
        if site_id in unique_site_ids:
            graph.add_node(site_id, pos=(latitude, longitude))
    
    # Aggregate predictions by site
    site_to_flows = {}
    for site_id in unique_site_ids:
        indices = np.where(all_site_ids == site_id)[0]
        valid_indices = indices[indices < len(predicted_flows)]
        if len(valid_indices) > 0:
            site_to_flows[site_id] = predicted_flows[valid_indices].mean(axis=0)
        else:
            site_to_flows[site_id] = np.zeros(predicted_flows.shape[1])
    
    # Add edges (only for nearby nodes)
    earth_radius = 6371e3  # meters
    max_distance = 5  # km
    for i, site_id_a in enumerate(unique_site_ids):
        for j, site_id_b in enumerate(unique_site_ids[i+1:], i+1):
            coords_a = locations[np.where(site_ids == site_id_a)[0][0]][1:3]
            coords_b = locations[np.where(site_ids == site_id_b)[0][0]][1:3]
            latitude_a, longitude_a = coords_a
            latitude_b, longitude_b = coords_b
            
            phi_a = math.radians(latitude_a)
            phi_b = math.radians(latitude_b)
            delta_phi = math.radians(latitude_b - latitude_a)
            delta_lambda = math.radians(longitude_b - longitude_a)
            
            haversine_a = math.sin(delta_phi/2)**2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda/2)**2
            haversine_c = 2 * math.atan2(math.sqrt(haversine_a), math.sqrt(1-haversine_a))
            distance = earth_radius * haversine_c / 1000
            
            if distance <= max_distance:
                speed = flow_to_speed(site_to_flows[site_id_b].mean())
                travel_time = (distance / speed) * 60 + 0.5
                graph.add_edge(site_id_a, site_id_b, weight=travel_time)
    
    return graph

# Find Top K Paths with A*
def find_optimal_paths(G, nodes, origin, destination, num_paths=5):
    origin = int(origin)
    destination = int(destination)
    if origin not in G.nodes:
        print(f"Error: Origin {origin} not found in the graph.")
        return []
    if destination not in G.nodes:
        print(f"Error: Destination {destination} not found in the graph.")
        return []
    
    if not nx.has_path(G, origin, destination):
        print(f"Error: No path exists between {origin} and {destination}.")
        return []
    
    # Convert NetworkX graph to dictionary format
    graph_dict = {int(node): [(int(neighbor), G[node][neighbor]['weight']) for neighbor in G.neighbors(node)] for node in G.nodes}
    
    paths = []
    original_edges = list(G.edges(data=True))
    
    for _ in range(num_paths):
        goal, _, path, cost = astar(graph_dict, nodes, origin, [destination], heuristic="E")
        if not path:
            break
        paths.append({'path': [int(node) for node in path], 'travel_time': cost})
        # Temporarily remove edges in the current path
        for i in range(len(path)-1):
            if G.has_edge(path[i], path[i+1]):
                G.remove_edge(path[i], path[i+1])
                # Update graph_dict
                graph_dict[path[i]] = [(n, w) for n, w in graph_dict[path[i]] if n != path[i+1]]
    
    # Restore the graph
    G.add_edges_from(original_edges)
    
    return paths

# Main Execution
def main(file_path, origin, destination):
    traffic_data, locations, scats_numbers = load_and_process_data(file_path, sheet_name='Data', header=1)
    
    look_back = 4
    X_train, y_train, X_test, y_test, scaler = prepare_time_series_data(traffic_data, look_back)
    
    input_size = X_train.shape[2]
    hidden_size = 50
    num_layers = 2
    output_size = X_train.shape[2]
    
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    gru_model = GRUModel(input_size, hidden_size, num_layers, output_size)
    
    print("Training LSTM...")
    lstm_loss = train_model(lstm_model, X_train, y_train, X_test, y_test, scaler, locations, scats_numbers)
    print("Training GRU...")
    gru_loss = train_model(gru_model, X_train, y_train, X_test, y_test, scaler, locations, scats_numbers)
    
    lstm_model.eval()
    gru_model.eval()
    
    with torch.no_grad():
        lstm_predictions = lstm_model(X_test).numpy()
        gru_predictions = gru_model(X_test).numpy()
    
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    gru_predictions = scaler.inverse_transform(gru_predictions)
    y_test_transformed = scaler.inverse_transform(y_test.numpy())
    
    lstm_mse = mean_squared_error(y_test_transformed, lstm_predictions)
    lstm_mae = mean_absolute_error(y_test_transformed, lstm_predictions)
    gru_mse = mean_squared_error(y_test_transformed, gru_predictions)
    gru_mae = mean_absolute_error(y_test_transformed, gru_predictions)
    
    print(f"LSTM MSE: {lstm_mse:.2f}, MAE: {lstm_mae:.2f}")
    print(f"GRU MSE: {gru_mse:.2f}, MAE: {gru_mae:.2f}")
    
    predictions = lstm_predictions if lstm_mse < gru_mse else gru_predictions
    
    G = create_traffic_network(locations, scats_numbers, predictions, scats_numbers)
    nodes = {int(scats): (lat, lon) for scats, lat, lon in locations}
    
    print(f"Graph nodes: {list(G.nodes)}")
    print(f"Graph edges: {list(G.edges)}")
    
    paths = find_optimal_paths(G, nodes, origin, destination)
    
    if not paths:
        print("No valid paths found.")
    else:
        for i, path_info in enumerate(paths, 1):
            print(f"\nRoute {i}:")
            print(f"Path: {' -> '.join(map(str, path_info['path']))}")
            print(f"Estimated travel time: {path_info['travel_time']:.2f} minutes")

if __name__ == "__main__":
    file_path = "Scats Data October 2006.xls"
    origin = 2200
    destination = 3120
    main(file_path, origin, destination)