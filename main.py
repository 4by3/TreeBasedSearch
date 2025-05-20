import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler  
import networkx as nx
from parse_train import load_and_process_data, prepare_time_series_data, create_traffic_network
from train import LSTMModel, GRUModel, FNNRegressor, train_model
from algorithms.astar import astar
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Train only the selected model
def train_selected_model(traffic_data, locations, scats_numbers, ml, look_back=4):
    # Prepare data for LSTM and GRU
    X_train, y_train, X_test, y_test, scaler = prepare_time_series_data(traffic_data, look_back)
    
    input_size = X_train.shape[2]  # Single feature
    hidden_size = 50
    num_layers = 2
    output_size = X_train.shape[2]
    
    if ml == "LSTM":
        print("Training LSTM...")
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        loss = train_model(model, X_train, y_train, X_test, y_test, scaler, locations, scats_numbers)
        return model, scaler, X_train, y_train, X_test, y_test, None
    elif ml == "GRU":
        print("Training GRU...")
        model = GRUModel(input_size, hidden_size, num_layers, output_size)
        loss = train_model(model, X_train, y_train, X_test, y_test, scaler, locations, scats_numbers)
        return model, scaler, X_train, y_train, X_test, y_test, None
    elif ml == "FNN":
        print("Training FNN...")
        X_fnn, y_fnn = compute_fnn_travel_time(traffic_data)
        model, fnn_scaler = train_fnn(X_fnn, y_fnn)
        return model, None, None, None, None, None, fnn_scaler
    else:
        raise ValueError(f"Invalid ML model: {ml}")

def train_fnn(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = FNNRegressor(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'FNN epoch {epoch} – loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        test_loss = loss_fn(preds, y_test)
        print(f'FNN test loss: {test_loss.item():.4f}')
        for i in range(3):
            print(f'Actual: {y_test[i].item():.2f} – Predicted: {preds[i].item():.2f}')

    return model, scaler

# Convert flow to travel time
def compute_fnn_travel_time(data):
    # Data is 1D, reshape to 2D for compatibility
    X = data.reshape(-1, 1)
    flow = data  # Single feature

    # Speed calculation
    speed = np.where(flow <= 351, 60, 0)
    mask = flow > 351
    a, b, c = -1.4648375, 93.75, -flow[mask]
    disc = b**2 - 4*a*c
    root = np.sqrt(disc)
    congested_speed = (-b + root) / (2*a)
    speed[mask] = congested_speed
    speed = np.clip(speed, 1, 60)

    # Travel time: distance / speed + 30s delay
    travel_time = (0.5 / (speed / 60)) + 0.5  # 0.5km link + 30sec
    return X, travel_time

# Find optimal paths
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
    
    graph_dict = {int(node): [(int(neighbor), G[node][neighbor]['weight']) for neighbor in G.neighbors(node)] for node in G.nodes}
    
    paths = []
    original_graph_dict = graph_dict.copy()
    
    for i in range(num_paths):
        goal, _, path, cost = astar(graph_dict, nodes, origin, [destination], heuristic="E")
        if not path:
            print(f"No more paths found after {i} paths.")
            break
        paths.append({'path': [int(node) for node in path], 'travel_time': cost})
        
        # Temporarily increase weights
        for j in range(len(path)-1):
            for neighbor, weight in graph_dict[path[j]]:
                if neighbor == path[j+1]:
                    graph_dict[path[j]] = [(n, w*2 if n == neighbor else w) for n, w in graph_dict[path[j]]]
    
    graph_dict.update(original_graph_dict)
    
    return paths

def main(file_path, origin, destination, ml, time):
    # Load data
    traffic_data, locations, scats_numbers = load_and_process_data(file_path, time, sheet_name='Data', header=1)
    print(f"Main: Time {time}, Traffic data shape: {traffic_data.shape}, Sample: {traffic_data[:5]}")
    
    look_back = 4
    input_size = 1  # Single feature (traffic volume for the selected time)
    hidden_size = 50
    num_layers = 2
    output_size = input_size
    
    # Train only the selected model
    print(f"Training {ml} model...")
    model, scaler, X_train, y_train, X_test, y_test, fnn_scaler = train_selected_model(traffic_data, locations, scats_numbers, ml, look_back)
    
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        if ml in ["LSTM", "GRU"]:
            # Prepare time-specific input data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(traffic_data.reshape(-1, 1))
            X = []
            for i in range(len(scaled_data) - look_back):
                X.append(scaled_data[i:i+look_back])
            if len(X) == 0:
                print(f"Error: Not enough data for look_back={look_back}. Traffic data length: {len(traffic_data)}")
                return []
            X = np.array(X)  # Shape: (samples, look_back, 1)
            X_tensor = torch.FloatTensor(X)
            try:
                predictions = model(X_tensor[-1:]).numpy()  # Shape: (1, output_size)
                time_specific_predictions = scaler.inverse_transform(predictions)[0]
            except Exception as e:
                print(f"Error during model prediction: {e}")
                return []
        else:  # FNN
            X_fnn, _ = compute_fnn_travel_time(traffic_data)
            if len(X_fnn) == 0:
                print("No data available for FNN prediction.")
                return []
            X_fnn_time = X_fnn[-1:]  # Use the last time step
            X_fnn_time_scaled = fnn_scaler.transform(X_fnn_time)
            X_fnn_time_tensor = torch.tensor(X_fnn_time_scaled, dtype=torch.float32)
            try:
                time_specific_predictions = model(X_fnn_time_tensor).numpy().flatten()
            except Exception as e:
                print(f"Error during FNN prediction: {e}")
                return []
    
    print(f"Time-specific predictions: {time_specific_predictions[:5]}")
    
    # Create traffic network
    G = create_traffic_network(locations, scats_numbers, time_specific_predictions, scats_numbers)
    nodes = {int(scats): (lat, lon) for scats, lat, lon in locations}
    
    # Find paths
    paths = find_optimal_paths(G, nodes, origin, destination)
    
    print("\n=== Optimal Paths ===")
    if not paths:
        print("No valid paths found.")
    else:
        for i, path_info in enumerate(paths, 1):
            print(f"\nRoute {i}:")
            path = path_info['path']
            print(f"Path: {' -> '.join(map(str, path))}")
            print(f"Estimated travel time: {path_info['travel_time']:.2f} minutes")
            print("Edge details:")
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                if G.has_edge(u, v):
                    weight = G[u][v]['weight']
                    print(f"  {u} -> {v}: {weight:.2f} min")
                else:
                    print(f"  {u} -> {v}: Missing edge!")
    
    plot_traffic_network(G, nodes, paths)
    return paths

import matplotlib.pyplot as plt

def plot_traffic_network(G, nodes, paths):
    plt.figure(figsize=(10, 8))
    for node, (lat, lon) in nodes.items():
        plt.scatter(lon, lat, c='blue', s=100, label='Nodes' if node == list(nodes.keys())[0] else "", zorder=3)
        plt.text(lon, lat, str(node), fontsize=8, ha='right', va='bottom', zorder=4)
    
    for u, v in G.edges():
        u_lon, u_lat = nodes[u][1], nodes[u][0]
        v_lon, v_lat = nodes[v][1], nodes[v][0]
        plt.plot([u_lon, v_lon], [u_lat, v_lat], 'k-', alpha=0.3, linewidth=1, zorder=1)
    
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    for i, path_info in enumerate(paths):
        path = path_info['path']
        travel_time = path_info['travel_time']
        color = colors[i % len(colors)]
        for j in range(len(path)-1):
            u, v = path[j], path[j+1]
            if G.has_edge(u, v):
                u_lon, u_lat = nodes[u][1], nodes[u][0]
                v_lon, v_lat = nodes[v][1], nodes[v][0]
                plt.plot([u_lon, v_lon], [u_lat, v_lat], color=color, linewidth=2, zorder=2, 
                         label=f'Route {i+1} ({travel_time:.2f} min)' if j == 0 else "")
    
    plt.title('Traffic Network with Optimal Routes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig('traffic_network.png')

if __name__ == "__main__":
    file_path = "Scats Data October 2006.xls"
    origin = 2200
    destination = 3120
    ml = "LSTM"
    time = "0:00"
    main(file_path, origin, destination, ml, time)