import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import networkx as nx
from parse_train import load_and_process_data, prepare_time_series_data, create_traffic_network
from train import LSTMModel, GRUModel, FNNRegressor, train_model
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
import matplotlib.pyplot as plt

def train_selected_model(traffic_data, locations, scats_numbers, ml, look_back=4):
    # Prepare data for all models (LSTM, GRU, FNN)
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
        model = FNNRegressor(input_size, hidden_size, look_back, output_size)
        loss = train_model(model, X_train, y_train, X_test, y_test, scaler, locations, scats_numbers)
        return model, scaler, X_train, y_train, X_test, y_test, None
    else:
        raise ValueError(f"Invalid ML model: {ml}")

def find_optimal_paths(G, nodes, origin, destination, model, scaler, ml, traffic_data, look_back=4, num_paths=5):
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
    
    # Create a copy of the graph to modify weights
    G = G.copy()
    
    # Prepare input for ML model to predict edge weights
    if len(traffic_data) < look_back:
        print(f"Error: Not enough data for look_back={look_back}. Traffic data length: {len(traffic_data)}")
        return []
    
    scaler = MinMaxScaler() if scaler is None else scaler
    scaled_data = scaler.fit_transform(traffic_data.reshape(-1, 1))
    X = scaled_data[-look_back:].reshape(1, look_back, 1)  # Shape: (1, look_back, 1)
    X_tensor = torch.FloatTensor(X)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()  # Predicted traffic flow
        predicted_flow = scaler.inverse_transform(predictions)[0]  # Inverse transform to original scale
    
    # Update edge weights in the graph using ML predictions
    for u, v in G.edges():
        try:
            idx = list(G.nodes).index(u)  # Map node to index in predictions
            travel_time = predicted_flow[idx] if idx < len(predicted_flow) else G[u][v]['weight']
            travel_time = max(float(travel_time), 0.1)  # Avoid zero/negative weights
            G[u][v]['weight'] = travel_time
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not update weight for edge ({u}, {v}): {e}")
            continue
    
    paths = []
    for i in range(num_paths):
        try:
            # Find the shortest path using NetworkX's shortest_path
            path = nx.shortest_path(G, source=origin, target=destination, weight='weight')
            # Calculate total travel time for the path
            travel_time = sum(G[path[j]][path[j+1]]['weight'] for j in range(len(path)-1))
            paths.append({'path': [int(node) for node in path], 'travel_time': travel_time})
            
            # Temporarily increase weights to diversify paths
            for j in range(len(path)-1):
                u, v = path[j], path[j+1]
                if G.has_edge(u, v):
                    G[u][v]['weight'] *= 2
        except nx.NetworkXNoPath:
            print(f"No more paths found after {i} paths.")
            break
    
    return paths

def main(file_path, origin, destination, ml, time):
    # Load data
    traffic_data, locations, scats_numbers = load_and_process_data(file_path, time, sheet_name='Data', header=1)
    print(f"Main: Time {time}, Traffic data shape: {traffic_data.shape}, Sample: {traffic_data[:5]}")
    
    look_back = 4
    
    # Train the selected model
    print(f"Training {ml} model...")
    model, scaler, X_train, y_train, X_test, y_test, _ = train_selected_model(traffic_data, locations, scats_numbers, ml, look_back)
    
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        scaler = MinMaxScaler() if scaler is None else scaler
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
    
    print(f"Time-specific predictions: {time_specific_predictions[:5]}")
    
    # Create traffic network
    G = create_traffic_network(locations, scats_numbers, time_specific_predictions, scats_numbers)
    nodes = {int(scats): (lat, lon) for scats, lat, lon in locations}
    
    # Find paths using ML model predictions
    paths = find_optimal_paths(G, nodes, origin, destination, model, scaler, ml, traffic_data, look_back)
    
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
    ml = "FNN"  # Test with FNN
    time = "0:00"
    main(file_path, origin, destination, ml, time)