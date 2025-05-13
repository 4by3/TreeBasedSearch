import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import networkx as nx
from parse_train import load_and_process_data, prepare_time_series_data, create_traffic_network
from train import LSTMModel, GRUModel, train_model
from algorithms.astar import astar
import torch

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
    original_edges = list(G.edges(data=True))
    
    for _ in range(num_paths):
        goal, _, path, cost = astar(graph_dict, nodes, origin, [destination], heuristic="E")
        if not path:
            break
        paths.append({'path': [int(node) for node in path], 'travel_time': cost})
        for i in range(len(path)-1):
            if G.has_edge(path[i], path[i+1]):
                G.remove_edge(path[i], path[i+1])
                graph_dict[path[i]] = [(n, w) for n, w in graph_dict[path[i]] if n != path[i+1]]
    
    G.add_edges_from(original_edges)
    
    return paths

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