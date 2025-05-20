import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import networkx as nx
from parse_train import load_and_process_data, prepare_time_series_data, create_traffic_network
from train import LSTMModel, GRUModel, train_model
from algorithms.astar import astar
import torch
import math
# new stuff from my fnn file
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# small feedforward brain that learns travel time
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
    
#added recently : accept a time argument and map it to an index matching downsampled data ###################################
def time_to_index(time_str, start_time="00:00", interval_hours=2):
    from datetime import datetime, timedelta
    fmt = "%H:%M"
    t = datetime.strptime(time_str, fmt)
    start = datetime.strptime(start_time, fmt)
    delta = t - start
    total_minutes = delta.seconds // 60
    # Round to nearest 2-hour interval in minutes (120 min)
    nearest_2h_minutes = round(total_minutes / (interval_hours * 60)) * (interval_hours * 60)
    # Compute index
    index = nearest_2h_minutes // (interval_hours * 60)
    return index

# turns flow into time using weird traffic math
def compute_fnn_travel_time(data):
    volume_cols = [i for i in range(data.shape[1])]
    X = data[:, volume_cols]
    flow = np.mean(X, axis=1)

    # do the dumb equation to get speed from flow
    speed = np.where(flow <= 351, 60, 0)
    mask = flow > 351
    a, b, c = -1.4648375, 93.75, -flow[mask]
    disc = b**2 - 4*a*c
    root = np.sqrt(disc)
    congested_speed = (-b + root) / (2*a)
    speed[mask] = congested_speed
    speed = np.clip(speed, 1, 60)

    # physics time = distance / speed + 30s delay
    travel_time = (0.5 / (speed / 60)) + 0.5  # 0.5km link + 30sec
    return X, travel_time

# trains the fnn model on the flow + travel time
# def train_fnn(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
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
            print(f'fnn epoch {epoch} – loss: {loss.item():.4f}')

    # not perfect but it learns
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        test_loss = loss_fn(preds, y_test)
        print(f'fnn test loss: {test_loss.item():.4f}')
        for i in range(3):
            print(f'actual: {y_test[i].item():.2f} – predicted: {preds[i].item():.2f}')

    return model

# finds shortest paths based on updated travel times (used by A*)
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
        
        # Temporarily increase weights instead of removing edges
        for j in range(len(path)-1):
            for neighbor, weight in graph_dict[path[j]]:
                if neighbor == path[j+1]:
                    graph_dict[path[j]] = [(n, w*2 if n == neighbor else w) for n, w in graph_dict[path[j]]]
    
    # Restore original graph
    graph_dict.update(original_graph_dict)
    
    return paths

def main(file_path, origin, destination, ml, time):
    traffic_data, locations, scats_numbers = load_and_process_data(file_path, sheet_name='Data', header=1)

    #paramater time has not yet been implemented

# do the fnn thing first
    #print("training fnn... hold on")
    #X_fnn, y_fnn = compute_fnn_travel_time(traffic_data)
    #fnn_model = train_fnn(X_fnn, y_fnn)

    look_back = 4
    X_train, y_train, X_test, y_test, scaler = prepare_time_series_data(traffic_data, look_back)
    
    input_size = X_train.shape[2]
    hidden_size = 50
    num_layers = 2
    output_size = X_train.shape[2]
    
#FUTURE TO DO: PLEASE MOVE THESE MESSAGES/TRAINING STUFFF INTO THEIR OWN FUNCTIONS SO THAT THE PROGRAM DOESNT KEEP TRAINING EVERYTIME A ROUTE IS REQUESTED

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
    

    #####added recently###################################################################
    time_idx = time_to_index(time)
    # Ensure time index is in range
    if time_idx >= len(y_test):
      print(f"Selected time {time} is out of prediction range.")
      return
    if(ml == "LSTM"):
        predictions = lstm_predictions
    if(ml == "GRU"):
        predictions = gru_predictions
    if(ml == "FNN"):
        print("not implemented")
        #PLS INTEGRATE FNN INTO MAIN PLS THANKS
        return

    # Previous prediction selection code: "Selects best model"
    # predictions = lstm_predictions if lstm_mse < gru_mse else gru_predictions

    G = create_traffic_network(locations, scats_numbers, np.array([predictions]), scats_numbers) #changed predictions to np.array()
    nodes = {int(scats): (lat, lon) for scats, lat, lon in locations}
    


    
    
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
    # Example usage (integrate this into your main function)
    plot_traffic_network(G, nodes, paths)

import matplotlib.pyplot as plt
import networkx as nx

def plot_traffic_network(G, nodes, paths):
    #Issues: Certain routes overlap entirely, see if can fix? 



    # Create a new figure
    plt.figure(figsize=(10, 8))
    
    # Plot all nodes
    for node, (lat, lon) in nodes.items():
        plt.scatter(lon, lat, c='blue', s=100, label='Nodes' if node == list(nodes.keys())[0] else "", zorder=3)
        plt.text(lon, lat, str(node), fontsize=8, ha='right', va='bottom', zorder=4)
    
    # Plot all edges
    for u, v in G.edges():
        u_lon, u_lat = nodes[u][1], nodes[u][0]
        v_lon, v_lat = nodes[v][1], nodes[v][0]
        plt.plot([u_lon, v_lon], [u_lat, v_lat], 'k-', alpha=0.3, linewidth=1, zorder=1)
    
    # Define colors for paths
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    
    # Plot each path
    for i, path_info in enumerate(reversed(paths)):
        path = path_info['path']
        travel_time = path_info['travel_time']
        color = colors[i % len(colors)]
        
        # Plot edges of the path
        for j in range(len(path)-1):
            u, v = path[j], path[j+1]
            if G.has_edge(u, v):
                u_lon, u_lat = nodes[u][1], nodes[u][0]
                v_lon, v_lat = nodes[v][1], nodes[v][0]
                plt.plot([u_lon, v_lon], [u_lat, v_lat], color=color, linewidth=2, zorder=2, 
                         label=f'Route {i+1} ({travel_time:.2f} min)' if j == 0 else "")
    
    # Add title and labels
    plt.title('Traffic Network with Optimal Routes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add legend
    plt.legend()
    
    # Save and show the plot
    plt.savefig('traffic_network.png')
    


#Default route when main.py is called
if __name__ == "__main__":
    file_path = "Scats Data October 2006.xls"
    origin = 2200
    destination = 3120
    ml = "LSTM"
    time = "00:00"  # default time matching your downsampling
    main(file_path, origin, destination, ml, time) #added time###################