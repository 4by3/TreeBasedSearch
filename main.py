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
import osmnx as ox
import contextily as ctx
import geopandas as gpd


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
    
    plot_traffic_network(paths)

    return paths

def plot_traffic_network(paths):

#Manually mapped all accurate coordinates according to google maps/ double checked in Open Street Map

    accurate_coords = {
    4063: [145.08134332548985, -37.81283592877103],
    2820: [145.03155183066065, -37.79296986828489],
    3682: [145.09820331372262, -37.8359673054147],
    3002: [145.02788339408582, -37.813637197375414],
    3180: [145.0848573563545, -37.79460273158218],
    2200: [145.09938334560562, -37.81501296725529],
    4264: [145.03535015355246, -37.82261254350859],
    3812: [145.0622012005279, -37.83601917494857],
    4272: [145.04769929673225, -37.830313924346484],
    4263: [145.02638402878858, -37.82154005135917],
    2846: [145.0601568178126, -37.86001194176046], #just before monash freeway
    4821: [145.00989697582207, -37.81149077149288],
    3662: [145.02905191330416, -37.80729549310517],
    4270: [145.0341994077563, -37.828738880379774],
    4030: [145.0639152957734, -37.793410512099776],
    2825: [145.0634406119048, -37.78534221810583], # ramp to freeway
    3126: [145.09992869910593, -37.82624711113532], #Note: importing boroondoora using osmnx plots SCATS 3126 & 3682 slightly incorrectly as the actual sites are not within Boroondoora 
    2000: [145.09562180317207, -37.850357259161946],
    4273: [145.0449896508963, -37.845436627841615],
    4812: [145.0170085329806, -37.827281845132845],
    2827: [145.0784185427985, -37.78120949356685], #offramp
    4035: [145.05963746600156, -37.81588248518899],
    4262: [145.01645198186006, -37.820083565362474],
    4324: [145.0381570116631, -37.80771533828903],
    970:  [145.092746893974, -37.865761744859604],
    3122: [145.06569125592358, -37.82225747406839],
    4034: [145.06068839578037, -37.81037153542011],
    4051: [145.07048808811066, -37.792597207687244],
    4057: [145.08315409898924, -37.80348407348983],
    3127: [145.07923377093226, -37.823725416386594],
    3685: [145.0951098691237, -37.8532106306441],
    4043: [145.05397277116634, -37.84570814232805],
    3001: [145.02350920104078, -37.813099463343434],
    4321: [145.05063645176094, -37.79934144010679],
    4032: [145.06252651285038, -37.80078978584537],
    4040: [145.0567067979362, -37.83136895498156],
    3804: [145.06368784982087, -37.83214248524872],
    3120: [145.05859741155297, -37.82132373680786],
    4335: [145.03589984452952, -37.8049439439461],
    4266: [145.0448016388467, -37.82373473863532]
        
    }


    map_filter = ('["highway"~"motorway|trunk|primary|secondary|tertiary"]')
    Boroondoora = ox.graph_from_place("City of Boroondara, Victoria, Australia", network_type='drive', custom_filter = map_filter)
    Boroondoora.add_node(4821, x = accurate_coords[4821][0], y = accurate_coords[4821][1]  )
    Boroondoora.add_node(2846, x = accurate_coords[2846][0], y = accurate_coords[2846][1]  )
    Boroondoora.add_edge(4821, 12069077120 )
    Boroondoora.add_edge(2846, 28175517)
    #manually added edge to node 4821 since it is out of the range of the map.

    scats_id = []
    actual_nodes = []
    for Scat, (lon, lat) in accurate_coords.items():
        try:
            
            node_id = ox.distance.nearest_nodes(Boroondoora, X = lon, Y = lat)
            actual_nodes.append(node_id)
            scats_id.append(str(Scat))

        except Exception as e:
            print(f"Skipping {Scat}: {e}")


    fig, ax = ox.plot_graph(
    Boroondoora,
    node_size=0,
    edge_linewidth=0.8,
    bgcolor='white',
    show=False,
    close=False
    )

    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    for i, paths in enumerate(paths):
        path = paths['path']
        travel_time = paths['travel_time']
        
        #Getting coordinates from path
        xs = [accurate_coords[n][0] for n in path]
        ys = [accurate_coords[n][1] for n in path]
        
        #Plotting routes
        ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=2, label=f"Route {i+1} ({travel_time:.2f} min)")

    #Creating visible nodes
    x = [Boroondoora.nodes[n]['x'] for n in actual_nodes]
    y = [Boroondoora.nodes[n]['y'] for n in actual_nodes]
    ax.scatter(x, y, c='red', s=15, label='SCAT locations')

    

    #Adding the Boroondoora map background
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ctx.add_basemap(ax, crs=ox.settings.default_crs, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.legend()

    for xi, yi, label in zip(x, y, scats_id):
        ax.text(xi, yi + 0.0005, label, fontsize=5, ha='center', va='bottom', color='black')

    #Saving to traffic_network.png
    plt.savefig("traffic_network.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ox.plot_graph(Boroondoora, node_size=3, edge_linewidth= 0.7, show = True)





   


if __name__ == "__main__":
    file_path = "Scats Data October 2006.xls"
    origin = 2200
    destination = 2846
    ml = "FNN"  # Test with FNN
    time = "0:00"
    main(file_path, origin, destination, ml, time)