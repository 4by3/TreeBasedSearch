import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import contextily as ctx
from parse_train import load_and_process_data, prepare_time_series_data, flow_to_speed, create_traffic_network
from networkx.algorithms.simple_paths import shortest_simple_paths

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class FNNModel(nn.Module):
    def __init__(self, input_size=4):
        super(FNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_selected_model(traffic_data, locations, scats_numbers, model_type, look_back=4, epochs=3, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalers = []
    X_trains, y_trains, X_tests, y_tests = [], [], [], []
    
    for site_data in traffic_data:
        X_train, y_train, X_test, y_test, scaler = prepare_time_series_data(site_data, look_back)
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_tests.append(X_test)
        y_tests.append(y_test)
        scalers.append(scaler)
    
    if model_type == "LSTM":
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=2).to(device)
    elif model_type == "GRU":
        model = GRUModel(input_size=1, hidden_size=50, num_layers=2).to(device)
    elif model_type == "FNN":
        model = FNNModel(input_size=look_back).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_train, y_train in zip(X_trains, y_trains):
            X_train, y_train = X_train.to(device), y_train.to(device)
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.6f}")
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_test, y_test in zip(X_tests, y_tests):
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            loss = criterion(outputs, y_test)
            test_loss += loss.item()
    
    print(f"Test Loss: {test_loss:.6f}")
    return model, scalers, X_trains, y_trains, X_tests, y_tests, device

def find_optimal_paths(G, nodes, origin, destination, model, scalers, model_type, traffic_data, look_back, initial_flows):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Verify nodes in graph
    missing_nodes = [node for node in nodes.keys() if node not in G.nodes]
    if missing_nodes:
        print(f"Warning: Nodes not in graph: {missing_nodes}")
    
    # Predict future flows for each site
    predicted_flows = []
    with torch.no_grad():
        for site_idx, site_data in enumerate(traffic_data):
            scaler = scalers[site_idx]
            recent_data = site_data[-look_back:]
            scaled_data = scaler.transform(recent_data.reshape(-1, 1))
            input_tensor = torch.FloatTensor(scaled_data).reshape(1, look_back, 1).to(device)
            if model_type == "FNN":
                input_tensor = input_tensor.reshape(1, look_back)
            output = model(input_tensor)
            predicted_scaled = output.cpu().numpy().flatten()
            predicted_flow = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()[0]
            predicted_flows.append(max(0, predicted_flow))
    
    # Debug: Print flows
    print(f"Initial flows: {initial_flows}")
    print(f"Predicted flows: {predicted_flows}")
    # Blend flows (weighted average, 70% initial, 30% predicted)
    flows = [0.7 * initial + 0.3 * predicted for initial, predicted in zip(initial_flows, predicted_flows)]
    print(f"Blended flows: {flows}")
    
    # Update graph with blended flows
    G_updated = create_traffic_network(
        locations=[(scats, nodes[scats][0], nodes[scats][1]) for scats in nodes],
        site_ids=nodes.keys(),
        predicted_flows=np.array(flows),
        all_site_ids=np.array(list(nodes.keys()))
    )
    
    # Debug: Print edge weights for key edges
    print("Edge weights in updated graph:")
    for u, v, data in G_updated.edges(data=True):
        if u in [2200, 4063, 2825] and v in [4063, 2825, 2846]:
            print(f"  {u} -> {v}: {data['weight']:.2f} min")
    
    # Find up to 5 shortest paths
    paths = []
    try:
        for path in shortest_simple_paths(G_updated, origin, destination, weight='weight'):
            if len(paths) >= 5:
                break
            total_cost = sum(G_updated[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
            paths.append({
                'path': path,
                'travel_time': total_cost
            })
        print(f"Found {len(paths)} paths")
    except Exception as e:
        print(f"Error finding shortest paths: {e}")
    
    return paths

def plot_traffic_network(paths):
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
        2846: [145.0601568178126, -37.86001194176046],
        4821: [145.00989697582207, -37.81149077149288],
        3662: [145.02905191330416, -37.80729549310517],
        4270: [145.0341994077563, -37.828738880379774],
        4030: [145.0639152957734, -37.793410512099776],
        2825: [145.0634406119048, -37.78534221810583],
        3126: [145.09992869910593, -37.82624711113532],
        2000: [145.09562180317207, -37.850357259161946],
        4273: [145.0449896508963, -37.845436627841615],
        4812: [145.0170085329806, -37.827281845132845],
        2827: [145.0784185427985, -37.78120949356685],
        4035: [145.05963746600156, -37.81588248518899],
        4262: [145.01645198186006, -37.820083565362474],
        4324: [145.0381570116631, -37.80771533828903],
        970: [145.092746893974, -37.865761744859604],
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
    
    map_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
    Boroondoora = ox.graph_from_place("City of Boroondara, Victoria, Australia", network_type='drive', custom_filter=map_filter)
    Boroondoora.add_node(4821, x=accurate_coords[4821][0], y=accurate_coords[4821][1])
    Boroondoora.add_node(2846, x=accurate_coords[2846][0], y=accurate_coords[2846][1])
    Boroondoora.add_edge(4821, 12069077120)
    Boroondoora.add_edge(2846, 28175517)
    
    scats_id = []
    actual_nodes = []
    for scat, (lon, lat) in accurate_coords.items():
        try:
            node_id = ox.distance.nearest_nodes(Boroondoora, X=lon, Y=lat)
            actual_nodes.append(node_id)
            scats_id.append(str(scat))
        except Exception as e:
            print(f"Skipping {scat}: {e}")
    
    fig, ax = ox.plot_graph(
        Boroondoora,
        node_size=0,
        edge_linewidth=0.8,
        bgcolor='white',
        show=False,
        close=False
    )
    
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    for i, path_info in enumerate(paths):
        path = path_info['path']
        travel_time = path_info['travel_time']
        xs = [accurate_coords[n][0] for n in path if n in accurate_coords]
        ys = [accurate_coords[n][1] for n in path if n in accurate_coords]
        ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=2, label=f"Route {i+1} ({travel_time:.2f} min)")
    
    x = [Boroondoora.nodes[n]['x'] for n in actual_nodes]
    y = [Boroondoora.nodes[n]['y'] for n in actual_nodes]
    ax.scatter(x, y, c='red', s=15, label='SCATS locations')
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ctx.add_basemap(ax, crs=ox.settings.default_crs, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.legend()
    
    for xi, yi, label in zip(x, y, scats_id):
        ax.text(xi, yi + 0.0005, label, fontsize=5, ha='center', va='bottom', color='black')
    
    plt.savefig("traffic_network.png", dpi=300, bbox_inches='tight')
    plt.close()

def main(file_path, origin, destination, ml, time):
    traffic_data, locations, scats_numbers, initial_flows = load_and_process_data(file_path, time)
    print(f"Main: Time {time}, Traffic data shape: {traffic_data.shape}, Sample: {traffic_data[0][:5]}")
    
    look_back = 4
    
    print(f"Training {ml} model...")
    model, scalers, X_train, y_train, X_test, y_test, device = train_selected_model(
        traffic_data, locations, scats_numbers, ml, look_back
    )
    
    G = create_traffic_network(locations, scats_numbers, initial_flows, scats_numbers)
    nodes = {int(scats): (lat, lon) for scats, lat, lon in locations}
    
    paths = find_optimal_paths(G, nodes, origin, destination, model, scalers, ml, traffic_data, look_back, initial_flows)
    
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

if __name__ == "__main__":
    file_path = "Scats Data October 2006.xls"
    origin = 2200
    destination = 2846
    ml = "FNN"
    time = "0:00"
    main(file_path, origin, destination, ml, time)