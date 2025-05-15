import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import networkx as nx
import math
from pathlib import Path

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
    
    df['SCATS Number'] = df['SCATS Number'].astype(int)
    locations = df[['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].values
    scats_numbers = df['SCATS Number'].values
    
    return traffic_data, locations, scats_numbers

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

def create_traffic_network(locations, site_ids, predicted_flows, all_site_ids):
    # Initializing the graph
    graph = nx.Graph()
    unique_site_ids = np.unique([int(site_id) for site_id in site_ids])
    
    # Adding nodes with their geographic positions
    for site_id, latitude, longitude in locations:
        site_id = int(site_id)
        if site_id in unique_site_ids:
            graph.add_node(site_id, pos=(latitude, longitude))
    
    # Calculating average flows for each site
    site_to_flows = {}
    for site_id in unique_site_ids:
        indices = np.where(all_site_ids == site_id)[0]
        valid_indices = indices[indices < len(predicted_flows)]
        if len(valid_indices) > 0:
            site_to_flows[site_id] = predicted_flows[valid_indices].mean(axis=0)
        else:
            site_to_flows[site_id] = np.zeros(predicted_flows.shape[1])
    
    # Defining edges based on SCATS Neighbours from Scats Data.csv
    edge_dict = {
    4063: [4057, 4034, 2200, 3127],
    2820: [3662, 4321],
    3682: [3126, 3804, 2000],
    3002: [3001, 3662, 4263],
    3180: [4051, 4057],
    2200: [4063],
    4264: [4324, 4263, 4266, 4270],
    3812: [4040, 3804],
    4272: [4270, 4040, 4273],
    4263: [3002, 4262, 4264],
    2846: [970],
    4821: [3001],
    3662: [2820, 4335, 4324, 3002, 3001],
    4270: [4812, 4264, 4272],
    4030: [2825, 4051, 4321, 4032],
    2825: [4030],
    3126: [3127, 3682],
    2000: [3685, 4043, 3682],
    4273: [4043, 4272],
    4812: [4270],
    2827: [4051],
    4035: [4034, 3120],
    4262: [4263, 3001],
    4324: [3662, 4034, 4264],
    970: [2846, 3685],
    3122: [3804, 3120, 3127],
    4034: [4032, 4324, 4063, 4035],
    4051: [2827, 4030, 3180],
    4057: [3180, 4032, 4063],
    3127: [4063, 3122, 3126],
    3685: [2000, 970],
    4043: [4040, 4273, 2000],
    3001: [4821, 4262, 3002, 3662],
    4321: [4335, 2820, 4032, 4030],
    4032: [4030, 4321, 4057, 4034],
    4040: [3120, 4266, 4272, 3804, 3812, 4043],
    3804: [3122, 4040, 3812, 3682],
    3120: [3122, 4040, 4035],
    4335: [3662, 4321],
    4266: [4040, 4264]
}
    
    # Calculating distances for travel time using haversine formula
    earth_radius = 6371e3  # in meters
    for site_id_a, neighbors in edge_dict.items():
        if site_id_a not in unique_site_ids:
            continue
        for site_id_b in neighbors:
            if site_id_b not in unique_site_ids:
                continue
            # Ensure both sites exist in the graph
            if site_id_a in graph.nodes and site_id_b in graph.nodes:
                # Get coordinates
                coords_a = locations[np.where(site_ids == site_id_a)[0][0]][1:3]
                coords_b = locations[np.where(site_ids == site_id_b)[0][0]][1:3]
                latitude_a, longitude_a = coords_a
                latitude_b, longitude_b = coords_b
                
                # Haversine formula for distance
                phi_a = math.radians(latitude_a)
                phi_b = math.radians(latitude_b)
                delta_phi = math.radians(latitude_b - latitude_a)
                delta_lambda = math.radians(longitude_b - longitude_a)
                
                haversine_a = math.sin(delta_phi/2)**2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda/2)**2
                haversine_c = 2 * math.atan2(math.sqrt(haversine_a), math.sqrt(1-haversine_a))
                distance = earth_radius * haversine_c / 1000  # Convert to kilometers
                
                # Calculate travel time using flow-to-speed
                speed = flow_to_speed(site_to_flows[site_id_b].mean())
                travel_time = (distance / speed) * 60 + 0.5  # in minutes
                graph.add_edge(site_id_a, site_id_b, weight=travel_time)
    
    return graph