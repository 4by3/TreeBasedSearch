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
    graph = nx.Graph()
    unique_site_ids = np.unique([int(site_id) for site_id in site_ids])
    
    for site_id, latitude, longitude in locations:
        site_id = int(site_id)
        if site_id in unique_site_ids:
            graph.add_node(site_id, pos=(latitude, longitude))
    
    site_to_flows = {}
    for site_id in unique_site_ids:
        indices = np.where(all_site_ids == site_id)[0]
        valid_indices = indices[indices < len(predicted_flows)]
        if len(valid_indices) > 0:
            site_to_flows[site_id] = predicted_flows[valid_indices].mean(axis=0)
        else:
            site_to_flows[site_id] = np.zeros(predicted_flows.shape[1])
    
    earth_radius = 6371e3
    max_distance = 5
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