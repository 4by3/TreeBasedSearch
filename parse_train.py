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
    print("All SCATS sites before filtering:", df['SCATS Number'].unique())

    df = df.dropna(subset=['NB_LATITUDE', 'NB_LONGITUDE'])
    df = df[df['NB_LATITUDE'].between(-90, 90) & df['NB_LONGITUDE'].between(-180, 180)]

    
    
    volume_columns = [col for col in df.columns if col.startswith('V')]
    if not volume_columns:
        raise ValueError("No traffic volume columns (V00-V95) found in the dataset.")
    
    
     # Set Date as index for time series operations
    df.set_index(date_col, inplace=True)
    
    
    # Group by index (datetime) and average only volume columns
    df_numeric = df[volume_columns]  # select only volume columns
    df_numeric = df_numeric.groupby(df_numeric.index).mean()

# If you need to preserve other info, handle separately



    print("SCATS sites after  filtering:", df['SCATS Number'].unique())

    # 1. Reindex to a complete time index (assuming 15 min frequency)
    full_time_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
    df_numeric = df_numeric.reindex(full_time_index)
    
    # 2. Interpolate missing traffic volume data
    df_numeric = df_numeric.interpolate(method='time')
    
    # 3. Downsample to every 2 hours (e.g., mean traffic volume)
    df_downsampled = df_numeric.resample('2h').mean()

    
    # Prepare output variables as before, but from downsampled data
    traffic_data = df_downsampled.values


    '''df['SCATS Number'] = df['SCATS Number'].astype(int)
    locations = df[['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].values
    scats_numbers = df['SCATS Number'].values '''

    # For locations and scats_numbers, use original since locations won't change
    df_locations = df.dropna(subset=['NB_LATITUDE', 'NB_LONGITUDE'])
    locations = df_locations[['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates().values
    scats_numbers = df_locations['SCATS Number'].drop_duplicates().values.astype(int)
    
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
    site_to_coords = {int(site_id): (lat, lon) for site_id, lat, lon in locations}
    
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
    

    # Create fully bidirectional edge dictionary
    bidirectional_edges = {}
    for site_id_a, neighbors in edge_dict.items():
        bidirectional_edges[site_id_a] = list(set(neighbors))  # Remove duplicates
        for site_id_b in neighbors:
            if site_id_b not in bidirectional_edges:
                bidirectional_edges[site_id_b] = []
            if site_id_a not in bidirectional_edges[site_id_b]:
                bidirectional_edges[site_id_b].append(site_id_a)
    
    all_site_ids_in_dict = set(bidirectional_edges.keys())
    for site_id in all_site_ids_in_dict:
        if site_id in site_to_coords:
            graph.add_node(site_id, pos=site_to_coords[site_id])
    
    site_to_flows = {}
    for site_id in all_site_ids_in_dict:
        indices = np.where(all_site_ids == site_id)[0]
        valid_indices = indices[indices < len(predicted_flows)]
        site_to_flows[site_id] = predicted_flows[valid_indices].mean(axis=0) if len(valid_indices) > 0 else np.zeros(predicted_flows.shape[1])
    
    earth_radius = 6371e3
    added_edges = set()
    missing_edges = []
    for site_id_a, neighbors in bidirectional_edges.items():
        if site_id_a not in graph.nodes:
            continue
        for site_id_b in neighbors:
            if site_id_b not in graph.nodes:
                missing_edges.append((site_id_a, site_id_b))
                continue
            edge = tuple(sorted([site_id_a, site_id_b]))
            if edge in added_edges:
                continue
            lat_a, lon_a = site_to_coords[site_id_a]
            lat_b, lon_b = site_to_coords[site_id_b]
            
            phi_a = math.radians(lat_a)
            phi_b = math.radians(lat_b)
            delta_phi = math.radians(lat_b - lat_a)
            delta_lambda = math.radians(lon_b - lon_a)
            haversine_a = math.sin(delta_phi/2)**2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda/2)**2
            haversine_c = 2 * math.atan2(math.sqrt(haversine_a), math.sqrt(1-haversine_a))
            distance = earth_radius * haversine_c / 1000
            
            speed = flow_to_speed(site_to_flows[site_id_b].mean())
            travel_time = (distance / speed) * 60 + 0.5
            graph.add_edge(site_id_a, site_id_b, weight=travel_time)
            added_edges.add(edge)
    
    # Log missing edges for debugging
    if missing_edges:
        print(f"Missing edges due to absent nodes: {missing_edges}")
    
    return graph