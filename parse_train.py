import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import networkx as nx
import math
from pathlib import Path
from datetime import datetime

def load_and_process_data(file_path, time_str, sheet_name='Data', summary_sheet='Summary Of Data', header=1):
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    
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
        summary_df = pd.read_excel(file_path, engine=engine, sheet_name=summary_sheet, skiprows=3)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        raise
    
    # Debug: Print summary sheet columns and first 5 rows
    print(f"Summary sheet columns: {list(summary_df.columns)}")
    print(f"Summary sheet first 5 rows:\n{summary_df.head().to_string()}")
    
    # Find SCATS Number and Total columns
    scats_col = None
    for col in summary_df.columns:
        if 'scats' in col.lower() and 'number' in col.lower():
            scats_col = col
            break
    if scats_col is None:
        summary_df.columns = ['SCATS Number', 'Location', 'Total', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5']
        scats_col = 'SCATS Number'
        print("Assigned manual column names: ['SCATS Number', 'Location', 'Total', ...]")
    
    total_col = 'Total' if 'Total' in summary_df.columns else summary_df.columns[summary_df.columns.str.contains('Total', case=False, na=False)].tolist()[0]
    
    # Fill NaN values in SCATS Number
    summary_df[scats_col] = summary_df[scats_col].ffill().astype(int)
    
    # Get valid SCATS numbers and total valid days
    valid_scats = summary_df[scats_col].dropna().astype(int).unique()
    scats_days = {}
    for scats in valid_scats:
        total_days = summary_df[summary_df[scats_col] == scats][total_col].sum()
        if total_days > 0:
            scats_days[scats] = int(total_days)
    
    # Extract coordinates (first per SCATS number)
    locations = df[['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()
    locations = locations.rename(columns={'SCATS Number': 'scats_number', 'NB_LATITUDE': 'latitude', 'NB_LONGITUDE': 'longitude'})
    locations = locations[locations['scats_number'].notna() & locations['scats_number'].isin(valid_scats)]
    locations['scats_number'] = locations['scats_number'].astype(int)
    locations = locations.groupby('scats_number').first().reset_index()[['scats_number', 'latitude', 'longitude']].values.tolist()
    
    scats_numbers = np.array([int(loc[0]) for loc in locations])
    
    # Map time_str to column name
    try:
        time_obj = datetime.strptime(time_str, '%H:%M')
        time_index = int((time_obj.hour * 60 + time_obj.minute) / 15)
        time_column = f'V{time_index:02d}'
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM (e.g., '0:00').")
    
    # Verify time column exists
    if time_column not in df.columns:
        raise ValueError(f"Time column {time_column} not found in data. Available columns: {list(df.columns)}")
    
    flow_columns = [f'V{i:02d}' for i in range(96)]
    if not all(col in df.columns for col in flow_columns):
        missing = [col for col in flow_columns if col not in df.columns]
        raise ValueError(f"Missing flow columns: {missing}")
    
    # Select SCATS Number, Date, time_column, and flow_columns
    columns_to_select = ['SCATS Number', 'Date', time_column] + flow_columns
    # Remove duplicates in columns_to_select
    columns_to_select = list(dict.fromkeys(columns_to_select))
    traffic_data = df[df['SCATS Number'].isin(scats_numbers)][columns_to_select]
    traffic_data['SCATS Number'] = traffic_data['SCATS Number'].astype(int)
    
    # Debug: Print traffic_data shape and columns
    print(f"traffic_data shape: {traffic_data.shape}")
    print(f"traffic_data columns: {list(traffic_data.columns)}")
    
    # Scale flows to hourly (multiply by 4)
    traffic_data[time_column] = traffic_data[time_column].mul(4, fill_value=0)
    traffic_data[flow_columns] = traffic_data[flow_columns].mul(4, fill_value=0)
    
    # Filter by valid days
    traffic_data['Date'] = pd.to_datetime(traffic_data['Date'], errors='coerce')
    valid_traffic_data = []
    for scats in scats_numbers:
        scats_data = traffic_data[traffic_data['SCATS Number'] == scats]
        if scats in scats_days:
            valid_days = sorted(scats_data['Date'].dt.strftime('%-m/%-d/%y').dropna().unique())[:scats_days[scats]]
            scats_data = scats_data[scats_data['Date'].dt.strftime('%-m/%-d/%y').isin(valid_days)]
        valid_traffic_data.append(scats_data)
    traffic_data = pd.concat(valid_traffic_data, ignore_index=True)
    
    # Average time-specific flows per SCATS number
    time_specific_flow = traffic_data.groupby('SCATS Number')[time_column].mean().reset_index()
    time_specific_flow = time_specific_flow[time_specific_flow['SCATS Number'].isin(scats_numbers)]
    
    # Prepare time-series data (averaged flows for all time columns)
    traffic_data_series = traffic_data.groupby('SCATS Number')[flow_columns].mean().reset_index()
    
    max_days = 31
    timesteps_per_day = 96
    max_timesteps = max_days * timesteps_per_day
    site_traffic_data = []
    for scats in scats_numbers:
        site_data = traffic_data_series[traffic_data_series['SCATS Number'] == scats][flow_columns].values
        if site_data.shape[0] > 0:
            flattened_data = site_data.flatten()
            if len(flattened_data) < max_timesteps:
                padded_data = np.pad(flattened_data, (0, max_timesteps - len(flattened_data)), mode='constant', constant_values=0)
                site_traffic_data.append(padded_data)
            else:
                site_traffic_data.append(flattened_data[:max_timesteps])
        else:
            site_traffic_data.append(np.zeros(max_timesteps))
    
    site_traffic_data = np.array(site_traffic_data)
    print(f"Site traffic data shape: {site_traffic_data.shape}")
    
    # Create traffic flows array
    traffic_flows = np.zeros(len(scats_numbers))
    for i, scats in enumerate(scats_numbers):
        flow = time_specific_flow[time_specific_flow['SCATS Number'] == scats][time_column]
        traffic_flows[i] = flow.values[0] if not flow.empty else 0.0
    
    # Debug: Print time-specific flows
    print(f"Time-specific flows for {time_str}:\n{time_specific_flow.to_string()}")
    
    return site_traffic_data, locations, scats_numbers, traffic_flows

def prepare_time_series_data(data, look_back=4):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
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
    
    if isinstance(flow, (np.ndarray, list)):
        flow = np.array(flow)
        discriminant = b**2 - 4*a*c
        speeds = np.zeros_like(flow, dtype=float)
        mask = discriminant >= 0
        speeds[~mask] = 60.0
        speed1 = (-b + np.sqrt(discriminant[mask])) / (2*a)
        speed2 = (-b - np.sqrt(discriminant[mask])) / (2*a)
        speeds[mask] = np.where(flow[mask] <= 351, np.minimum(60.0, np.maximum(speed1, speed2)),
                               np.where(flow[mask] <= 1500, np.maximum(speed1, speed2),
                                        np.minimum(speed1, speed2)))
    else:
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
    
    return speeds

def create_traffic_network(locations, site_ids, predicted_flows, all_site_ids):
    graph = nx.Graph()
    site_to_coords = {int(site_id): (lat, lon) for site_id, lat, lon in locations}
    
    edge_dict = {
        4063: [4057, 4034, 2200, 3127],
        2820: [3662, 4321], #2827 & 2825 are part of the freeway, maybe add
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
        2825: [4030], #2820 can be accessed via freeway, maybe add
        3126: [3127, 3682],
        2000: [3685, 4043, 3682],
        4273: [4043, 4272],
        4812: [4270],
        2827: [4051], #2820 can connect via freeway, maybe add
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
    
    bidirectional_edges = {}
    for site_id_a, neighbors in edge_dict.items():
        bidirectional_edges[site_id_a] = list(set(neighbors))
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
        if len(indices) > 0 and indices[0] < len(predicted_flows):
            site_to_flows[site_id] = float(predicted_flows[indices[0]])
        else:
            site_to_flows[site_id] = 0.0
    
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
            
            speed = flow_to_speed(site_to_flows[site_id_b])
            travel_time = (distance / speed) * 60 + 0.5
            graph.add_edge(site_id_a, site_id_b, weight=travel_time)
            added_edges.add(edge)
    
    if missing_edges:
        print(f"Missing edges due to absent nodes: {missing_edges}")
    
    return graph