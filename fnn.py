import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- Load SCATS data from Excel ----------------
def load_scats_data(file_path):
    # slap open the excel file and drop any garbage rows
    df = pd.read_excel(file_path, sheet_name='Data', header=1)

    # get rid of broken SCATS sites (you can argue later if this list is legit)
    bad_sites = [970, 2000, 3001, 3002, 3685, 4057]
    df = df[~df['SCATS Number'].isin(bad_sites)]

    # grab the volume columns (V00–V95)
    volume_cols = [col for col in df.columns if col.startswith('V')]
    df = df.dropna(subset=volume_cols)

    # average flow across all volume cols, cause that's what the diagram expects
    avg_flow = df[volume_cols].mean(axis=1)
    return df[volume_cols].values, avg_flow


# ---------------- Convert flow to travel time ----------------
def flow_to_speed(flow):
    # if flow is low, don't bother – cap at 60 because no one's around
    speed = np.where(flow <= 351, 60, 0)

    # otherwise, solve the dumb parabola for actual congested speed
    mask = flow > 351
    a = -1.4648375
    b = 93.75
    c = -flow[mask]

    discriminant = b**2 - 4*a*c
    root = np.sqrt(discriminant)
    congested_speed = (-b + root) / (2*a)  # lower root, green curve only

    speed[mask] = congested_speed
    return np.clip(speed, 1, 60)  # don’t go faster than light or slower than a slug


def compute_travel_time(flow, distance_km=0.5):
    # classic physics: time = distance / speed
    speed = flow_to_speed(flow)
    time_minutes = (distance_km / (speed / 60)) + 0.5  # 0.5 = 30 sec delay
    return time_minutes


# ---------------- Define the FNN model ----------------
class FNNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(FNNRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # just give me a number, thanks
        )

    def forward(self, x):
        return self.model(x)


# ---------------- Main training flow ----------------
def main():
    file_path = 'Scats Data October 2006.xls'

    X_raw, flow = load_scats_data(file_path)
    y = compute_travel_time(flow)

    # scale the inputs because raw values are chaos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # split into train and test — the usual 80/20 deal
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # make the data play nice with PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # init model and optimizer
    model = FNNRegressor(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---------------- Training Loop ----------------
    for epoch in range(100):
        model.train()
        pred = model(X_train)
        loss = criterion(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch {epoch} — loss: {loss.item():.4f}")

    # ---------------- Evaluation ----------------
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test)
        print(f"\nfinal test loss: {test_loss.item():.4f}")

        # show some predictions because why not
        print("\nsample predictions (actual vs predicted travel times in mins):")
        for i in range(5):
            print(f"  actual: {y_test[i].item():.2f} → predicted: {test_pred[i].item():.2f}")


if __name__ == '__main__':
    main()