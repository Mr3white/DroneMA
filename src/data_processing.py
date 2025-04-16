import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from filterpy.kalman import KalmanFilter

class RSSIDataset(Dataset):
    def __init__(self, processed_data):
        self.processed_data = processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        rssi, dist = self.processed_data[idx]
        return rssi.clone().detach().view(-1, 1), dist.clone().detach().view(-1, 1)

def apply_kalman_filter(rssi_values, R=1, Q=1):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([rssi_values.iloc[0]])
    kf.F = np.array([[1]]) 
    kf.H = np.array([[1]])
    kf.P *= 1000
    kf.R = R
    kf.Q = Q

    filtered_rssi = []
    for rssi in rssi_values:
        kf.predict()
        kf.update(rssi)
        filtered_rssi.append(kf.x[0])
    return filtered_rssi

def preprocess_data(data, window_length):
    data['filtered_rssi'] = apply_kalman_filter(data['rssi'])
    processed_data = []
    for start_index in range(len(data) - window_length + 1):
        end_index = start_index + window_length
        window_rssi = data['rssi'][start_index:end_index].values
        actual_dist = data['dist'][start_index:end_index].values
        processed_data.append((window_rssi, actual_dist))
    return processed_data

def create_dataloader(files, window_length, batch_size=128):
    processed_data = []
    for file in files:
        data = pd.read_csv(file)
        processed = preprocess_data(data, window_length)
        processed_data.extend(processed)
    
    dataset = RSSIDataset(torch.tensor(np.array(processed_data), dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
