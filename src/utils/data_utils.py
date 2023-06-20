import torch
import numpy as np
import pandas as pd

def create_ts_prediction_data(data, seq_length):
    
    X = []
    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        X.append(_x)

    return np.array(X)

def regular_numeric_scaler(data):
    pass

def get_random_data_batch(data, batch_size):
    nrows = data.shape[0]

    bathc_idx = np.random.randint(low=0, high=nrows, size=batch_size)
    selected_data = data[bathc_idx, :]

    return torch.tensor(selected_data).float()