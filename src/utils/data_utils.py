import torch
import numpy as np

def create_rolling_indices(num_timesteps_in, num_timesteps_out, n_timesteps, fix_start):
    
    # generate rolling window indices
    indices = [
        (0 if fix_start else i, i + (num_timesteps_in + num_timesteps_out))
        for i in range(n_timesteps - (num_timesteps_in + num_timesteps_out) + 1)
    ]

    return indices

def create_rolling_window_ts(target, features, num_timesteps_in, num_timesteps_out, fix_start=False):
        
    if features.shape[0] != target.shape[0]:
        raise Exception("Features and target must have the same number of timesteps")

    n_timesteps = features.shape[0]
    indices = create_rolling_indices(num_timesteps_in=num_timesteps_in,
                                     num_timesteps_out=num_timesteps_out,
                                     n_timesteps=n_timesteps,
                                     fix_start=fix_start)
    
    # use rolling window indices to subset data
    window_features, window_target = [], []
    for i, j in indices:
        window_features.append(np.array(features[i:(i + num_timesteps_in), :]))
        window_target.append(np.array(target[(i + num_timesteps_in - 1):j, :]))

    return torch.tensor(window_features), torch.tensor(window_target)

def timeseries_train_test_split(X, y, train_ratio):
    train_ratio = int(len(X) * train_ratio)
    
    X_train = X[:train_ratio, :]
    y_train = y[:train_ratio, :]
    X_test = X[train_ratio:, :]
    y_test = y[train_ratio:, :]

    return X_train, X_test, y_train, y_test

def regular_numeric_scaler(data):
    pass

def get_random_data_batch(data, batch_size):
    nrows = data.shape[0]

    bathc_idx = np.random.randint(low=0, high=nrows, size=batch_size)
    selected_data = data[bathc_idx, :]

    return torch.tensor(selected_data).float()