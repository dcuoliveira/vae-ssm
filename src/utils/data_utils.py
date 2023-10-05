import torch
import numpy as np

def create_rolling_indices(num_timesteps_in, num_timesteps_out, n_timesteps, fix_start):
    
    # generate rolling window indices
    indices = []
    for i in range(n_timesteps - num_timesteps_out):

        if fix_start:
            if i == 0:
                indices.append((0, (i + num_timesteps_in)))
            else:
                if indices[-1][1] == (n_timesteps - num_timesteps_out):
                    continue
                indices.append((0,  indices[-1][1] + num_timesteps_out))
        else:
            if i == 0:
                indices.append((i, (i + num_timesteps_in)))
            else:
                if indices[-1][1] == (n_timesteps - num_timesteps_out):
                    continue
                indices.append((indices[-1][0] + num_timesteps_out,  indices[-1][1] + num_timesteps_out))

    return indices

def create_rolling_window_ts(target, features, num_timesteps_in, num_timesteps_out, fix_start=False, drop_last=True):
    """"
    This function is used to create the rolling window time series to be used on DL ex-GNN.

    One important thing to note is that, since we are in the context of sharpe ratio optimization,
    and we are assuming positions are being taken at the close price of the same day, we have to 
    increase our target variable (prices) by one time step so as to compute the sharpe ratio properly.
    """
        
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
        window_features.append(features[i:j, :])
        window_target.append(target[(i + 1):(j + num_timesteps_out), :])

    if drop_last:
        window_features = window_features[:-1]
        window_target = window_target[:-1]

    return torch.stack(window_features), torch.stack(window_target)

def timeseries_train_test_split(X, y, train_ratio):

    if X.shape[0] != y.shape[0]:
        raise Exception("Features and target must have the same number of timesteps")
    
    train_size = int(len(X) * train_ratio)
    
    X_train = X[:train_size, :]
    y_train = y[:train_size, :]
    X_test = X[train_size:, :]
    y_test = y[train_size:, :]

    return X_train, X_test, y_train, y_test

def create_online_rolling_window_ts(target, features, num_timesteps_in, num_timesteps_out, fix_start=False, drop_last=True):
    """"
    This function is used to create the rolling window time series to be used on DL ex-GNN.

    One important thing to note is that, since we are in the context of sharpe ratio optimization,
    and we are assuming positions are being taken at the close price of the same day, we have to 
    increase our target variable (prices) by one time step so as to compute the sharpe ratio properly.
    """
        
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
        window_features.append(features[i:j, :])
        window_target.append(target[i:(j + num_timesteps_out), :])

    if drop_last:
        window_features = window_features[:-1]
        window_target = window_target[:-1]

    return torch.stack(window_features), torch.stack(window_target)

def regular_numeric_scaler(data):
    pass

def get_random_data_batch(data, batch_size):
    nrows = data.shape[0]

    bathc_idx = np.random.randint(low=0, high=nrows, size=batch_size)
    selected_data = data[bathc_idx, :]

    return torch.tensor(selected_data).float()