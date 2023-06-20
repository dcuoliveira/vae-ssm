import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def decoder_mse(decoder):
     
    true = []
    pred = []
    for i, val in decoder.items():
        true += val["y"].to_list()
        pred += val["pred"].to_list()
    
    mse = mean_squared_error(y_true=true, y_pred=pred)

    return mse

def from_decoder_to_dict(decoder: tuple,
                         data: torch.tensor):
    
    if len(decoder[0]) != len(decoder[1]) != data.shape[0]:
        raise Exception("Input sizes doesnt match")

    n = len(decoder[0])

    output = {}
    for i in range(n):
        pred = decoder[0][i].detach().numpy()
        sd = decoder[1][i].detach().numpy()
        y = data[i, :, :].detach().numpy()

        tmp = {"y": pd.DataFrame(y)[0],
               "pred": pd.DataFrame(pred)[0],
               "sd": pd.DataFrame(sd)[0]}
        output[i] = pd.DataFrame(tmp)

    return output

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