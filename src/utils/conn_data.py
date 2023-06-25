import pickle
import pandas as pd
import os
import torch

from settings import INPUTS_PATH

def find_gpu_device():
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    
    return device_name

def save_pickle(path: str,
                obj: dict):

    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):
    file = open(path, 'rb')
    target_dict = pickle.load(file)

    return target_dict

def load_data(dataset_name):
    df = pd.read_csv(os.path.join(INPUTS_PATH,  "{}.csv".format(dataset_name)))
    df.set_index("date", inplace=True)

    df = df.dropna()

    return df
