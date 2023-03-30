import pickle
import pandas as pd
import os

from settings import INPUTS_PATH

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

    df = df.dropna()

    return df
