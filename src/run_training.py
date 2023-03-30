import os
import pandas as pd
import numpy as np
import argparse
from time import time

from models.machine_learning import RandomForestWrapper
from models.deep_learning import NN3Wrapper
from training.optimization import train_model
from models.models_metadata import models_metadata

parser = argparse.ArgumentParser()
parser.add_argument('--init_steps', type=int, default=252, help='Init steps to estimate before predicting.')
parser.add_argument('--prediction_steps', type=int, default=1, help='Steps ahead to predict.')
parser.add_argument('--model_name', type=str, default="rf", help='Steps to estimate graph embeddings.')
parser.add_argument('--n_iter', type=int, default=10, help='Number of samples from the hyperparameter space.')
parser.add_argument('--n_splits', type=int, default=5, help='Number of splits to use in the cv procedure.')
parser.add_argument('--n_jobs', type=int, default=-1, help='Number of cores to use.')
parser.add_argument('--verbose', type=bool, default=False, help='Print errors and partial results if True.')
parser.add_argument('--seed', type=int, default=1, help='Seed to use in the hyperparameter search.')


if __name__ == '__main__':
    init = time()

    args = parser.parse_args()
    pred_results, fs_results = train_model(df=df,
                                           init_steps=args.init_steps,
                                           predict_steps=args.predict_steps,
                                           Wrapper=models_metadata[args.model_name],
                                           n_iter=args.n_iter,
                                           n_splits=args.n_splits,
                                           n_jobs=args.n_jobs,
                                           verbose=args.verbose,
                                           seed=args.seed)

    tempo = (time() - init) / 60
    print("\nDONE\ntotal run time = ", np.round(tempo, 2), "min")

