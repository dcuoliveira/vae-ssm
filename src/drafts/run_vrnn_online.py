import argparse
import os
import matplotlib.pyplot as plt
from time import time
import torch.utils.data as data
import torch.nn as nn
import torch
from tqdm import tqdm
import pandas as pd

from models.VRNN import VRNN
from utils.conn_data import load_data
from utils.data_utils import create_online_rolling_window_ts, timeseries_train_test_split_online

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="fredmd_raw_df", help='dataset name to be loaded')
parser.add_argument('--num_timesteps_in', type=int, default=20, help='init steps to estimate before predicting')
parser.add_argument('--num_timesteps_out', type=int, default=1, help='steps ahead to predict')
parser.add_argument('--fix_start', type=bool, default=False, help='fix start of the window')
parser.add_argument('--batch_size', type=int, default=10, help='batch_size to sample from the rolling window tensor')
parser.add_argument('--train_ratio', type=int, default=0.5, help='ratio of the data to consider as training')
parser.add_argument('--train_shuffle', type=bool, default=False, help='shuffle blocks of the training data')
parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs to consider')
parser.add_argument('--h_dim', type=int, default=5, help='size of the rnn latent space')
parser.add_argument('--z_dim', type=int, default=5, help='size of the vae latent space')
parser.add_argument('--n_layers', type=int, default=1, help='number of hidden layers in the rnn model')
parser.add_argument('--model_name', type=str, default="vrnn", help='model name to be saved')

if __name__ == "__main__":       

    args = parser.parse_args()

    # training hyperparameters
    seed = 199402
    num_timesteps_in = args.num_timesteps_in
    num_timesteps_out = args.num_timesteps_out
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    n_epochs = args.n_epochs
    fix_start = args.fix_start
    train_shuffle = args.train_shuffle
    drop_last = False
    
    # optimization hyperparameters
    learning_rate = 1e-6
    max_norm_clip = 10
    print_every = 10
    device = torch.device('cpu')
    mse_loss = nn.MSELoss()

    # load toy data
    df = load_data(dataset_name=args.dataset_name)
    # cpi all items yoy
    timeseries = torch.tensor((df[["CPIAUCSL"]].pct_change(12) * 100).dropna().values.astype('float32'))

    # subset timeseries to be divisble by (num_timesteps_in + 1)
    # this is a way to avoid rolling windows of different sizes
    timeseries = timeseries[:-(timeseries.shape[0] % (num_timesteps_in + 1)), :]
   
    # create rolling window block timeseries
    # features = target because we are in the context of univariate timeseries forecasting (i.e. y_t = f(y_t-1, y_t-2, ...))
    X_steps, y_steps = create_online_rolling_window_ts(features=timeseries, 
                                                       target=timeseries,
                                                       num_timesteps_in=num_timesteps_in,
                                                       num_timesteps_out=num_timesteps_out,
                                                       fix_start=fix_start,
                                                       drop_last=drop_last)

    # model hyperparameters
    x_dim = X_steps.shape[-1]
    h_dim = args.h_dim
    z_dim = args.z_dim
    n_layers =  args.n_layers
    model_name = args.model_name

    # manual seed
    torch.manual_seed(seed)
    plt.ion()

    # define model
    model = VRNN(x_dim, h_dim, z_dim, n_layers)
    model = model.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training and evaluation slots
    results = {}
    train_losses = {}
    test_losses = {}

    init = time()
    model.train()

     # (4) training/validation + oos testing
    test_preds = torch.zeros((X_steps.shape[0], num_timesteps_out, X_steps.shape[2]))
    test_ys = torch.zeros((X_steps.shape[0], num_timesteps_out, X_steps.shape[2]))
    train_loss = torch.zeros((X_steps.shape[0], 1))
    train_kld_loss = torch.zeros((X_steps.shape[0], 1))
    train_nll_loss = torch.zeros((X_steps.shape[0], 1))

    test_loss = torch.zeros((X_steps.shape[0], 1))
    test_kld_loss = torch.zeros((X_steps.shape[0], 1))
    test_nll_loss = torch.zeros((X_steps.shape[0], 1))

    pbar = tqdm(range(X_steps.shape[0]-1), total=(X_steps.shape[0] + 1))
    for step in pbar:
        X_t = X_steps[step, :, :]
        y_t1 = y_steps[step, :, :]

        X_train_t, X_test_t, y_train_t1, y_test_t1 = timeseries_train_test_split_online(X=X_t,
                                                                                        y=y_t1,
                                                                                        train_ratio=train_ratio)
        
        train_loader = data.DataLoader(data.TensorDataset(X_train_t, y_train_t1),
                                       shuffle=train_shuffle,
                                       batch_size=batch_size,
                                       drop_last=drop_last)

        train_loss_vals = 0
        train_kld_loss_vals = 0
        train_nll_loss_vals = 0
        for epoch in range(n_epochs):

            # train/validate model
            model.train()
            kld_loss_val = nll_loss_val = 0
            for X_batch, prices_batch in train_loader:

                # forward propagation
                kld_loss, nll_loss, enc, dec = model.forward(X_batch[None, :, :])   
                y_batch_pred = dec[0][0, :, :]

                kld_loss_val += kld_loss
                nll_loss_val += nll_loss

                # aggregate loss function = KLdivergence - log-likelihood
                loss = (kld_loss + nll_loss)

                # back propagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss_vals += loss.item()
                train_kld_loss_vals += kld_loss.item()
                train_nll_loss_vals += nll_loss.item()

        avg_train_loss_vals = train_loss_vals / (n_epochs * len(train_loader))
        avg_train_kld_vals = train_kld_loss_vals / (n_epochs * len(train_loader))
        avg_train_nll_vals = train_nll_loss_vals / (n_epochs * len(train_loader))

        # oos test model  
        model.eval()
        with torch.no_grad():

            # forward propagation
            kld_loss, nll_loss, enc, dec = model.forward(X_test_t[None, :, :])   
            y_test_pred = dec[0][0, :, :]

            test_loss_val = (kld_loss + nll_loss)

            # save results
            test_preds[step, :, :] = y_test_pred[-num_timesteps_out:]
            test_ys[step, :, :] = y_test_t1[-num_timesteps_out:]

        train_loss[step, :] = avg_train_loss_vals
        train_kld_loss[step, :] = avg_train_kld_vals
        train_nll_loss[step, :] = avg_train_nll_vals

        test_loss[step, :] = test_loss_val
        test_kld_loss[step, :] = kld_loss.item()
        test_nll_loss[step, :] = nll_loss.item()

        pbar.set_description("Steps: %d, Test kld : %1.5f, Test nll : %1.5f" % (step, kld_loss, nll_loss))

    if test_preds.dim() == 3:
        preds = test_preds.reshape(test_preds.shape[0] * test_preds.shape[1], test_preds.shape[2])
    else:
        preds = test_preds

    if test_ys.dim() == 3:
        ys = test_ys.reshape(test_ys.shape[0] * test_ys.shape[1], test_ys.shape[2])
    else:
        ys = test_ys

    # (4) save results
    preds_df = pd.DataFrame(preds.numpy(), columns=["preds"])
    ys_df = pd.DataFrame(ys.numpy(), columns=["target"])

    df = pd.concat([preds_df, ys_df], axis=1)

    results = {

        "train_loss": train_loss,
        "train_kld_loss": train_kld_loss,
        "train_nll_loss": train_nll_loss,
        "test_loss": test_loss,
        "test_kld_loss": test_kld_loss,
        "test_nll_loss": test_nll_loss,
        "prediction": preds_df,
        "target": ys_df

    }
    output_path = os.path.join(os.path.dirname(__file__),
                                "data",
                                "outputs",
                                model_name)
    output_name = "{model_name}_{num_timesteps_in}_{num_timesteps_out}_{epochs}_{batch_size}_{h_dim}_{z_dim}_{n_layers}.pt".format(model_name=model_name,
                                                                                                                                   num_timesteps_in=num_timesteps_in,
                                                                                                                                   num_timesteps_out=num_timesteps_out,
                                                                                                                                   batch_size=batch_size,
                                                                                                                                   epochs=n_epochs,
                                                                                                                                   h_dim=h_dim,
                                                                                                                                   z_dim=z_dim,
                                                                                                                                   n_layers=n_layers)
    torch.save(results, os.path.join(output_path, output_name))
    