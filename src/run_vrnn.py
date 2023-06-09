import argparse
import os
import matplotlib.pyplot as plt
from time import time
import torch.utils.data as data
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from models.VRNN import VRNN
from utils.conn_data import load_data
from utils.data_utils import timeseries_train_test_split, create_rolling_window_ts

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="fredmd_raw_df", help='dataset name to be loaded')
parser.add_argument('--num_timesteps_in', type=int, default=20, help='init steps to estimate before predicting')
parser.add_argument('--num_timesteps_out', type=int, default=1, help='steps ahead to predict')
parser.add_argument('--fix_start', type=bool, default=False, help='fix start of the window')
parser.add_argument('--batch_size', type=int, default=10, help='batch_size to sample from the rolling window tensor')
parser.add_argument('--train_ratio', type=int, default=0.5, help='ratio of the data to consider as training')
parser.add_argument('--train_shuffle', type=bool, default=True, help='shuffle blocks of the training data')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to consider')
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
    
    # optimization hyperparameters
    learning_rate = 1e-6
    max_norm_clip = 10
    print_every = 10
    device = torch.device('cpu')
    mse_loss = nn.MSELoss()

    # model hyperparameters
    x_dim = 1
    h_dim = args.h_dim
    z_dim = args.z_dim
    n_layers =  args.n_layers
    model_name = args.model_name

    # load toy data
    df = load_data(dataset_name=args.dataset_name)
    # cpi all items yoy
    timeseries = (df[["CPIAUCSL"]].pct_change(12) * 100).dropna().values.astype('float32')

    # create train and test datasets
    X_train, X_val, _, _ = timeseries_train_test_split(timeseries, timeseries, train_ratio=train_ratio)
    X_val, X_test, _, _ = timeseries_train_test_split(X_val, X_val, train_ratio=train_ratio)

    # create rolling window tensors
    X_train, y_train = create_rolling_window_ts(features=X_train, 
                                                target=X_train,
                                                num_timesteps_in=num_timesteps_in,
                                                num_timesteps_out=num_timesteps_out,
                                                fix_start=fix_start)
    X_val, y_val = create_rolling_window_ts(features=X_val, 
                                            target=X_val,
                                            num_timesteps_in=num_timesteps_in,
                                            num_timesteps_out=num_timesteps_out,
                                            fix_start=fix_start)
    X_test, y_test = create_rolling_window_ts(features=X_test, 
                                              target=X_test,
                                              num_timesteps_in=num_timesteps_in,
                                              num_timesteps_out=num_timesteps_out,
                                              fix_start=fix_start)

    # create loader
    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=train_shuffle, drop_last=True)
    val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)

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

    train_kld_loss_values = train_nll_losss_values = []
    val_kld_loss_values = val_nll_losss_values = []
    pbar = tqdm(range(n_epochs + 1), total=(n_epochs + 1))
    for epoch in pbar:
        
        # train vrnn
        train_kld_loss = train_nll_loss = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            X_device = X.to(device)

            # forward propagation
            kld_loss, nll_loss, enc, dec = model.forward(X_device)
            train_kld_loss += kld_loss
            train_nll_loss += nll_loss

            # aggregate loss function = KLdivergence - log-likelihood
            loss = (kld_loss + nll_loss)

            # back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # grad norm clipping
            # used to mitigate the problem of exploding gradients, specially when using RNNs
            nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                     max_norm=max_norm_clip)
        train_kld_loss_values.append(train_kld_loss / len(train_loader))
        train_nll_losss_values.append(train_nll_loss / len(train_loader))

        # evaluate model 
        model.eval()
        with torch.no_grad():
            val_kld_loss = val_nll_loss = 0
            for batch_idx, (X, y) in enumerate(val_loader):
                X_device = X.to(device)

                # forward propagation
                kld_loss, nll_loss, enc, dec = model.forward(X_device)
                val_kld_loss += kld_loss
                val_nll_loss += nll_loss
            val_kld_loss_values.append(val_kld_loss / len(val_loader))
            val_nll_losss_values.append(val_nll_loss / len(val_loader))
                
        pbar.set_description('Epoch: {} KLD Loss: {:.6f} ({:.6f}) NLL Loss: {:.6f} ({:.6f})'.format(epoch,
                                                                                                    (train_kld_loss / len(train_loader)).detach().item(),
                                                                                                    (val_kld_loss / len(val_loader)).detach().item(),
                                                                                                    (train_nll_loss / len(train_loader)).detach().item(),
                                                                                                    (val_nll_loss / len(val_loader)).detach().item()))
    
    with torch.no_grad():
        window_adj = (torch.ones(num_timesteps_in, X_test.shape[2]) * np.nan)

        # compute train weight predictions
        train_kld_loss, train_nll_loss, train_enc, train_dec = model.forward(X_train)
        y_train_pred = train_dec[0][:, -num_timesteps_out, :]
        y_train_pred = torch.concat((window_adj, y_train_pred) , dim=0)
        
        # compute val weight predictions
        val_kld_loss, val_nll_loss, val_enc, val_dec = model.forward(X_val)
        y_val_pred = val_dec[0][:, -num_timesteps_out, :]
        y_val_pred = torch.concat((window_adj, y_val_pred) , dim=0)

        # compute test weight predictions
        test_kld_loss, test_nll_loss, test_enc, test_dec = model.forward(X_test)
        y_test_pred = test_dec[0][:, -num_timesteps_out, :]
        y_test_pred = torch.concat((window_adj, y_test_pred) , dim=0)

    y = torch.tensor(timeseries.copy())
    y_pred = torch.concat((y_train_pred, y_val_pred, y_test_pred), dim=0)
    y_out = pd.DataFrame(torch.concat((y, y_pred), dim=1), columns=["y", "y_pred"])

    y_out.plot(secondary_y="y_pred")

    results = {

        "train_kld_loss": train_kld_loss_values,
        "train_nll_loss": train_nll_losss_values,
        "val_kld_loss": val_kld_loss_values,
        "val_nll_loss": val_nll_losss_values,
        "timeseries": y_out

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
    