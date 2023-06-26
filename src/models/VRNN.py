import sys
import os
import torch
import torch.nn as nn
import pandas as pd

import math

# temporally add repo to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from utils.conn_data import find_gpu_device

"""
Implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
"""

# changing device
device = torch.device(find_gpu_device())
EPS = torch.finfo(torch.float).eps # numerical logs

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        # self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())

        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)


    def forward(self, x):

        all_enc_mean = all_enc_std = torch.zeros(x.size(0), x.size(1), self.z_dim, device=device)
        all_dec_mean = all_dec_std = torch.zeros(x.size(0), x.size(1), self.x_dim, device=device)
        kld_loss = 0
        nll_loss = 0

        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=device)
        for t in range(x.size(0)):
            # (0) prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            phi_x_t = self.phi_x(x[t])

            # (1.1) encoder: p(z_t|x_t) 
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            # (1.2) sampling and reparameterization: p(z_t|x_t) 
            z_t = self._reparametrization_trick(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # (2) decoder: p(x_t|z_t)
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # (3) recurrence: h_t := h = f\theta(phi_x_t, phi_z_t, h)
            # NOTE: GRU parameters are becoming nan after epoch=6 and batch_idx=6. Solved by lowering the learning rate.
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # (4) computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            # nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_mean[t, :, :] = enc_mean_t
            all_enc_std[t, :, :] = enc_std_t 
            all_dec_mean[t, :, :] = dec_mean_t 
            all_dec_std[t, :, :] = dec_std_t

        return kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std)


    def _reparametrization_trick(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))

DEBUG = True

if __name__ == "__main__":
    
    if DEBUG:
        import os
        import sys
        import matplotlib.pyplot as plt
        from time import time
        import torch.utils.data as torchdata

        # temporally add repo to path
        sys.path.append(os.path.join(os.getcwd(), "src"))

        from utils.conn_data import load_data, save_pickle
        from utils.data_utils import create_ts_prediction_data

        # load toy data
        df = load_data(dataset_name="fredmd_raw_df")
        # cpi all items yoy
        timeseries = (df[["CPIAUCSL"]].pct_change(12) * 100).dropna().values.astype('float32')
        # timeseries = (df[["CPIAUCSL", "GS1"]].pct_change(12) * 100).dropna().values.astype('float32')
        timeseries = (timeseries - timeseries.min()) / (timeseries.max() - timeseries.min())

        ## hyperparameters ##
        seed = 199402
        seq_length = 40
        batch_size = 10
        train_size_perc = 0.6
        train_size = int(timeseries.shape[0] * train_size_perc)
        learning_rate = 1e-6
        max_norm_clip = 10
        n_epochs = 100
        print_every = 10

        x_dim = timeseries.shape[1] # number of time series in the dataset
        h_dim = 5 # size of the rnn latent space
        z_dim = 5 # size of the vae latent space
        n_layers =  1
        mse_loss = nn.MSELoss()

        model_name = "vrnn"

        # build sequence prediction data
        X = create_ts_prediction_data(timeseries, seq_length)

        # manual seed
        torch.manual_seed(seed)
        plt.ion()

        # dataset loaders
        X_train = X[0:train_size]
        X_test = X[train_size:]
        train_loader = torchdata.DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torchdata.DataLoader(X_test, batch_size=batch_size, shuffle=False, drop_last=True)

        # define model
        model = VRNN(x_dim, h_dim, z_dim, n_layers)
        model = model.to(device)

        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # training and evaluation slots
        results = {}
        train_losses = {}
        train_enc_dec_y = torch.zeros(n_epochs + 1, batch_size * seq_length, 1 + (2 * z_dim) + (2 * x_dim), device=device)
        test_losses = {}
        test_enc_dec_y = torch.zeros(int(X_test.shape[0] / batch_size), batch_size * seq_length, 1 + (2 * z_dim) + (2 * x_dim), device=device)

        # training vrnn
        init = time()
        model.train()
        for epoch in range(n_epochs + 1):

            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                device_data = data.to(device)

                # forward propagation
                optimizer.zero_grad()
                kld_loss, nll_loss, enc, dec = model(device_data)

                # aggregate loss function = KLdivergence - log-likelihood
                loss = kld_loss + nll_loss

                # back propagation
                loss.backward()
                optimizer.step()

                # grad norm clipping
                # used to mitigate the problem of exploding gradients, specially when using RNNs
                nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                         max_norm=max_norm_clip)

                # aggregate loss
                train_loss += loss.item()

            # save encoder outputs
            enc_mean = enc[0].cpu().detach()
            enc_sd = enc[1].cpu().detach()
            enc_mean = enc_mean.reshape(enc_mean.shape[0] * enc_mean.shape[1], enc_mean.shape[2])
            enc_sd = enc_sd.reshape(enc_sd.shape[0] * enc_sd.shape[1], enc_sd.shape[2])

            # save decoder outputs
            dec_mean = dec[0].cpu().detach()
            dec_sd = dec[1].cpu().detach()
            dec_mean = dec_mean.reshape(dec_mean.shape[0] * dec_mean.shape[1], dec_mean.shape[2])
            dec_sd = dec_sd.reshape(dec_sd.shape[0] * dec_sd.shape[1], dec_sd.shape[2])

            # save true and mse
            true = device_data.cpu().detach()
            true = true.reshape(true.shape[0] * true.shape[1], true.shape[2])
            mse = mse_loss(dec_mean, true).item()

            all = torch.cat((true, enc_mean, enc_sd, dec_mean, dec_sd), dim=1)

            train_enc_dec_y[epoch, :, :] = all

            avg_kld = (kld_loss / batch_size).cpu().detach().item()
            avg_nll = (nll_loss / batch_size).cpu().detach().item()

            # save losses
            train_losses[epoch] = {"kld": avg_kld,
                                   "nll": avg_nll,
                                   "mse": mse}
            
            # printing
            if epoch % print_every == 0:
                print('Train Epoch: {} KLD Loss: {:.6f} \t NLL Loss: {:.6f} \t MSE: {:.6f}'.format(epoch,
                                                                                                   avg_kld,
                                                                                                   avg_nll,
                                                                                                   mse))
        
        # aggregate training results
        results["train"] = {"eval_metrics": pd.DataFrame(train_losses).T,
                            "outputs": train_enc_dec_y}

        # test vrnn
        model.eval()
        test_loss = 0
        for batch_idx, test_data in enumerate(test_loader):
            device_test_data = test_data.to(device)

            # forward propagation
            optimizer.zero_grad()
            kld_loss, nll_loss, enc, dec = model(device_test_data)

            # aggregate loss function = KLdivergence - log-likelihood
            loss = kld_loss + nll_loss

            # back propagation
            loss.backward()
            optimizer.step()

            # grad norm clipping
            # used to mitigate the problem of exploding gradients, specially when using RNNs
            nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                        max_norm=max_norm_clip)

            # aggregate loss
            test_loss += loss.item()

            # save encoder outputs
            enc_mean = enc[0].cpu().detach()
            enc_sd = enc[1].cpu().detach()
            enc_mean = enc_mean.reshape(enc_mean.shape[0] * enc_mean.shape[1], enc_mean.shape[2])
            enc_sd = enc_sd.reshape(enc_sd.shape[0] * enc_sd.shape[1], enc_sd.shape[2])

            # save decoder outputs
            dec_mean = dec[0].cpu().detach()
            dec_sd = dec[1].cpu().detach()
            dec_mean = dec_mean.reshape(dec_mean.shape[0] * dec_mean.shape[1], dec_mean.shape[2])
            dec_sd = dec_sd.reshape(dec_sd.shape[0] * dec_sd.shape[1], dec_sd.shape[2])

            # save true and mse
            true = device_data.cpu().detach()
            true = true.reshape(true.shape[0] * true.shape[1], true.shape[2])
            mse = mse_loss(dec_mean, true).item()

            all = torch.cat((true, enc_mean, enc_sd, dec_mean, dec_sd), dim=1)

            test_enc_dec_y[batch_idx, :, :] = all

            avg_kld = (kld_loss / batch_size).cpu().detach().item()
            avg_nll = (nll_loss / batch_size).cpu().detach().item()

            # save losses
            test_losses[batch_idx] = {"kld": avg_kld,
                                  "nll": avg_nll,
                                  "mse": mse}
            
            print('Test Batch: {} \t KLD Loss: {:.6f} \t NLL Loss: {:.6f} \t MSE: {:.6f}'.format(batch_idx,
                                                                                                 kld_loss / batch_size,
                                                                                                 nll_loss / batch_size,
                                                                                                 mse))
            
        # aggregate training results
        results["test"] = {"eval_metrics": pd.DataFrame(test_losses).T,
                           "outputs": test_enc_dec_y}
        
        results["model"] = model.state_dict()
        
        output_path = os.path.join(os.getcwd(),
                                   "src",
                                   "data",
                                   "outputs",
                                   model_name)
        output_name = "{model_name}_{seq_length}_{epochs}_{batch_size}_{h_dim}_{z_dim}_{n_layers}.pt".format(model_name=model_name,
                                                                                                             seq_length=seq_length,
                                                                                                             batch_size=batch_size,
                                                                                                             epochs=n_epochs,
                                                                                                             h_dim=h_dim,
                                                                                                             z_dim=z_dim,
                                                                                                             n_layers=n_layers)
        torch.save(results, os.path.join(output_path, output_name))
        