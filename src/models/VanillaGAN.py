import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.data_utils import get_random_data_batch
from models.Scalers import Scalers

class Generator(nn.Module):
    def __init__(self, n_input, n_output) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.fc0 = nn.Sequential(nn.Linear(2, 2))
        
    def forward(self, x):
        x = self.fc0(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, n_input, n_output) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.fc0 = nn.Sequential( 
            nn.Linear(2, 5), 
            nn.Tanh(), 
            nn.Linear(5, 3), 
            nn.Tanh(), 
            nn.Linear(3, 1))
        
    def forward(self, x):
        x = self.fc0(x)
        return x
    
class VanillaGAN(Scalers):
    def __init__(self,
                 n_input,
                 n_output,
                 n_epoch,
                 criterion,
                 batch_size,
                 scaler_type):
        
        # TODO - have to implement scaler for mixed frequency data
        Scalers.__init__(self, scaler_type=scaler_type)

        # instantiate generator and discriminator
        self.generator = Generator(n_input=n_input, n_output=n_output)
        self.discriminator = Discriminator(n_input=n_input, n_output=1)

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.criterion = criterion

    def get_random_data_batch(self, data, batch_size):
        nrows = data.shape[0]

        bathc_idx = np.random.randint(low=0, high=nrows, size=batch_size)
        selected_data = data[bathc_idx, :]

        return torch.tensor(selected_data).float()

    def compute_discriminator_loss(self, X, optimizer):

        # clean discriminator gradient
        optimizer.zero_grad()

        ## sample fake data
        Z = torch.normal(mean=0, std=1, size=(self.batch_size, self.ncols))
        fake_X = self.generator.forward(x=Z).detach()

        # run discriminator on fake data
        prediction_fake = self.discriminator.forward(x=fake_X)

        # run discriminator on real data
        predictions_real = self.discriminator.forward(x=X)

        # update discriminator loss
        # maximize mean_i[log(1 - D(x_i)) + log(1 - D(G(z_i)))]
        fake_loss = self.criterion(prediction_fake, torch.zeros((self.batch_size, 1)))
        real_loss = self.criterion(predictions_real, torch.ones((self.batch_size, 1)))
        d_loss = (fake_loss + real_loss) / 2
        d_loss.backward()
        optimizer.step()

        return d_loss
    
    def compute_generator_loss(self, optimizer):

        # clean generator gradient
        optimizer.zero_grad()

        ## sample fake data
        Z = torch.normal(mean=0, std=1, size=(self.batch_size, self.ncols))
        fake_X = self.generator.forward(x=Z)

        # run discriminator
        prediction = self.discriminator.forward(x=fake_X)

        # compute generator loss - try to fool the discriminator
        # instead of minimizing mean_i[log(1 - D(G(z_i)))] we are maximizing mean_i[log(D(G(z_i)))]
        g_loss = self.criterion(prediction, torch.ones((self.batch_size, 1)))
        g_loss.backward()
        optimizer.step()

        return g_loss

    def train(self,
              data,
              d_learning_rate,
              g_learning_rate,
              gaussian_initialization=False,
              apply_scaling=False):

        # initialize nn weights by sampling a gaussian dist 
        if gaussian_initialization:
            for w in self.discriminator.parameters():
                nn.init.normal_(w, 0, 0.02)
            for w in self.generator.parameters():
                nn.init.normal_(w, 0, 0.02)

        # apply data scaling
        if apply_scaling:
            self.fit(data)
            proc_data = self.transform(data)
        else:
            proc_data = data

        # build data loader
        proc_data_loader = DataLoader(proc_data, shuffle=True, batch_size=self.batch_size, drop_last=True)

        # util dims
        self.nrows = proc_data.shape[0]
        self.ncols = proc_data.shape[1]

        # define optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=g_learning_rate)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=d_learning_rate)

        g_samples = torch.zeros((self.n_epoch, self.nrows, self.ncols))
        g_losses = []
        d_losses = []
        for epoch in range(self.n_epoch):
            m = d_loss = g_loss = 0
            for X in proc_data_loader:
                
                m  += X.shape[0]

                ## compute discriminator loss
                d_loss += self.compute_discriminator_loss(X=X.float(), optimizer=d_optimizer)

                ## compute generator loss
                g_loss += self.compute_generator_loss(optimizer=g_optimizer)

            # visualize generator
            Z = torch.normal(mean=0, std=1, size=(self.nrows, self.ncols))
            fake_X = self.generator.forward(x=Z).detach()
            # g_samples[epoch, :, :] = torch.tensor(self.inverse_transform(fake_X)).float()
            g_samples[epoch, :, :] = torch.tensor(fake_X).float()

            # compute expected loss
            avg_d_loss, avg_g_loss = d_loss / m, g_loss / m
            g_losses.append(avg_g_loss.item())
            d_losses.append(avg_d_loss.item())

            # print expected loss
            print('Epoch {}: g_loss: {:.8f} d_loss: {:.8f}\r'.format(epoch, avg_d_loss, avg_g_loss))

        training_results = {
            "data": data,
            "generator_samples": g_samples,
            "generator_loss": g_losses,
            "discriminator_loss": d_losses,
            }

        return training_results