import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import get_random_data_batch
from models.Scalers import Scalers

class Generator(nn.Module):
    def __init__(self, n_input, n_output) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_input, 256),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, self.n_output),
                    nn.Tanh()
                    )
        
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, n_input, n_output) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_input, 256),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, self.n_output),
                    nn.Sigmoid()
                    )
        
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class VanillaGAN(Scalers):
    def __init__(self,
                 n_input,
                 n_output,
                 n_epoch,
                 n_iter,
                 criterion,
                 batch_size_perc,
                 scaler_type):
        
        # TODO - have to implement scaler for mixed frequency data
        Scalers.__init__(self, scaler_type=scaler_type)

        # instantiate generator and discriminator
        self.generator = Generator(n_input=n_input, n_output=n_output)
        self.discriminator = Discriminator(n_input=n_input, n_output=1)

        self.n_epoch = n_epoch
        self.n_iter = n_iter
        self.batch_size_perc = batch_size_perc
        self.criterion = criterion

    def get_random_data_batch(self, data, batch_size):
        nrows = data.shape[0]

        bathc_idx = np.random.randint(low=0, high=nrows, size=batch_size)
        selected_data = data[bathc_idx, :]

        return torch.tensor(selected_data).float()

    def compute_discriminator_loss(self, true_data, fake_data, optimizer):
        
        # clean discriminator gradient
        optimizer.zero_grad()

        # run discriminator on real data
        predictions_real = self.discriminator.forward(x=true_data)

        # run discriminator on fake data
        prediction_fake = self.discriminator.forward(x=fake_data)

        # update discriminator loss
        fake_loss = self.criterion(prediction_fake, torch.zeros((self.batch_size, 1)))
        real_loss = self.criterion(predictions_real, torch.ones((self.batch_size, 1)))
        d_loss = fake_loss + real_loss
        d_loss.backward()
        optimizer.step()

        return d_loss
    
    def compute_generator_loss(self, fake_data, optimizer):

        # clean generator gradient

        # run discriminator
        prediction = self.discriminator.forward(x=fake_data)

        # compute generator loss - try to fool the discriminator
        g_loss = self.criterion(prediction, torch.ones((self.batch_size, 1)))
        g_loss.backward()
        optimizer.step()

        return g_loss

    def train(self, data, learning_rate):

        # apply data scaling
        self.fit(data)
        proc_data = self.transform(data)

        # util dims
        self.nrows = proc_data.shape[0]
        self.ncols = proc_data.shape[1]
        self.batch_size = int(self.nrows * self.batch_size_perc)

        # define optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        g_losses = []
        d_losses = []
        d_loss = g_loss = 0
        for epoch in range(self.n_epoch):
          for i in range(self.n_iter):
            batch_data = get_random_data_batch(data=proc_data, batch_size=self.batch_size)

            # generate fake data
            noise = torch.normal(mean=0, std=1, size=(self.batch_size, self.ncols))
            fake_data = self.generator.forward(x=noise).detach()

            # compute discriminator loss
            d_loss += self.compute_discriminator_loss(true_data=batch_data, fake_data=fake_data, optimizer=d_optimizer)

            # compute generator loss
            g_loss += self.compute_generator_loss(fake_data=fake_data, optimizer=g_optimizer)
        
          g_losses.append(g_loss / i)
          d_losses.append(d_loss / i)
          print('Epoch {}: g_loss: {:.8f} d_loss: {:.8f}\r'.format(epoch, g_loss / (i + 1), d_loss / (i + 1)))

        training_results = {
            "generator_loss": g_losses,
            "discriminator_loss": d_losses,
            }
        
        return training_results