import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from settings import INPUTS_PATH
from models.VanillaGAN import VanillaGAN

plt.style.use("bmh")

# read dataset and dropna 
data = pd.read_csv(os.path.join(INPUTS_PATH, "adult.csv"))
data = data[data != "?"].dropna()

num_cols = ['age', 'education.num', 'hours.per.week']
data = data[num_cols]

# training hyperparameters
learning_rate = 2e-4
criterion = F.binary_cross_entropy
n_epoch = 500
n_iter = 1
batch_size_perc = 0.2

# model hyperparameters
n_input = data.shape[1]
n_output = data.shape[1]

# train vanilla GAN
vgan = VanillaGAN(n_input=n_input,
                  n_output=n_output,
                  n_epoch=n_epoch,
                  n_iter=n_iter,
                  criterion=criterion,
                  batch_size_perc=batch_size_perc,
                  scaler_type="min_max_scaler")
results = vgan.train(data=data, learning_rate=learning_rate)

a = 1
