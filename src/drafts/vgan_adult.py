import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from settings import INPUTS_PATH, OUTPUTS_PATH
from models.VanillaGAN import VanillaGAN
from utils.conn_data import save_pickle

plt.style.use("bmh")

data = pd.read_csv(os.path.join(INPUTS_PATH, "adult.csv"))
data = data.dropna()
data = data[data != "?"]
data = torch.tensor(data[["hours.per.week", "education.num"]].to_numpy())
target_name = "education"

# training hyperparameters
criterion = nn.BCEWithLogitsLoss(reduction="sum")
d_learning_rate, g_learning_rate, n_epoch, batch_size = 0.05, 0.005, 100, 100

# model hyperparameters
n_input = data.shape[1]
n_output = data.shape[1]
model_name = "vgan"

# check if output dir exists
if not os.path.isdir(os.path.join(OUTPUTS_PATH, model_name)):
    os.mkdir(os.path.join(OUTPUTS_PATH, model_name))

# train vanilla GAN
vgan = VanillaGAN(n_input=n_input,
                  n_output=n_output,
                  n_epoch=n_epoch,
                  criterion=criterion,
                  batch_size=batch_size,
                  scaler_type="min_max_scaler")
results = vgan.train(data=data, d_learning_rate=d_learning_rate, g_learning_rate=g_learning_rate)

# check if output dir exists
if not os.path.isdir(os.path.join(OUTPUTS_PATH, model_name, target_name)):
    os.mkdir(os.path.join(OUTPUTS_PATH, model_name, target_name))

# save training results
save_pickle(path=os.path.join(OUTPUTS_PATH, model_name, target_name, "training_results.pickle"),
            obj=results)

# plot generator/discriminator training losses
plt.plot(results["generator_loss"], label='Generator')
plt.plot(results["discriminator_loss"], label='Discriminator')
plt.legend()
plt.savefig(os.path.join(OUTPUTS_PATH, model_name, target_name, "training_losses.png"))
plt.close()

# plot true samples vs fake samples
plt.scatter(results["generator_samples"][-1][:,0], results["generator_samples"][-1][:,1], label='generated', color="blue")
plt.scatter(results["data"][:,0], results["data"][:,1], label='Real', color="red")
plt.legend()
plt.savefig(os.path.join(OUTPUTS_PATH, model_name, target_name, "true_vs_fake_samples.png"))