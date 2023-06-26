import wandb
from IPython import display
from torch.utils.data import DataLoader
import torchvision
import torch
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import os
import pickle
import tensorflow as tf
max_values = [11055, 11311, 127, 127, 127]

"""##**Upload dataset and unzip**"""

cache_file = r"C:\Users\Pablo\Documents\MusikIA\Numpy\nuevas_matrices.pickle"
with open(cache_file, 'rb') as f:
    datos = np.array(pickle.load(f))

def load_data(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(num_samples)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    print('Dataset completo')

    return dataset

num_samples=16
input_dim = 10000 * 5
output_dim = 1
batch_size = 100
dataset = load_data(datos, batch_size=1000)

"""##**Define model**"""

class model(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


gan_model = model(input_dim, output_dim)
batch_size = 100

"""##**Make dataset iterable**"""

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)

"""##**Visualize dataset**"""

import matplotlib.pyplot as plt

# %matplotlib inline
plt.imshow(dataset[0][0], cmap='gray')

"""##**NETWORKS**"""


class DiscriminativeNet(torch.nn.Module):

    def __init__(self):
        super(DiscriminativeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024 * 558 * 544, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024 * 558 * 544)
        x = self.out(x)
        return x


class GenerativeNet(torch.nn.Module):

    def __init__(self):
        super(GenerativeNet, self).__init__()

        self.linear = torch.nn.Linear(100, 1024 * 558 * 544)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 558, 544)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)


# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)


# Create Network instances and init weights
generator = GenerativeNet()
generator.apply(init_weights)

discriminator = DiscriminativeNet()
discriminator.apply(init_weights)

# Enable cuda if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

"""##**Optimization**"""

# Optimizers
d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
loss = nn.BCELoss()

# Number of epochs
num_epochs = 200

"""##**Training**"""


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


"""##**Generate Samples For Testing**"""

def save_samples(epoch, samples):
    filename = f'generated_music/sample_epoch{epoch}.csv'
    np.savetxt(filename, samples, delimiter=',', fmt='%.4f')
test_noise = noise(num_samples)

"""##**Start training**"""

wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project="MusikAI_V4")

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)

        if epoch % 100 == 0:
            generated_data = generator(noise, training=False)
            generated_data = generated_data * max_values
            generated_data = generated_data.numpy().astype(int)
            save_samples(epoch, generated_data)

        # Registrar métricas en wandb
        wandb.log({'epoch': epoch, 'd_loss': d_error, 'g_loss': g_error})
        print(f"Epoch {epoch}: Discriminator loss = {d_error}, Generator loss = {g_error}")

generator.save(f"{wandb.run.dir}/generator.h5")

print('generando')