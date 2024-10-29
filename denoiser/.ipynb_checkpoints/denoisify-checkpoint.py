import numpy as np
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.init as init

#load the model
torch.manual_seed(999)
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(DenoisingAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size).double(),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size).double(),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size).double(),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size).double(),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size).double()
        )

    
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size).double(),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size).double(),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size).double(),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size).double(),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size).double()
        )


    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
model = DenoisingAutoencoder(input_size, hidden_size, latent_size).to(device)
model.load_state_dict(torch.load('denoiser_numtraj=2000_noise=1.0.pth'))
model.eval()

def denoise(data):
    with torch.no_grad():
        denoised_dataS = model(torch.tensor(data).double())
    return denoised_dataS.cpu().detach().numpy()