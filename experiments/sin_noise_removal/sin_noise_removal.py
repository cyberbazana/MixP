import torch
from IPython import display
import numpy as np

def generate_noisy_sine_data(n_samples=50, noise_std=0.2):
    x = (torch.rand(n_samples) * 2 - 1) * torch.pi 
    
    y = torch.sin(x)
    clean_data = torch.stack([x, y], dim=1)
    
    noisy_data = clean_data + torch.randn_like(clean_data) * noise_std
    
    return noisy_data, clean_data


def evaluate_mse_on_sin(model, n_points=10000):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x_test = torch.linspace(-np.pi, np.pi, n_points).view(-1, 1).to(device)
        y_true = torch.sin(x_test)
        test_data = torch.cat([x_test, y_true], dim=1)
        latent = model.encoder(test_data)
        reconstructed_data = model.decoder(latent)
        mse_value = torch.mean((reconstructed_data[:, 1] - y_true.squeeze())**2).item()
        
    print(f"Test MSE (10k points): {mse_value:.6f}")
    return mse_value