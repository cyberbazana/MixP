import torch
import numpy as np
import matplotlib.pyplot as plt


def get_manual_2d_spiral(n_samples=2000, noise=0.0):
    theta = 2 * np.pi * torch.rand(n_samples)
    r = 0.1 + 0.9 * (theta / (2 * np.pi))
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    X_2d = torch.stack([x, z], dim=1)
    if noise > 0:
        X_2d += noise * torch.randn_like(X_2d)
    return X_2d, theta

@torch.no_grad()
def visualize_learned_manifold(model, data_clear, data_noisy, n_grid=1000):
    model.eval()
    device = next(model.parameters()).device
    data_clear = data_clear.to(device)
    data_noisy = data_noisy.to(device)
    z_real = model.encoder(data_clear)
    z_min, z_max = z_real.min().item(), z_real.max().item()
    z_grid = torch.linspace(z_min, z_max, n_grid).view(-1, 1).to(device)
    manifold_path = model.decoder(z_grid).cpu()
    plt.figure(figsize=(10, 10))
    
    
    plt.scatter(data_clear.cpu()[:, 0], data_clear.cpu()[:, 1], 
                c='lightgray', s=10, alpha=0.4, label='True Data')
    plt.scatter(data_noisy.cpu()[:, 0], data_noisy.cpu()[:, 1], 
                c='blue', s=10, alpha=0.4, label='Train Data')

    plt.plot(manifold_path[:, 0], manifold_path[:, 1], 
             color='red', lw=3, label='Learned 1D Manifold')
    
    plt.title("Выученная структура спирали")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.axis('equal') 
    plt.show()
