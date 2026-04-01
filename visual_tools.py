from IPython import display
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@torch.no_grad()
def visualize_latent_space(model, loader, label=None):
    model.eval()
    display.clear_output(wait=True)
    device = next(model.parameters()).device
    all_z, all_colors = [], []
    
    for x_in, _, colors in loader:
        z = model.encoder(x_in.to(device))
       
        if isinstance(z, tuple):
            z = z[0]
        all_z.append(z.cpu().numpy())
        all_colors.append(colors.numpy())

    all_z = np.concatenate(all_z, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    dim = all_z.shape[1]
    
    is_pca = False
    if dim > 3:
        scaler = StandardScaler()
        all_z_scaled = scaler.fit_transform(all_z)
        
        pca = PCA(n_components=2)
        all_z = pca.fit_transform(all_z_scaled)
        
        dim_to_plot = 2
        is_pca = True
    else:
        dim_to_plot = dim
    fig = plt.figure(figsize=(10, 7))
    is_rgb = (all_colors.ndim == 2 and all_colors.shape[1] == 3)
    scatter_kwargs = {'s': 5, 'alpha': 0.6}
    if not is_rgb:
        scatter_kwargs['cmap'] = 'hsv' 

    if dim_to_plot == 1:
        ax = fig.add_subplot(111)
        sc = ax.scatter(all_z[:, 0], np.zeros_like(all_z[:, 0]), c=all_colors, **scatter_kwargs)
        ax.set_yticks([])
        
    elif dim_to_plot == 2:
        ax = fig.add_subplot(111)
        sc = ax.scatter(all_z[:, 0], all_z[:, 1], c=all_colors, **scatter_kwargs)
        
    elif dim_to_plot == 3:
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(all_z[:, 0], all_z[:, 1], all_z[:, 2], c=all_colors, **scatter_kwargs)
        ax.view_init(elev=20, azim=160) 


    title = f"Латентное пространство (Dim: {all_z.shape[1]})"
    
        
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label=label)
    plt.show()



@torch.no_grad()
def visualize_mnist_latent(model, loader):
    model.eval()
    device = next(model.parameters()).device
    display.clear_output(wait=True)

    all_z = []
    all_labels = []
    for images, labels in loader:
        z = model.encoder(images.to(device)) 
        all_z.append(z.cpu().numpy())
        all_labels.append(labels.numpy())
        
    all_z = np.concatenate(all_z)
    all_labels = np.concatenate(all_labels)
    
    plt.figure(figsize=(10, 8))
   
    scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Цифра MNIST')
    plt.show()



@torch.no_grad()
def visualize_full_reconstruction(model, loader, colors=None):
    display.clear_output(wait=True)

    device = next(model.parameters()).device
    model.eval()
    x_recon, x_orig = [], []
    
    for data, _, _ in loader:
        x_recon.append(model(data.to(device)).cpu())
        x_orig.append(data.cpu())

    x_recon = torch.stack(x_recon)
    x_orig = torch.stack(x_orig)
    dim = x_orig.shape[-1]
    
    if dim == 784:
        fig, axes = plt.subplots(2, 10, figsize=(15, 4))
        for i in range(10):
            axes[0, i].imshow(x_orig[i].view(28, 28), cmap='gray')
            axes[1, i].imshow(x_recon[i].view(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        axes[0, 0].set_ylabel("Original", size='large')
        axes[1, 0].set_ylabel("Recon", size='large')
        plt.suptitle("Реконструкция изображений")

    elif dim == 3:
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(x_orig[:, 0], x_orig[:, 1], x_orig[:, 2], c=colors, cmap='viridis', s=10)
        ax1.set_title("Оригинал (3D)")
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(x_recon[:, 0], x_recon[:, 1], x_recon[:, 2], c=colors, cmap='viridis', s=10)
        ax2.set_title("Реконструкция (3D)")
        
        for ax in [ax1, ax2]: ax.view_init(elev=10, azim=-60)

    elif dim == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        ax1.scatter(x_orig[:, 0], x_orig[:, 1], c=colors, cmap='viridis', s=10)
        ax1.set_title("Оригинал (2D)")
        
        ax2.scatter(x_recon[:, 0], x_recon[:, 1], c=colors, cmap='viridis', s=10)
        ax2.set_title("Реконструкция (2D)")
        for ax in [ax1, ax2]: ax.grid(True, alpha=0.3)

    else:
        print(f"Неизвестная размерность данных: {dim}")
        return

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def visualize_sine_manifold(model, loader):
    display.clear_output(wait=True)
    device = next(model.parameters()).device
    model.eval()
    
    all_noisy = []
    for x_in, _, _ in loader:
        all_noisy.append(x_in.cpu())
    noisy = torch.cat(all_noisy).numpy()

    z_samples = model.encoder(torch.from_numpy(noisy).to(device))
    z_min, z_max = z_samples.min().item(), z_samples.max().item()
    z_grid = torch.linspace(z_min, z_max, 1000).view(-1, 1).to(device)
    manifold = model.decoder(z_grid).cpu().numpy()

    x_true = np.linspace(-np.pi, np.pi, 1000)
    y_true = np.sin(x_true)

    plt.figure(figsize=(12, 6))
    
    plt.plot(x_true, y_true, color='black', linestyle='--', alpha=0.4, lw=1.5, label='Ground Truth (Sine)')
    
    plt.scatter(noisy[:, 0], noisy[:, 1], color='blue', alpha=0.2, s=5, label='Noisy Samples')
    
    plt.plot(manifold[:, 0], manifold[:, 1], color='red', lw=3, label='Learned Manifold')
    plt.legend(loc='best', frameon=True)
    plt.xlim(-np.pi - 0.5, np.pi + 0.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.2)
    plt.show()


@torch.no_grad()
def visualize_std_analysis(model, loader, n=15):
    model.eval()
    device = next(model.parameters()).device
    all_z = []
    all_angles = []
    for batch in loader:
        x_in, _,  batch_angles = batch
        z = model.encoder(x_in.to(device)).cpu()
        all_z.append(z)
        all_angles.append(batch_angles)
    
    all_z = torch.cat(all_z, dim=0)
    all_angles = torch.cat(all_angles, dim=0).numpy()
    
    mu = all_z.mean(dim=0)
    std = all_z.std(dim=0) + 1e-8
    
    z_norm = (all_z - mu) / std

    limit = 2
    grid_x = np.linspace(-limit, limit, n)
    grid_y = np.linspace(-limit, limit, n)
    
    figure = np.zeros((28 * n, 28 * n))
    for i, y_val in enumerate(grid_y[::-1]):
        for j, x_val in enumerate(grid_x):
            z_sample_norm = torch.tensor([[x_val, y_val]], dtype=torch.float32)
            z_raw = (z_sample_norm * std + mu).to(device)
            img = model.decoder(z_raw).view(28, 28).cpu().numpy()
            figure[i * 28:(i+1)*28, j * 28:(j+1)*28] = img

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    sc = ax1.scatter(z_norm[:, 0], z_norm[:, 1], c=all_angles, cmap='hsv', s=5, alpha=0.5)
    ax1.set_title("Нормированное пространство")
    ax1.set_xlabel("Z1")
    ax1.set_ylabel("Z2")
    ax1.set_xlim(-limit-0.5, limit+0.5)
    ax1.set_ylim(-limit-0.5, limit+0.5)
    ax1.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax1, label='Угол')

    ax2.imshow(figure, cmap='gray', extent=[-limit, limit, -limit, limit])
    ax2.set_title(f"Сетка генерации {n}x{n}")

    plt.tight_layout()
    plt.show()

