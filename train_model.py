from torch.amp import autocast, GradScaler
import torch
import torch.nn.functional as F
import numpy as np
import os
from technical_func import set_random_seed
from data_file import TensorIndexDataset
from torch.utils.data import DataLoader



@torch.no_grad()
def update_teacher_model(student_model, teacher_model, alpha_teacher):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha_teacher).add_(student_param.data, alpha=1. - alpha_teacher)
    for tb, sb in zip(teacher_model.buffers(), student_model.buffers()):
        if tb.is_floating_point():
            tb.data.mul_(alpha_teacher).add_(sb.data, alpha=1. - alpha_teacher)
        else:
            tb.data.copy_(sb.data)

def sample_dirichlet_weights(batch_size, n_points, alpha, device):
    dist = torch.distributions.Dirichlet(
        torch.full((n_points,), float(alpha), device=device)
    )
    return dist.sample((batch_size,))  # [B, N]


def weighted_group_sum(groups, weights):
    view_shape = [groups.size(0), groups.size(1)] + [1] * (groups.dim() - 2)
    return (groups * weights.view(*view_shape)).sum(dim=1)


@torch.no_grad()
def batch_knn_groups(z, n_points, radius):
    B = z.size(0)
    z_flat = z.view(B, -1) 
    dist = torch.cdist(z_flat, z_flat) 
    mask = (dist <= radius).float() 
    eye = torch.eye(B, device=z.device)
    mask = torch.max(mask, eye)
    
    idx = torch.multinomial(mask, num_samples=n_points, replacement=True)
    
    return idx # [B, n_points]


def get_data_loader(batch_size, dataset, shuffle=True, drop_last=False):
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            drop_last=drop_last,
        )



@torch.no_grad()
def validate(model, loader, reg_mode="none", n_points=3, alpha_mix = 0.7, radius_encoder=1.0, radius_decoder=1.0):
    model.eval()
    device = next(model.parameters()).device
    metrics = {"loss": 0, "rec_loss": 0, "enc_loss" : 0, "dec_loss": 0}
    n_batches = 0

    for x_in, target, _ in loader:
        x_in, target = x_in.to(device), target.to(device)
        B = x_in.size(0)
        recon = model(x_in)
        rec_loss = F.mse_loss(recon, target)

        enc_loss = torch.tensor(0.0, device=device)
        dec_loss = torch.tensor(0.0, device=device)

        if reg_mode in ("encoder", "both"):
            idx = batch_knn_groups(x_in, n_points, radius_encoder)
            g = x_in[idx]
            z_g = model.encoder(g.flatten(0, 1)).view(B, n_points, -1)
            lam = sample_dirichlet_weights(B, n_points, alpha_mix, device)
            
            z_pred = model.encoder(weighted_group_sum(g, lam))
            z_tar = weighted_group_sum(z_g, lam)
            enc_loss = F.mse_loss(z_pred, z_tar)

        if reg_mode in ("decoder", "both"):
            z = model.encoder(x_in)
            idx = batch_knn_groups(z, n_points, radius_decoder)
            g_in, g_tar = x_in[idx], target[idx]
            z_g = model.encoder(g_in.flatten(0, 1)).view(B, n_points, -1)
            lam = sample_dirichlet_weights(B, n_points, alpha_mix, device)
            
            x_pred = model.decoder(weighted_group_sum(z_g, lam))
            x_tar = weighted_group_sum(g_tar, lam)
            dec_loss = F.mse_loss(x_pred, x_tar)

        metrics["loss"] += (rec_loss + enc_loss + dec_loss).item()
        metrics["rec_loss"] += rec_loss.item()
        metrics["enc_loss"] += enc_loss.item()
        metrics["dec_loss"] += dec_loss.item()   
        n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}



def train_one_epoch(
    student,
    teacher,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    reg_mode="none",
    n_points=3,
    alpha_mix=0.7,
    lambda_enc=1.0,
    lambda_dec=1.0,
    radius_encoder=None,
    radius_decoder=None,
    teacher_momentum=0.99,
    rampup_epochs=5,
    epoch_idx=0,
):
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_rec = 0.0
    
    total_enc = 0.0
    total_dec = 0.0
    n_batches = 0

    ramp = min(1.0, (epoch_idx + 1) / max(rampup_epochs, 1))

    for x_in, target, _ in train_loader:
        x_in = x_in.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast("cuda"):
            recon = student(x_in)
            rec_loss = F.mse_loss(recon, target)

            enc_loss = torch.tensor(0.0, device=device)
            dec_loss = torch.tensor(0.0, device=device)

            if reg_mode != "none":
                if reg_mode in ("encoder", "both"):
                    group_idx = batch_knn_groups(
                        x_in,
                        n_points=n_points, radius=radius_encoder
                    )
                    noisy_groups = x_in[group_idx]
                    B, N = group_idx.shape
                    with torch.no_grad():
                        z_groups_teacher = teacher.encoder(
                            noisy_groups.flatten(0, 1)
                        ).view(B, N, -1)

                    lam = sample_dirichlet_weights(B, N, alpha_mix, device)
                    x_mix = weighted_group_sum(noisy_groups, lam)
                    z_target = weighted_group_sum(z_groups_teacher, lam)
                    z_pred = student.encoder(x_mix)
                    enc_loss = F.mse_loss(z_pred, z_target)

                if reg_mode in ("decoder", "both"):
                    with torch.no_grad():
                        z_teacher = teacher.encoder(x_in)

                    group_idx = batch_knn_groups(
                        z_teacher,
                        n_points=n_points, radius=radius_decoder
                    )

                    noisy_groups = x_in[group_idx]   # [B, N, ...]
                    clean_groups = target[group_idx]   # [B, N, ...]

                    B, N = group_idx.shape
                    with torch.no_grad():
                        z_groups_teacher = teacher.encoder(noisy_groups.flatten(0, 1)).view(B, N, -1)

                    lam = sample_dirichlet_weights(B, N, alpha_mix, device)
                    z_mix = weighted_group_sum(z_groups_teacher, lam)
                    x_target = weighted_group_sum(clean_groups, lam)
                    x_pred = student.decoder(z_mix)
                    dec_loss = F.mse_loss(x_pred, x_target)

            loss = rec_loss + ramp * (lambda_enc * enc_loss + lambda_dec * dec_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        update_teacher_model(student, teacher, alpha_teacher=teacher_momentum)

        total_loss += loss.detach().item()
        total_rec += rec_loss.detach().item()
        total_enc += enc_loss.detach().item()
        total_dec += dec_loss.detach().item()
        n_batches += 1
    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / n_batches,
        "rec_loss": total_rec / n_batches,
        "enc_loss": total_enc / n_batches,
        "dec_loss": total_dec / n_batches,
    }


def fit_model(
    student,
    teacher,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    reg_mode="none",
    n_points=3,
    alpha_mix=0.7,
    lambda_enc=1.0,
    lambda_dec=1.0,
    radius_encoder=None,
    radius_decoder=None,
    teacher_momentum=0.99,
    verbose_latent=False,
    func_latent=None,
    verbose_recon=False,
    func_recon=None,
    period=1000,
):
    device = "cuda:0"
    
    teacher.load_state_dict(student.state_dict())
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler()
    history_train = []
    history_val = []

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            reg_mode=reg_mode,
            n_points=n_points,
            alpha_mix=alpha_mix,
            lambda_enc=lambda_enc,
            lambda_dec=lambda_dec,
            radius_encoder=radius_encoder,
            radius_decoder=radius_decoder,
            teacher_momentum=teacher_momentum,
            rampup_epochs=max(3, epochs // 10),
            epoch_idx=epoch,
        ) 

        

        if (verbose_latent and (epoch % period == 0 or epoch == epochs - 1)):
            func_latent(student, train_loader)
        if ((epoch % period == 0 or epoch == epochs - 1)):
            if (verbose_recon):
                func_recon(student, val_loader)
            print("Train metrics:", train_metrics, "cur epoch:", epoch)
        history_train.append(train_metrics)
        if (val_loader is not None):
            val_metrics = validate(student, val_loader, reg_mode=reg_mode, n_points=n_points, alpha_mix=alpha_mix, radius_encoder=radius_encoder, radius_decoder=radius_decoder)
            history_val.append(val_metrics)
            if ((epoch % period == 0 or epoch == epochs - 1)):
                print("Val metrics:", val_metrics, "cur epoch:", epoch)

            

        
    return student, teacher, history_train, history_val