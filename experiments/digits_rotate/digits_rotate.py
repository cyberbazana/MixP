from torchvision import datasets as tv_datasets
from PIL import Image
from data_file import TensorIndexDataset
import torch
import numpy as np

def _to_tensor(img):
    arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
    return arr.unsqueeze(0)

def make_rotated_versions(img, num_rotate, rotate_range):
    angles = np.linspace(0.0, rotate_range, num_rotate, endpoint=False, dtype=np.float32)
    outs = []
    for angle in angles:
        rotated = img.rotate(float(angle), resample=Image.BILINEAR, fillcolor=0)
        outs.append(_to_tensor(rotated))
    return torch.stack(outs, dim=0)


def _load_mnist(root: str = "./data", train: bool = True):
    if tv_datasets is not None:
        return tv_datasets.MNIST(root=root, train=train, download=True)


def _get_digit_images(dataset, digit, count = None):
    imgs = []
    for img, label in dataset:
        if int(label) == int(digit):
            imgs.append(img)
            if count is not None and len(imgs) >= count:
                break
    return imgs

def flatten_images(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1)

def build_rotating_eight_experiment(
    root: str = "./data",
    num_base_images: int = 100,
    num_rotate: int = 100,
    angle_step: float = 3.6,
):
    ds = _load_mnist(root=root, train=True)
    base_imgs = _get_digit_images(ds, digit=8, count=num_base_images)
    rotate_range = num_rotate * angle_step
    
    single_seq_angles = np.linspace(0.0, rotate_range, num_rotate, endpoint=False, dtype=np.float32)
    all_angles = np.tile(single_seq_angles, num_base_images)
    
    all_imgs = [make_rotated_versions(img, num_rotate=num_rotate, rotate_range=rotate_range) for img in base_imgs]
    x = torch.cat(all_imgs, dim=0)
    x_flat = flatten_images(x)
    
    return TensorIndexDataset(x_flat, x_flat.clone(), torch.from_numpy(all_angles)), x, {
        "digit": 8,
        "num_base_images": num_base_images,
        "num_rotate": num_rotate,
        "angle_step": angle_step,
        "rotate_range": rotate_range,
    }

def build_rotated_digit_sequence_experiment(
    root = "./data",
    digit = 3,
    num_rotate = 20,
    rotate_range = 360.0,
    n_different=1,
):
    ds = _load_mnist(root=root, train=True)
    imgs = _get_digit_images(ds, digit=digit, count=300)[100:100+n_different]
    
    all_raw_images = []
    angles = np.linspace(0.0, rotate_range, num_rotate, endpoint=False, dtype=np.float32)

    for img in imgs:
        x_rot = make_rotated_versions(img, num_rotate=num_rotate, rotate_range=rotate_range)
        x_flat = flatten_images(x_rot)
        all_raw_images.append(x_flat)

    return TensorIndexDataset(torch.stack(all_raw_images).view(-1, 784), torch.stack(all_raw_images).view(-1, 784).clone(), torch.from_numpy(angles).repeat(n_different).view(-1, 1)), torch.stack(all_raw_images), {
        "digit": digit,
        "num_rotate": num_rotate,
        "rotate_range": rotate_range,
        "n_trajectories": n_different
    }