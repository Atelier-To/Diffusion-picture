import torch
from diffusion_loader import train_loader, dataset, add_noise
from Unet_accelerate import Unet_ac
from torchvision import transforms
from datasets import load_dataset

add_noise()
Unet_ac(train_loader)

