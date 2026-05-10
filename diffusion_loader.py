import torch
from datasets import load_dataset
from diffusers.utils import make_image_grid
from diffusers import DDPMScheduler
from torchvision import transforms

#参数设定
Batch_size = 16
IMAGE_SIZE = 64

dataset = load_dataset("data",split="train")

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=Batch_size,
    shuffle=True,
)

def transform(examples):
    images = [preprocess(image) for image in examples["image"]]
    return {"image": images}
dataset.set_transform(transform)

#增加噪音，生成图片
def add_noise():
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    clean_images = next(iter(train_loader))["image"]
    noise = torch.randn(clean_images.shape, device=clean_images.device)
    bs = clean_images.shape[0] #batch size
    timesteps = torch.arange(10, 170, 10, dtype=torch.int64)
    noisy_images = noise_scheduler.add_noise(clean_images,noise,timesteps)
    grid1 = make_image_grid([transforms.ToPILImage()(clean_image) for clean_image in clean_images],rows=4,cols=4)
    grid2 = make_image_grid([transforms.ToPILImage()(noisy_image) for noisy_image in noisy_images],rows=4,cols=4)
    grid1.save("example/clean.png")
    grid2.save("example/noisy.png")
