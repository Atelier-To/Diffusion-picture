import torch
from diffusers import UNet2DModel, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from diffusion_loader import IMAGE_SIZE, Batch_size
import os
from accelerate import Accelerator
from diffusers import DDPMScheduler
import torch.nn.functional as F
from tqdm.auto import tqdm
#设置加速器对象
def Unet_ac(train_loader):
    model = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # 定义训练参数
    NUM_EPOCH = 3
    LR = 1e-4
    LR_WARMUP_STEPS = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=NUM_EPOCH * len(train_loader),
    )

    # 模型保存路径
    model_save_dir = "anime-64"
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="tensorboard",
        project_dir=os.path.join(model_save_dir, "logs"),
    )

    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)
    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    global_step = 0
    for epoch in range(NUM_EPOCH):
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCH}")
        for step, batch in enumerate(train_loader):
            clean_images = batch["image"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]  # batch size
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device, dtype=torch.int64)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.detach().item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            accelerator.log(logs, step=global_step)
            global_step += 1
        progress_bar.close()

        # 每个 epoch 结束后保存模型
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                model_save_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            noise_scheduler.save_pretrained(model_save_dir)
            print(f"Epoch {epoch+1}/{NUM_EPOCH} — model saved to {model_save_dir}")

    random_seed = 42
    SAVE_ARTIFACT_EPOCH = 1
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model)
                            ,scheduler=noise_scheduler,)
    if(epoch +1)% SAVE_ARTIFACT_EPOCH == 0 or epoch == NUM_EPOCH - 1:
        images = pipeline(
            batch_size = Batch_size,
            generator = torch.manual_seed(random_seed),
        ).images
        image_grid = make_image_grid(images,rows=4,cols=4)

        #保存图片
        test_dir = os.path.join(model_save_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")
        pipeline.save_pretrained(model_save_dir)