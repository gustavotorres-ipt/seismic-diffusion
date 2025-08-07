import os
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.transforms.functional import to_pil_image
from torch import bfloat16, nn
import torch.nn.functional as F
from tqdm import tqdm
from model_loader import load_clip_model, EmbedProjector
from dataset import load_datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from transformers import AutoTokenizer

DEVICE = "cuda"
NUM_EPOCHS = 20
BATCH_SIZE = 8
CHECKPOINT_DIR = "."
CHECKPOINT_NAME = 'checkpoint_diffusion.pt'
BEST_DIFFUSION_MODEL = "best_diffusion_model.pt"
BEST_PRIOR_MODEL = "best_prior_model.pt"
CLIP_WEIGHTS_FILE = "customized_clip.pth"

def load_checkpoint_if_available():
    files_dir = os.listdir(".")
    if CHECKPOINT_NAME in files_dir:
        return CHECKPOINT_NAME

    return None

# Load trained CLIP Model
custom_clip_model, preprocessor = load_clip_model()
custom_clip_model.to(DEVICE)

custom_clip_model.load_state_dict(torch.load(CLIP_WEIGHTS_FILE))

projector = EmbedProjector().to("cuda")

# Create dataloader to load images
train_dataset, val_dataset, test_dataset = load_datasets(preprocessor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
).to(DEVICE)

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae").to(DEVICE)

scheduler = DDIMScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
)
scheduler.set_timesteps(50)

unet.train()
projector.train()
vae.eval()  # frozen
optimizer = torch.optim.AdamW(
    list(unet.parameters()) + list(projector.parameters()), lr=5e-6
)

start_epoch = 0
latest_ckpt = load_checkpoint_if_available()

if latest_ckpt:
    print(f"Resuming from checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
    unet.load_state_dict(checkpoint["model_state"])
    projector.load_state_dict(checkpoint["projector_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    print(f"Resumed from epoch {start_epoch}")

else:
    print("No checkpoint found. Starting from scratch.")


best_avg_loss = float("inf")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

for epoch in range(start_epoch, NUM_EPOCHS):
    total_loss = 0
    print(f"----- EPOCH {epoch} -----")

    for images, captions in tqdm(train_loader):

        with torch.no_grad():
            tokenized_texts = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt",
                max_length=128
            ).to(DEVICE)

            images = images.to(DEVICE) # (B, 3, 256, 256), [0,1] range

            # Forward pass: compute image and text features
            embed_512 = custom_clip_model.encode_image(images)
            embed_512 = embed_512.to(DEVICE)   # (B, 512)

            images = resize(images, (256, 256))

            latents = vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215  # (B, 4, 32, 32)

            # 2. Sample random noise & timestep
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (latents.size(0),), device=latents.device
            ).long()

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # 3. Project embedding and predict noise
        embed_768 = projector(embed_512)           # (B, 1, 768)
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=embed_768).sample

        # 4. Loss: predicted noise vs real noise
        loss = F.mse_loss(noise_pred, noise)

        # 5. Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Average loss epoch {epoch}: {average_loss:.4f}.")

    # ---- Save Checkpoint at End of Epoch ----
    checkpoint = {
        "epoch": epoch + 1,
        "model_state": unet.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "projector_state": projector.state_dict(),
        "loss": average_loss,
    }

    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    if average_loss < best_avg_loss:

        best_checkpoint = {
            "epoch": epoch + 1,
            "model_state": unet.state_dict(),
            "projector_state": projector.state_dict(),
        }
        best_avg_loss = average_loss
        torch.save(best_checkpoint, BEST_DIFFUSION_MODEL)
    print(f"Best average loss: {best_avg_loss:.4f}")
