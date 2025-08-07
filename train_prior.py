import os
from regex import P
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL, DDPMScheduler
from diffusers import PriorTransformer
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
BATCH_SIZE = 4
CHECKPOINT_DIR = "."
CHECKPOINT_NAME = "checkpoint_prior.pt"
BEST_PRIOR_MODEL = "best_prior_model.pt"

scheduler = DDPMScheduler(num_train_timesteps=1000)

start_epoch = 0

# Load trained CLIP Model
custom_clip_model, preprocessor = load_clip_model()
custom_clip_model.load_state_dict(torch.load("customized_clip.pth"))
custom_clip_model.to(DEVICE)

prior = PriorTransformer(embedding_dim=512, num_embeddings=1)
#.from_pretrained("runwayml/stable-diffusion-prior")  
prior.to(DEVICE)

optimizer = torch.optim.AdamW(
    prior.parameters(), lr=1e-4
)

latest_ckpt = CHECKPOINT_NAME \
    if CHECKPOINT_NAME in os.listdir(CHECKPOINT_DIR) else None

if latest_ckpt:
    print(f"Resuming from checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
    prior.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    print(f"Resumed from epoch {start_epoch}")

else:
    start_epoch = 0
    print("No checkpoint found. Starting from scratch.")

# Create dataloader to load images
train_dataset, val_dataset, test_dataset = load_datasets(preprocessor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

best_avg_loss_val = float("inf")

for epoch in range(NUM_EPOCHS):
    total_loss_train = 0
    print(f"----- EPOCH {epoch} -----")

    for images, captions in tqdm(train_loader):

        with torch.no_grad():
            tokenized_texts = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt",
                max_length=128
            ).to(DEVICE)

            images = images.to(DEVICE) # (B, 3, 256, 256), [0,1] range

            # Forward pass: compute image and text features
            text_embeddings = custom_clip_model.encode_text(tokenized_texts)
            text_embeddings = text_embeddings.unsqueeze(1)
            text_embeddings = text_embeddings.to(DEVICE)   # (B, 512)

            image_embeddings = custom_clip_model.encode_image(images)
            image_embeddings = image_embeddings.to(DEVICE)   # (B, 512)

            # 2. Sample timestep (random integer from 0 to num_train_timesteps - 1)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,)).long()

            # 3. Add noise to your image embedding
            # Assume `image_embedding` is (batch_size, embedding_dim)
            noise = torch.randn_like(image_embeddings)
            noisy_embeddings = scheduler.add_noise(image_embeddings, noise, timestep)

        pred_embeddings = prior(
            hidden_states=noisy_embeddings,
            timestep=timestep.to(DEVICE),
            proj_embedding=image_embeddings,  # the clean one
            encoder_hidden_states=text_embeddings,
        ).predicted_image_embedding

        loss = F.mse_loss(pred_embeddings, image_embeddings)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss_train += loss.item()

    total_loss_val = 0

    print("Validating...")
    for images, captions in tqdm(val_loader):
        with torch.no_grad():
            tokenized_texts = tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt",
                max_length=128
            ).to(DEVICE)

            images = images.to(DEVICE)

            # Forward pass: compute image and text features
            text_embeddings = custom_clip_model.encode_text(tokenized_texts)
            text_embeddings = text_embeddings.unsqueeze(1)
            text_embeddings = text_embeddings.to(DEVICE)   # (B, 512)

            image_embeddings = custom_clip_model.encode_image(images)
            image_embeddings = image_embeddings.to(DEVICE)   # (B, 512)

            # 2. Sample timestep (random integer from 0 to num_train_timesteps - 1)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,)).long()

            # 3. Add noise to your image embedding
            # Assume `image_embedding` is (batch_size, embedding_dim)
            noise = torch.randn_like(image_embeddings)
            noisy_embeddings = scheduler.add_noise(image_embeddings, noise, timestep)

            pred_embeddings = prior(
                hidden_states=noisy_embeddings,
                timestep=timestep.to(DEVICE),
                proj_embedding=image_embeddings,  # the clean one
                encoder_hidden_states=text_embeddings,
            ).predicted_image_embedding

            loss = F.mse_loss(pred_embeddings, image_embeddings)

        total_loss_val += loss.item()

    avg_loss_train = total_loss_train / len(train_loader)
    avg_loss_val = total_loss_val / len(val_loader)
    print(f"Average training loss epoch {epoch}: {avg_loss_train:.4f}.")
    print(f"Average validation loss epoch {epoch}: {avg_loss_val:.4f}.")

    # ---- Save Checkpoint at End of Epoch ----
    checkpoint = {
        "epoch": epoch + 1,
        "model_state": prior.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": avg_loss_train,
    }

    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    if avg_loss_val < best_avg_loss_val:

        best_checkpoint = {
            "epoch": epoch + 1,
            "model_state": prior.state_dict(),
        }
        best_avg_loss_val = avg_loss_val
        torch.save(best_checkpoint, BEST_PRIOR_MODEL)
    print(f"Best average loss: {best_avg_loss_val:.4f}")
