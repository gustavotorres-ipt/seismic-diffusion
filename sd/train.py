import torch
import open_clip
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import model_loader
import numpy as np
import matplotlib.pyplot as plt
from dataset import load_datasets
from torch.utils.data import DataLoader
from ddpm import DDPMSampler
from pipeline import WIDTH, HEIGHT, LATENTS_WIDTH, LATENTS_HEIGHT, get_time_embedding

NUM_EPOCHS = 2
BATCH_SIZE = 2
n_inference_steps = 30

def generate_tokens(tokenizer, prompts):
    with torch.no_grad():
        # Convert into a list of length Seq_Len=77
        tokens = tokenizer.batch_encode_plus(
            prompts, padding="max_length", max_length=77
        ).input_ids
        # (Batch_Size, Seq_Len)
        tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)

        return tokens


def generate_latent_imgs(encoder, sampler, generator, input_images_tensor):
    with torch.no_grad():
        latents_shape = (BATCH_SIZE, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        # (Batch_Size, 4, Latents_Height, Latents_Width)
        encoder_noise = torch.randn(latents_shape, generator=generator, device=DEVICE)

        # (Batch_Size, 4, Latents_Height, Latents_Width)
        latents = encoder(input_images_tensor, encoder_noise)
        latents = sampler.add_noise(latents, sampler.timesteps[0])
        return latents


def train(dataloader, models, sampler, generator):
    # preciso de um sampler, tokenizer, clip

    # Initialize Accelerator
    # accelerator = Accelerator()

    # Optimizer setup
    # optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=5e-6)
    # encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
    # # (Batch_Size, 4, Latents_Height, Latents_Width)
    # latents = encoder(input_images_tensor, encoder_noise)
    # latents = generate_initial_latents()
    encoder = models["encoder"]
    clip = models["clip"]
    diffusion = models["diffusion"]

    # Example training loop
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(dataloader):
            images = batch[0]
            texts = batch[1]

            input_images = images.to(DEVICE)

            latent_imgs = generate_latent_imgs(encoder, sampler, generator, input_images)
            
            tokens = generate_tokens(tokenizer, texts)
            
            context = clip(tokens)

            time_embedding = get_time_embedding(sampler.timesteps[i]).to(DEVICE)

            model_output = diffusion(latent_imgs, context, time_embedding)

            breakpoint()

            # Compute loss
            # loss = loss_fn(model_output, batch["labels"])

            # Backpropagation
            # accelerator.backward(loss)
            # optimizer.step()
            # optimizer.zero_grad()

            # Logging
            # print(f"Epoch {epoch}, Loss: {loss.item()}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    #model, _, preprocess = open_clip.create_model_and_transforms(
    #    'ViT-B-32', pretrained='laion2b_s34b_b79k'
    #)

    tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
    model_file = "../data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

    train_dataset, val_dataset, test_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    generator = torch.Generator(device=DEVICE)
    generator.seed()

    models["encoder"].to(DEVICE)
    models["clip"].to(DEVICE)
    models["diffusion"].to(DEVICE)

    sampler = DDPMSampler(generator)
    sampler.set_inference_timesteps(n_inference_steps)
    sampler.set_strength(strength=0.9)

    train(train_loader, models, sampler, generator)