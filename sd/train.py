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
from torch import nn
from ddpm import DDPMSampler
from accelerate import Accelerator
from pipeline import WIDTH, HEIGHT, LATENTS_WIDTH, LATENTS_HEIGHT, get_time_embedding

# TODO mudar os embeddings de (768,44) para (512,1)

NUM_EPOCHS = 10
BATCH_SIZE = 4
N_INFERENCE_STEPS = 50

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


def generate_latent_imgs(encoder, sampler, generator, input_images_tensor, timestep):
    with torch.no_grad():
        n_images = input_images_tensor.shape[0]

        latents_shape = (n_images, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        # (Batch_Size, 4, Latents_Height, Latents_Width)
        encoder_noise = torch.randn(latents_shape, generator=generator, device=DEVICE)

        # (Batch_Size, 4, Latents_Height, Latents_Width)
        latents = encoder(input_images_tensor, encoder_noise)
        latents = sampler.add_noise(latents, timestep)
        return latents, encoder_noise


def test(test_loader, models, sampler, generator):
    encoder = models["encoder"]
    clip = models["clip"]
    diffusion = models["diffusion"]
    loss_fn = nn.MSELoss()

    total_test_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            loss = calc_loss(batch, sampler, encoder, generator, clip, diffusion, loss_fn)

            loss_value = loss.item()
            total_test_loss += loss_value

        test_loss = total_test_loss / len(test_loader)
        print(f"[Test] Loss: {test_loss:.4f}")


def train(train_loader, val_loader, models, sampler, generator):
    encoder = models["encoder"]
    clip = models["clip"]
    diffusion = models["diffusion"]
    decoder = models["decoder"]

    loss_fn = nn.MSELoss()

    # Initialize Accelerator
    accelerator = Accelerator()

    all_params = list(encoder.parameters()) + list(clip.parameters()) + \
        list(diffusion.parameters()) + list(decoder.parameters())

    # Optimizer setup
    optimizer = torch.optim.AdamW(all_params, lr=5e-6)

    train_loss_per_epoch = []
    val_loss_per_epoch = []

    # Example training loop
    for epoch in range(NUM_EPOCHS):
        encoder.train()
        decoder.train()
        clip.train()
        diffusion.train()

        total_train_loss = 0

        for i, batch in enumerate(train_loader):
            loss = calc_loss(batch, sampler, encoder, generator, clip, diffusion, loss_fn)

            loss_value = loss.item()

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if i % 2 == 0:
                print(f"[Train] Epoch {epoch}, Step {i}, Loss: {loss_value:.4f}")

            total_train_loss += loss_value

        train_loss_per_epoch.append(total_train_loss / len(train_loader))
        print(f"[Train] Epoch {epoch}, Loss: {train_loss_per_epoch[-1]:.4f}")

        encoder.eval()
        decoder.eval()
        clip.eval()
        diffusion.eval()

        with torch.no_grad():
            total_val_loss = 0
            for i, batch in enumerate(val_loader):
                loss = calc_loss(batch, sampler, encoder, generator, clip, diffusion, loss_fn)

                loss_value = loss.item()
                total_val_loss += loss_value

                if i % 2 == 0:
                    print(f"[Val] Epoch {epoch}, Step {i}, Loss: {loss_value:.4f}")

            val_loss_per_epoch.append(total_val_loss / len(val_loader))

            print(f"[Val] Epoch {epoch}, Loss: {val_loss_per_epoch[-1]:.4f}")

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }


def calc_loss(batch, sampler, encoder, generator, clip, diffusion, loss_fn):
    images = batch[0]
    texts = batch[1]

    input_images = images.to(DEVICE)
    batch_size = input_images.shape[0]

    timestep = torch.randint(
        low=0, high=sampler.num_train_timesteps, size=(batch_size,),
        device=DEVICE
    )
    latent_imgs, noise = generate_latent_imgs(
        encoder, sampler, generator, input_images, timestep)

    tokens = generate_tokens(tokenizer, texts)

    context = clip(tokens)

    time_embedding = get_time_embedding(timestep).to(DEVICE)

    noise_pred = diffusion(latent_imgs, context, time_embedding)

    loss = loss_fn(noise_pred, noise)
    return loss

        # for _ in range(10):
        #     encoder.eval()
        #     decoder.eval()
        #     clip.eval()
        #     diffusion.eval()


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
    sampler.set_inference_timesteps(N_INFERENCE_STEPS)
    sampler.set_strength(strength=0.9)

    print("Training...")
    models =  train(train_loader, val_loader, models, sampler, generator)

    print("Testing...")
    test(test_loader, models, sampler, generator)

    output_file_name = 'multimodal_model_no_custom_clip.pth'
    torch.save({
        'clip': models['clip'].state_dict(),
        'encoder': models['encoder'].state_dict(),
        'decoder': models['decoder'].state_dict(),
        'diffusion': models['diffusion'].state_dict(),
    }, output_file_name)

    print(output_file_name, "saved.")
