import os
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.transforms.functional import to_pil_image
from torch import nn
from tqdm import tqdm
from model_loader import load_clip_model, EmbedProjector
from transformers import AutoTokenizer


NUM_IMAGES = 1
DEVICE = "cuda"

TRAINED_DIFFUSION_MODEL = "best_diffusion_model.pt"
CLIP_WEIGHTS_FILE = "customized_clip.pth"


def generate_images(scheduler, unet, projector, vae, text_features):
    with torch.no_grad():

        # embed_512 = torch.randn(NUM_IMAGES, 512).to(DEVICE)  # from the prior
        embed_512 = text_features
        latents = torch.randn(
            (NUM_IMAGES, 4, 32, 32),  # 4 latent channels, 64x64 size for 256x256 output
            device=DEVICE
        )

        embed_768 = projector(embed_512) # (B, 1, 768)

        # Denoising loop
        for t in tqdm(scheduler.timesteps):
            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=embed_768  # shape: (B, 1, 768)
            ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Decode latents to image using VAE decoder
        latents = latents / 0.18215  # Scaling factor used in SD 1.4/1.5
        images = vae.decode(latents).sample

        show_images(images)


def show_images(images):
    for i in range(NUM_IMAGES):
        image = images[i].clamp(0, 1).cpu()
        # Convert to PIL image
        final_image = to_pil_image(image[2, :, :])
        final_image.show()


def get_text_prompt_tokens():
    user_prompt = input("Image prompt: ")

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tok_texts = tokenizer(
        user_prompt, padding=True, truncation=True, return_tensors="pt",
        max_length=128
    ).to(DEVICE)
    return tok_texts


def main():
    with torch.no_grad():
        # Load trained CLIP Model
        custom_clip_model, _ = load_clip_model()

        custom_clip_model.load_state_dict(torch.load(CLIP_WEIGHTS_FILE))
        custom_clip_model.to(DEVICE)

        # prior = 

        tok_prompt = get_text_prompt_tokens()

        text_embeds = custom_clip_model.encode_text(tok_prompt)
        # convert_text_to_image_embeds(prior, text_embeds)


        if NUM_IMAGES > 1:
            text_embeds = text_embeds.repeat((NUM_IMAGES, 1))
        # make the text features dimensions same as num images

        # 1. Load pretrained UNet and scheduler (decoder stage)
        unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        ).to(DEVICE)

        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae").to(DEVICE)
        projector = EmbedProjector().to(DEVICE)

        scheduler = DDIMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )
        scheduler.set_timesteps(50)

        if TRAINED_DIFFUSION_MODEL in os.listdir("."):
            checkpoint = torch.load(TRAINED_DIFFUSION_MODEL, map_location=DEVICE)
            unet.load_state_dict(checkpoint["model_state"])
            projector.load_state_dict(checkpoint["projector_state"])

        generate_images(scheduler, unet, projector, vae, text_embeds)


if __name__ == "__main__":
    main()
