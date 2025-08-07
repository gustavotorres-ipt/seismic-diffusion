import os
from diffusers import PriorTransformer
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

n_timesteps_decoder = 50
n_timesteps_prior = 1000

TRAINED_DIFFUSION_MODEL = "best_diffusion_model.pt"
TRAINED_PRIOR = "best_prior_model.pt"
CLIP_WEIGHTS_FILE = "customized_clip.pth"


def convert_text_to_image_embeds(prior, text_embeddings):
    img_embeddings = torch.randn( text_embeddings.size()).to(DEVICE)

    print("Converting text embeddings to image embeddings...")
    for t in tqdm(range(n_timesteps_prior, 0, -1)):

        timestep = torch.tensor([t]).long()

        img_embeddings = prior(
            hidden_states=img_embeddings,
            timestep=timestep.to(DEVICE),
            proj_embedding=text_embeddings,
            encoder_hidden_states=text_embeddings.unsqueeze(1),
        ).predicted_image_embedding  # or predicted noise, depending on setup

    return img_embeddings


def generate_images(scheduler, unet, projector, vae, img_embeddings):
    with torch.no_grad():

        # embed_512 = torch.randn(NUM_IMAGES, 512).to(DEVICE)  # from the prior
        embed_512 = img_embeddings
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
    user_prompt = "A fault to the east."  # input("Image prompt: ")

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

        prior = PriorTransformer(embedding_dim=512, num_embeddings=1)
        if TRAINED_PRIOR in os.listdir("."):
            prior_checkpoint = torch.load(TRAINED_PRIOR)
            prior.load_state_dict(prior_checkpoint["model_state"])
        prior.to(DEVICE)

        tok_prompt = get_text_prompt_tokens()

        text_embeddings = custom_clip_model.encode_text(tok_prompt)
        img_embeddings = convert_text_to_image_embeds(prior, text_embeddings)

        if NUM_IMAGES > 1:
            img_embeddings = img_embeddings.repeat((NUM_IMAGES, 1))

        unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet"
        ).to(DEVICE)

        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae").to(DEVICE)
        projector = EmbedProjector().to(DEVICE)

        scheduler = DDIMScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )
        scheduler.set_timesteps(n_timesteps_decoder)

        if TRAINED_DIFFUSION_MODEL in os.listdir("."):
            checkpoint = torch.load(TRAINED_DIFFUSION_MODEL, map_location=DEVICE)
            unet.load_state_dict(checkpoint["model_state"])
            projector.load_state_dict(checkpoint["projector_state"])

        print("Generating image...")
        generate_images(scheduler, unet, projector, vae, img_embeddings)


if __name__ == "__main__":
    main()
