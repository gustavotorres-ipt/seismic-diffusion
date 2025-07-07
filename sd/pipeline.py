import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 256
HEIGHT = 256
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompts,
    uncond_prompts=None,
    input_images=None,
    strength=0.8,
    do_cfg=False,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=30,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
    batch_size=1,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        # Convert into a list of length Seq_Len=77
        tokens = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt",
            max_length=128
        ).to(device)
        # (Batch_Size, Seq_Len)
        # tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        context = clip.encode_text(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_images:
            encoder = models["encoder"]
            encoder.to(device)

            input_images_tensor = [image.resize((WIDTH, HEIGHT)) for image in input_images]
            # (Batch_Size, Height, Width, Channel)
            input_images_tensor = np.array(input_images_tensor)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_images_tensor = torch.tensor(input_images_tensor, dtype=torch.float32, device=device)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_images_tensor = rescale(input_images_tensor, (0, 255), (-1, 1))
            # input_images_tensor = input_images_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_images_tensor = input_images_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_images_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            timestep_tensor = torch.tensor([timestep], dtype=torch.float32, device=device)
            time_embedding = get_time_embedding(timestep_tensor).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(
        start=0, end=160, dtype=torch.float32, device=timestep.device) / 160)
    # Shape: (1, 160)
    x = timestep[:, None] * freqs[None]
    # x = torch.tensor(timestep, dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
