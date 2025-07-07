import torch
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from clip_seismic import load_clip_model

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    # diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    custom_clip_model = load_clip_model()

    if "multimodal_model" not in ckpt_path:
        custom_clip_model.load_state_dict(torch.load("customized_clip.pth"))
    custom_clip_model.to(device)

    return {
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
        'clip': custom_clip_model,
    }

    # clip = CLIP().to(device)
    # clip.load_state_dict(state_dict['clip'], strict=True)

