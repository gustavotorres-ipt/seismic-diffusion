#!/usr/bin/env python
# coding: utf-8

import torch
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import model_loader
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

BATCH_SIZE = 1


if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")


def plot_image(img):
    # img_pil = np.transpose(img, (1, 2, 0))
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.close()


tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model_file = "multimodal_model.pth"
model_file="../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE

prompts = ["Give me an image of a seismic fault." for _ in range(BATCH_SIZE)]
uncond_prompts = ["" for _ in range(BATCH_SIZE)]  # Also known as negative prompt
# cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_images = None
# Comment to disable image to image
# image_path = "../data/dog.jpeg"
# input_images = [Image.open(image_path) for _ in range(BATCH_SIZE)]
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "ddpm"
num_inference_steps = 30
seed = 42

output_images = pipeline.generate(
    prompts=prompts,
    uncond_prompts=uncond_prompts,
    input_images=input_images,
    strength=strength,
    do_cfg=False,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
)

# Combine the input image and the output image into a single image.
# output_images = Image.fromarray(output_images)

for img in output_images:
    plot_image(img)
