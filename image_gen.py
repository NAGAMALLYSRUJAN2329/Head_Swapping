import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import logging
from PIL import Image
from tqdm import tqdm
import os
import argparse
import random

logging.disable_progress_bar()

images_dir = "shhq_dataset/images"
save_dir = "shhq_dataset/gen_images"
os.makedirs(save_dir, exist_ok=True)
# python image_gen.py --gpu 0 --start 0 --end 10000

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Start index of images")
parser.add_argument("--end", type=int, default=50, help="End index of images")
parser.add_argument("--gpu", type=int, help="Gpu to run on")
args = parser.parse_args()

start = args.start
end = args.end
gpu = args.gpu

base_instruction = "Change the Expression, Pose, Outfit. Don't change the Hairstyle or Face identity."
scenes = [
    "in a library",
    "at a cafe",
    "in a garden",
    "playing guitar on a stage",
    "in the kitchen",
    "in a park",
    "in an office",
    "at home",
    "walking in the street",
    "standing near a window",
    "at the beach",
    "in the gym",
    "at the museum",
    "on a rooftop",
    "at the bus stop",
    "in a classroom",
    "in a bookstore",
    "in a shopping mall",
    "on a train",
    "in a restaurant"
]

def random_prompt():
    return f"Image of a person {random.choice(scenes)}. {base_instruction}"

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    cache_dir='weights'
)
pipe.to(f"cuda:{gpu}")
# pipe = torch.compile(pipe)

torch.backends.cudnn.benchmark = True

image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
image_files = image_files[start:end]
total_files = len(image_files)

for i in tqdm(range(total_files), desc="Processing"):
    image_path = os.path.join(images_dir, image_files[i])
    save_path = os.path.join(save_dir, image_files[i])
    if os.path.exists(save_path):
        continue
    prompt = random_prompt()
    # print(type(prompt))
    # print(prompt)
    image = pipe(
        image=Image.open(image_path),
        prompt=prompt,
        guidance_scale=1.0,
        num_inference_steps=28,
    ).images[0]
    image.save(save_path)