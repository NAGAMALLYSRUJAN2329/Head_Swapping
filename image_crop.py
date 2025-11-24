import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

input_folder = "/workspace/SHHQ-1.0/no_segment"
output_folder = "shhq_cropped_imgs"
os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))]

def process_image(filename):
    img_path = os.path.join(input_folder, filename)
    save_path = os.path.join(output_folder, filename)
    with Image.open(img_path) as img:
        cropped = img.crop((0, 0, 512, 512))
        cropped.save(save_path, optimize=True)
    return filename

with ProcessPoolExecutor(max_workers=100) as executor:
    list(tqdm(executor.map(process_image, files), total=len(files), desc="Processing"))