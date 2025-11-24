from model import MyStableDiffusionXLPipeline
import torch
from PIL import Image
import requests
from io import BytesIO

device = 'cuda:5'
pipe = MyStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True, cache_dir='weights', config_path = 'model_config.yaml').to(device)
# g = torch.Generator(device='cuda').manual_seed(0)

# for p in pipe.parameters():
#     p.requires_grad = False
# for p in pipe.trainable_parameters():
#     p.requires_grad = True
print(pipe.use_lora)
print(sum(p.numel() for p in pipe.get_unet_lora_params()))
pipe.load_adapters("output_small_2/weights/checkpoint_5400.pth")

img1 = Image.open("/workspace/small_shhq/images/image_000003.png").convert("RGB").resize((1024, 1024))
img2 = Image.open("/workspace/small_shhq/images/image_000006.png").convert("RGB").resize((1024, 1024))
prompt = "Image of a woman with hairstyle"

with torch.no_grad():
    output_images, io_masks = pipe.predict([img1], None, [img2], None, [prompt], [prompt], invert_till = 40)
output_images[0].save("temp.png")