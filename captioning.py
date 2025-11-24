import os
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Paths
input_folder = "shhq_dataset/gen_images"
output_folder = "shhq_dataset/gen_images_captions"
os.makedirs(output_folder, exist_ok=True)

weights_dir = os.path.join(os.getcwd(), "weights")
os.makedirs(weights_dir, exist_ok=True)

batch_size = 32
device = torch.device("cuda:6")
failure_file = os.path.join("shhq_dataset/failed_captions_2.txt")

# Load model + processor
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": device.index},
    cache_dir=weights_dir,
)
processor = AutoProcessor.from_pretrained(model_id, cache_dir=weights_dir)

# Prompt template
prompt_text = (
    "Write exactly ONE short, natural-sounding sentence describing the image. "
    "The caption MUST contain the word 'man' OR 'woman' (choose correctly), "
    "AND it MUST contain the word 'hairstyle'. "
    "Do not use synonyms. Do not write multiple sentences. "
    "Only output the caption text, nothing else."
)

# Collect images
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
print(f"Total image files: {len(image_files)}")

with open(failure_file, "w") as fail_f:
    for i in tqdm(range(0, len(image_files), batch_size), desc="Captioning"):
        batch_files = image_files[i:i+batch_size]

        # Build messages list (one conversation per image)
        messages = []
        for img_file in batch_files:
            img_path = os.path.join(input_folder, img_file)
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{os.path.abspath(img_path)}"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
            )

        # Prepare inputs
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=50, pad_token_id=processor.tokenizer.eos_token_id)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        captions = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Save captions + check failures
        for img_file, caption_text in zip(batch_files, captions):
            caption_text = caption_text.strip()
            # print(img_file, "→", caption_text)

            txt_name = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_name)

            if "hairstyle" not in caption_text.lower() and "hair" in caption_text.lower():
                caption_text = caption_text.replace("hair", "hairstyle").replace("Hair", "Hairstyle")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption_text)

            if not (("man" in caption_text.lower() or "woman" in caption_text.lower()) and "hairstyle" in caption_text.lower()):
                fail_f.write(f"{img_file}\n")