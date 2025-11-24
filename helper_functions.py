import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

from model import MyStableDiffusionXLPipeline

pipe = MyStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16, use_safetensors=True, cache_dir='weights', config_path = 'model_config.yaml').to('cuda')

# ---------- Helpers ----------

def to_pil(img):
    """Convert input (path, np.array, or PIL) to PIL.Image (RGB)."""
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    elif isinstance(img, Image.Image):
        img = img.convert("RGB")
    else:
        raise ValueError("img must be path, numpy array, or PIL.Image")
    return img


def get_masks(pil1, pil2, resize=(1024, 1024)):
    """Resize inputs and get face & hair masks as numpy arrays."""
    pil1 = pil1.resize(resize)
    pil2 = pil2.resize(resize)
    results = pipe.mask_model.get_masks([pil1, pil2])
    _, face_mask1, hair_mask1 = results[0]
    _, face_mask2, hair_mask2 = results[1]
    return pil1, pil2, np.array(face_mask1), np.array(hair_mask1), np.array(face_mask2), np.array(hair_mask2)


def apply_mask(pil_img, mask):
    """Apply binary mask to a PIL image, return masked PIL image."""
    np_img = np.array(pil_img)
    np_masked = np.zeros_like(np_img)
    np_masked[mask > 0] = np_img[mask > 0]
    return Image.fromarray(np_masked)


def apply_blur(pil_img, mask, radius=25):
    """Blur surroundings while keeping masked region sharp."""
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    np_img = np.array(pil_img)
    np_blur = np.array(blurred)
    np_out = np_blur.copy()
    np_out[mask > 0] = np_img[mask > 0]
    return Image.fromarray(np_out)


def cosine_sim(e1, e2):
    """Flatten tensors and compute cosine similarity."""
    e1_flat = e1.view(-1)
    e2_flat = e2.view(-1)
    dot = torch.dot(e1_flat, e2_flat).item()
    norm = torch.norm(e1_flat).item() * torch.norm(e2_flat).item()
    return dot / norm if norm > 0 else 0.0


def visualize(pil1, pil2, img1_proc, img2_proc, title1="Originals", title2="Processed"):
    """Visualize side-by-side images at 512x512."""
    vis1 = pil1.resize((512, 512))
    vis2 = pil2.resize((512, 512))
    vis3 = img1_proc.resize((512, 512))
    vis4 = img2_proc.resize((512, 512))

    concat1 = Image.new("RGB", (1024, 512))
    concat1.paste(vis1, (0, 0))
    concat1.paste(vis2, (512, 0))

    concat2 = Image.new("RGB", (1024, 512))
    concat2.paste(vis3, (0, 0))
    concat2.paste(vis4, (512, 0))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(concat1)
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(concat2)
    plt.title(title2)
    plt.axis("off")
    plt.show()


# ---------- Similarity Functions ----------

def hair_sim(img1, img2, resize=None):
    """Hair similarity using pipe.hair_encoder."""
    pil1, pil2 = to_pil(img1), to_pil(img2)

    # Get and apply hair masks
    pil1, pil2, _, hair1_mask, _, hair2_mask = get_masks(pil1, pil2)
    hair1 = apply_mask(pil1, hair1_mask)
    hair2 = apply_mask(pil2, hair2_mask)

    # Preprocess & encode
    def preprocess(img):
        if resize is not None:
            img = img.resize(resize)
        return pipe.from_pil_image(img).unsqueeze(0)

    t1, t2 = preprocess(hair1), preprocess(hair2)
    with torch.no_grad():
        e1 = pipe.hair_encoder(t1)
        e2 = pipe.hair_encoder(t2)

    sim = cosine_sim(e1, e2)
    print(f"Hair similarity: {sim:.4f}")

    visualize(pil1, pil2, hair1, hair2, "Original Images", "Masked Hair")
    return sim


def id_sim(img1, img2, mode="none", resize=(1957, 1250)):
    """
    Identity similarity using pipe.id_encoder.
    mode: 'none' (default), 'mask' (keep only face), 'blur' (blur surroundings).
    """
    pil1, pil2 = to_pil(img1), to_pil(img2)

    if mode in ("mask", "blur"):
        pil1, pil2, face1_mask, _, face2_mask, _ = get_masks(pil1, pil2)

        if mode == "mask":
            proc1, proc2 = apply_mask(pil1, face1_mask), apply_mask(pil2, face2_mask)
        elif mode == "blur":
            proc1, proc2 = apply_blur(pil1, face1_mask), apply_blur(pil2, face2_mask)
    else:
        proc1, proc2 = pil1, pil2

    def preprocess(img):
        img = img.resize(resize)
        return pipe.from_pil_image(img).unsqueeze(0)

    t1, t2 = preprocess(proc1), preprocess(proc2)
    with torch.no_grad():
        e1 = pipe.id_encoder(t1)
        e2 = pipe.id_encoder(t2)

    sim = cosine_sim(e1, e2)
    print(f"ID similarity ({mode}): {sim:.4f}")

    visualize(pil1, pil2, proc1, proc2, "Original Images", f"ID Input ({mode})")
    return sim


def clip_sim(img1, img2, mode="none", resize=(224, 224)):
    """
    CLIP similarity using pipe.clip_img_encoder.
    mode:
        'none'       -> full image (default)
        'hair'       -> masked hair
        'face'       -> masked face
        'blur_face'  -> blur surroundings except face
        'blur_hair'  -> blur surroundings except hair
    """
    pil1, pil2 = to_pil(img1), to_pil(img2)

    if mode != "none":
        pil1, pil2, face1_mask, hair1_mask, face2_mask, hair2_mask = get_masks(pil1, pil2)

        if mode == "hair":
            proc1, proc2 = apply_mask(pil1, hair1_mask), apply_mask(pil2, hair2_mask)
        elif mode == "face":
            proc1, proc2 = apply_mask(pil1, face1_mask), apply_mask(pil2, face2_mask)
        elif mode == "blur_face":
            proc1, proc2 = apply_blur(pil1, face1_mask), apply_blur(pil2, face2_mask)
        elif mode == "blur_hair":
            proc1, proc2 = apply_blur(pil1, hair1_mask), apply_blur(pil2, hair2_mask)
        else:
            raise ValueError(f"Unknown mode '{mode}'")
    else:
        proc1, proc2 = pil1, pil2

    def preprocess(img):
        img = img.resize(resize)
        return pipe.from_pil_image(img).unsqueeze(0)

    t1, t2 = preprocess(proc1), preprocess(proc2)
    with torch.no_grad():
        e1 = pipe.clip_img_encoder(t1)
        e2 = pipe.clip_img_encoder(t2)

    sim = cosine_sim(e1, e2)
    print(f"CLIP similarity ({mode}): {sim:.4f}")

    visualize(pil1, pil2, proc1, proc2, "Original Images", f"CLIP Input ({mode})")
    return sim
