# import os
# import random
# from PIL import Image
# import numpy as np

# class DataLoader:
#     def __init__(
#             self,
#             images_dir="shhq_dataset/images",
#             masks_dir="shhq_dataset/masks",
#             captions_dir="shhq_dataset/captions",
#             batch_size=4,
#             mask_format="shhq",  # currently only this is supported
#             ddp_rank=0,
#             ddp_world_size=1,
#             train_test_split=0.9,
#             seed=42,
#             shuffle=True,
#             failed_captions_file='shhq_dataset/failed_captions.txt',
#     ):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.captions_dir = captions_dir
#         self.batch_size = batch_size
#         self.ddp_rank = ddp_rank
#         self.ddp_world_size = ddp_world_size
#         self.seed = seed
#         self.shuffle = shuffle
#         self.mask_format = mask_format

#         self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         self.mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
#         self.caption_files = [f for f in os.listdir(captions_dir) if f.lower().endswith('.txt')]

#         if failed_captions_file and os.path.exists(failed_captions_file):
#             with open(failed_captions_file, "r") as f:
#                 failed_files = set(line.strip().split('.')[0] for line in f.readlines())
#             self.image_files = [f for f in self.image_files if os.path.splitext(f)[0] not in failed_files]
#             self.mask_files = [f for f in self.mask_files if os.path.splitext(f)[0] not in failed_files]
#             self.caption_files = [f for f in self.caption_files if os.path.splitext(f)[0] not in failed_files]

#         if shuffle:
#             random.seed(seed)
#             combined = list(zip(self.image_files, self.mask_files, self.caption_files))
#             random.shuffle(combined)
#             self.image_files, self.mask_files, self.caption_files = zip(*combined)
#             self.image_files, self.mask_files, self.caption_files = list(self.image_files), list(self.mask_files), list(self.caption_files)

#         # train / test split
#         split_idx = int(len(self.image_files) * train_test_split)
#         self.train_files = self.image_files[:split_idx]
#         self.test_files = self.image_files[split_idx:]

#         print(f"Total Image files in dataset are {len(self.image_files)}")
#         print(f"Total train image files are {len(self.train_files)}")
#         print(f"Total test image files are {len(self.test_files)}")

#         self.male_files = [f for f in self.train_files if "man" == self.get_gender(self.load_caption(f))]
#         self.female_files = [f for f in self.train_files if "woman" == self.get_gender(self.load_caption(f))]

#         self.current_position = self.batch_size * self.ddp_rank

#         self.deterministic_head_map = {}
#         for f in self.test_files:
#             caption = self.load_caption(f)
#             gender = self.get_gender(caption)
#             pool = self.female_files if gender == "woman" else self.male_files
#             random.seed(seed + hash(f) % 1000000)
#             self.deterministic_head_map[f] = random.choice(pool)

#     def load_caption(self, filename):
#         path = os.path.join(self.captions_dir, os.path.splitext(filename)[0] + ".txt")
#         with open(path, "r") as f:
#             return f.read().strip().lower()

#     def get_gender(self, caption):
#         if "woman" in caption:
#             return "woman"
#         return "man"

#     def process_mask(self, mask_path):
#         mask = Image.open(mask_path).convert("RGB")
#         mask = mask.crop((0, 0, 512, 512))
#         mask = mask.resize((1024, 1024), resample=Image.NEAREST)
#         mask_np = np.array(mask)
#         RED = np.array([255, 0, 0])
#         BLUE = np.array([16, 78, 139])
#         GREEN = np.array([0, 100, 0])
#         hair_mask = np.all(mask_np == RED, axis=-1).astype(np.uint8)
#         face_mask = (
#             np.all(mask_np == BLUE, axis=-1) |
#             np.all(mask_np == GREEN, axis=-1)
#         ).astype(np.uint8)
#         return hair_mask, face_mask

#     def make_head_mask(self, body_hair_masks, body_face_masks, head_hair_masks, head_face_masks):
#         return body_hair_masks

#     def _make_batch_from_files(self, files, deterministic=False):
#         body_images, body_hair_masks, body_face_masks, body_captions = [], [], [], []

#         for f in files:
#             body_images.append(Image.open(os.path.join(self.images_dir, f)).convert("RGB"))
#             hair_mask, face_mask = self.process_mask(os.path.join(self.masks_dir, os.path.splitext(f)[0] + ".png"))
#             body_hair_masks.append(hair_mask)
#             body_face_masks.append(face_mask)
#             body_captions.append(self.load_caption(f))

#         head_images, head_hair_masks, head_face_masks, head_captions = [], [], [], []
#         for i, (f, caption) in enumerate(zip(files, body_captions)):
#             if deterministic and f in self.deterministic_head_map:
#                 head_f = self.deterministic_head_map[f]
#             else:
#                 gender = self.get_gender(caption)
#                 pool = self.female_files if gender == "woman" else self.male_files
#                 head_f = random.choice(pool)

#             head_images.append(Image.open(os.path.join(self.images_dir, head_f)).convert("RGB"))
#             hair_mask, face_mask = self.process_mask(os.path.join(self.masks_dir, os.path.splitext(head_f)[0] + ".png"))
#             head_hair_masks.append(hair_mask)
#             head_face_masks.append(face_mask)
#             head_captions.append(self.load_caption(head_f))

#         head_masks = self.make_head_mask(body_hair_masks, body_face_masks, head_hair_masks, head_face_masks)

#         return (
#             body_images,
#             head_images,
#             body_hair_masks,
#             head_hair_masks,
#             body_captions,
#             head_captions,
#             head_face_masks,
#             head_masks,
#         )

#     def get_train_batch(self):
#         files = self.train_files
#         n = len(files)
#         selected_files = [files[(self.current_position + i) % n] for i in range(self.batch_size)]

#         self.current_position += self.batch_size * self.ddp_world_size
#         if self.current_position >= n:
#             self.current_position %= n

#         return self._make_batch_from_files(selected_files, deterministic=False)

#     def get_test_batch(self):
#         files = self.test_files
#         half_batch = self.batch_size // 2

#         deterministic_files = files[:half_batch]
#         random_files = random.sample(files, self.batch_size - half_batch)
#         selected_files = deterministic_files + random_files

#         return self._make_batch_from_files(
#             selected_files,
#             deterministic=True
#         )



import os
import random
import queue
import threading
from PIL import Image
import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter

class DataLoader:
    def __init__(
            self,
            images_dir: str = "shhq_dataset/images",
            masks_dir: str = "shhq_dataset/masks",
            captions_dir: str = "shhq_dataset/captions",
            batch_size: int = 4,
            mask_format: str = "shhq",  # currently only this is supported
            ddp_rank: int = 0,
            ddp_world_size: int = 1,
            train_test_split: float = 0.9,
            seed: int = 42,
            shuffle: bool = True,
            failed_captions_file: str = 'shhq_dataset/failed_captions.txt',
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.captions_dir = captions_dir
        self.batch_size = batch_size
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.seed = seed
        self.shuffle = shuffle
        self.mask_format = mask_format

        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
        self.caption_files = [f for f in os.listdir(captions_dir) if f.lower().endswith('.txt')]

        if failed_captions_file and os.path.exists(failed_captions_file):
            with open(failed_captions_file, "r") as f:
                failed_files = set(line.strip().split('.')[0] for line in f.readlines())
            self.image_files = [f for f in self.image_files if os.path.splitext(f)[0] not in failed_files]
            self.mask_files = [f for f in self.mask_files if os.path.splitext(f)[0] not in failed_files]
            self.caption_files = [f for f in self.caption_files if os.path.splitext(f)[0] not in failed_files]

        if shuffle:
            random.seed(seed)
            combined = list(zip(self.image_files, self.mask_files, self.caption_files))
            random.shuffle(combined)
            self.image_files, self.mask_files, self.caption_files = zip(*combined)
            self.image_files, self.mask_files, self.caption_files = list(self.image_files), list(self.mask_files), list(self.caption_files)

        # train / test split
        split_idx = int(len(self.image_files) * train_test_split)
        self.train_files = self.image_files[:split_idx]
        self.test_files = self.image_files[split_idx:]

        print(f"Total Image files in dataset are {len(self.image_files)}")
        print(f"Total train image files are {len(self.train_files)}")
        print(f"Total test image files are {len(self.test_files)}")

        # Preload captions into memory (full cache)
        self.caption_cache = {f: self._load_caption_file(f) for f in self.image_files}

        self.male_files = [f for f in self.train_files if "man" == self.get_gender(self.load_caption(f))]
        self.female_files = [f for f in self.train_files if "woman" == self.get_gender(self.load_caption(f))]

        self.current_position = self.batch_size * self.ddp_rank

        self.deterministic_head_map = {}
        for f in self.test_files:
            caption = self.load_caption(f)
            gender = self.get_gender(caption)
            pool = self.female_files if gender == "woman" else self.male_files
            random.seed(seed + hash(f) % 1000000)
            self.deterministic_head_map[f] = random.choice(pool)

    def _load_caption_file(self, filename: str):
        path = os.path.join(self.captions_dir, os.path.splitext(filename)[0] + ".txt")
        with open(path, "r") as f:
            return f.read().strip().lower()

    def load_caption(self, filename: str):
        return self.caption_cache[filename]

    def get_gender(self, caption: str):
        if "woman" in caption:
            return "woman"
        return "man"

    def process_mask(self, mask_path: str):
        mask = Image.open(mask_path).convert("RGB")
        mask = mask.crop((0, 0, 512, 512))
        mask = mask.resize((1024, 1024), resample=Image.NEAREST)
        mask_np = np.array(mask)
        RED = np.array([255, 0, 0])
        BLUE = np.array([16, 78, 139])
        GREEN = np.array([0, 100, 0])
        hair_mask = np.all(mask_np == RED, axis=-1).astype(np.uint8)
        face_mask = (
            np.all(mask_np == BLUE, axis=-1) |
            np.all(mask_np == GREEN, axis=-1)
        ).astype(np.uint8)
        return hair_mask, face_mask

    def _shift_mask(self, mask, shift, H, W):
        """Shift mask by (dy, dx) without wrap-around (pads with zeros)."""
        dy, dx = shift
        shifted = np.zeros_like(mask)

        y_start = max(0, dy)
        y_end   = H + min(0, dy)
        x_start = max(0, dx)
        x_end   = W + min(0, dx)

        src_y_start = max(0, -dy)
        src_y_end   = src_y_start + (y_end - y_start)
        src_x_start = max(0, -dx)
        src_x_end   = src_x_start + (x_end - x_start)

        shifted[y_start:y_end, x_start:x_end] = mask[src_y_start:src_y_end, src_x_start:src_x_end]
        return shifted

    def _soften_mask(self, mask: np.ndarray, sigma=3):
        """Apply Gaussian blur to soften edges, keeps mask in [0,1]."""
        blurred = gaussian_filter(mask.astype(float), sigma=sigma)
        return np.clip(blurred, 0, 1)

    def make_head_mask(self, body_hair_masks: np.ndarray, body_face_masks: np.ndarray, head_hair_masks: np.ndarray, head_face_masks: np.ndarray):
        B, H, W = body_face_masks.shape
        aligned_head_face_masks = np.zeros_like(body_face_masks)
        aligned_head_hair_masks = np.zeros_like(body_hair_masks)

        for i in range(B):
            # ---- FACE ----
            body_face = body_face_masks[i]
            head_face = head_face_masks[i]

            if body_face.sum() > 0 and head_face.sum() > 0:
                body_center = np.array(center_of_mass(body_face))
                head_center = np.array(center_of_mass(head_face))

                shift = np.round(body_center - head_center).astype(int)  # (dy, dx)
                aligned_head_face = self._shift_mask(head_face, shift, H, W)
            else:
                aligned_head_face = head_face

            aligned_head_face_masks[i] = aligned_head_face

            # ---- HAIR ----
            body_hair = body_hair_masks[i]
            head_hair = head_hair_masks[i]

            if body_hair.sum() > 0 and head_hair.sum() > 0:
                body_center = np.array(center_of_mass(body_hair))
                head_center = np.array(center_of_mass(head_hair))

                shift = np.round(body_center - head_center).astype(int)
                aligned_head_hair = self._shift_mask(head_hair, shift, H, W)
            else:
                aligned_head_hair = head_hair

            aligned_head_hair_masks[i] = aligned_head_hair

        # Union
        combined_face_masks = np.logical_or(body_face_masks, aligned_head_face_masks)
        combined_hair_masks = np.logical_or(body_hair_masks, aligned_head_hair_masks)
        head_mask = np.logical_or(combined_face_masks, combined_hair_masks).astype(np.uint8)

        head_mask = self._soften_mask(head_mask, sigma=3)
        # final_masks = [cv2.resize(m, (128, 128), interpolation=cv2.INTER_LINEAR) for m in softened_masks]
        downsampled_head_mask = [
            np.array(
                Image.fromarray((m * 255).astype(np.uint8))
                .resize((128, 128), resample=Image.BILINEAR)
            ).astype(np.float32) / 255.0
            for m in head_mask
        ]
        return head_mask, downsampled_head_mask

    def _make_batch_from_files(self, files: list, deterministic: bool = False):
        body_images, body_hair_masks, body_face_masks, body_captions = [], [], [], []

        for f in files:
            body_images.append(Image.open(os.path.join(self.images_dir, f)).convert("RGB"))
            hair_mask, face_mask = self.process_mask(os.path.join(self.masks_dir, os.path.splitext(f)[0] + ".png"))
            body_hair_masks.append(hair_mask)
            body_face_masks.append(face_mask)
            body_captions.append(self.load_caption(f))

        head_images, head_hair_masks, head_face_masks, head_captions = [], [], [], []
        for i, (f, caption) in enumerate(zip(files, body_captions)):
            if deterministic and f in self.deterministic_head_map:
                head_f = self.deterministic_head_map[f]
            else:
                gender = self.get_gender(caption)
                pool = self.female_files if gender == "woman" else self.male_files
                head_f = random.choice(pool)

            head_images.append(Image.open(os.path.join(self.images_dir, head_f)).convert("RGB"))
            hair_mask, face_mask = self.process_mask(os.path.join(self.masks_dir, os.path.splitext(head_f)[0] + ".png"))
            head_hair_masks.append(hair_mask)
            head_face_masks.append(face_mask)
            head_captions.append(self.load_caption(head_f))

        body_hair_masks = np.stack(body_hair_masks)
        body_face_masks = np.stack(body_face_masks)
        head_hair_masks = np.stack(head_hair_masks)
        head_face_masks = np.stack(head_face_masks)

        head_masks, downsampled_head_masks = self.make_head_mask(body_hair_masks, body_face_masks, head_hair_masks, head_face_masks)

        return (
            body_images,
            head_images,
            body_hair_masks,
            head_hair_masks,
            body_captions,
            head_captions,
            head_face_masks,
            head_masks,
            downsampled_head_masks,
        )

    def get_train_batch(self):
        files = self.train_files
        n = len(files)
        selected_files = [files[(self.current_position + i) % n] for i in range(self.batch_size)]

        self.current_position += self.batch_size * self.ddp_world_size
        if self.current_position >= n:
            self.current_position %= n

        return self._make_batch_from_files(selected_files, deterministic=False)

    def get_test_batch(self):
        files = self.test_files
        half_batch = self.batch_size // 2

        deterministic_files = files[:half_batch]
        random_files = random.sample(files, self.batch_size - half_batch)
        selected_files = deterministic_files + random_files

        return self._make_batch_from_files(
            selected_files,
            deterministic=True
        )
    
    def state_dict(self):
        global_step = self.current_position // (self.batch_size * self.ddp_world_size)
        return {
            "global_step": global_step,
            "seed": self.seed,
            "deterministic_head_map": self.deterministic_head_map,
        }

    def load_state_dict(self, state: dict):
        global_step = state["global_step"]
        self.seed = state["seed"]
        self.deterministic_head_map = state["deterministic_head_map"]

        self.current_position = global_step * self.batch_size * self.ddp_world_size
        self.current_position += self.batch_size * self.ddp_rank

        random.seed(self.seed)



# -------------------------------
# Async Prefetcher Wrapper
# -------------------------------
class PrefetchDataLoader:
    def __init__(self, dataloader: DataLoader, mode: str = "train", num_workers: int = 2, prefetch_batches: int = 4):
        self.dataloader = dataloader
        self.mode = mode
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.queue = queue.Queue(maxsize=prefetch_batches)

        self.stop_event = threading.Event()
        self.workers = [
            threading.Thread(target=self._worker_loop, daemon=True)
            for _ in range(num_workers)
        ]
        for w in self.workers:
            w.start()

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.mode == "train":
                    batch = self.dataloader.get_train_batch()
                else:
                    batch = self.dataloader.get_test_batch()
                self.queue.put(batch, timeout=1)
            except queue.Full:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                break

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.queue.get(timeout=5)
        except queue.Empty:
            raise StopIteration

    def shutdown(self):
        self.stop_event.set()
        for w in self.workers:
            w.join(timeout=1)
