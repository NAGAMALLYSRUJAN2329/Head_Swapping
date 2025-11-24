import os
import random
import queue
import threading
import numpy as np
from typing import List
from PIL import Image
from scipy.ndimage import center_of_mass, gaussian_filter, binary_dilation

class DataLoader:
    def __init__(
            self,
            mask_model,
            body_images_dir: str = "shhq_dataset/images",
            head_images_dir: str = "shhq_dataset/gen_images",
            masks_dir: str = "shhq_dataset/masks",
            body_captions_dir: str = "shhq_dataset/captions",
            head_captions_dir: str = "shhq_dataset/gen_images_captions",
            batch_size: int = 4,
            mask_format: str = "shhq",  # currently only this is supported
            ddp_rank: int = 0,
            ddp_world_size: int = 1,
            train_test_split: float = 0.9,
            seed: int = 42,
            shuffle: bool = True,
            failed_captions_files: List[str] = ['shhq_dataset/failed_captions.txt', 'shhq_dataset/failed_captions_2.txt'],
    ):
        self.body_images_dir = body_images_dir
        self.head_images_dir = head_images_dir
        self.masks_dir = masks_dir
        self.body_captions_dir = body_captions_dir
        self.head_captions_dir = head_captions_dir
        self.batch_size = batch_size
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.seed = seed
        self.shuffle = shuffle
        self.mask_format = mask_format
        self.mask_model = mask_model

        self.body_image_files = [f for f in os.listdir(body_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        failed_files = []
        for failed_captions_file in failed_captions_files:
            with open(failed_captions_file, "r") as f:
                failed_files += [line.strip().split('.')[0] for line in f.readlines()]
        failed_files = set(failed_files)
        self.body_image_files = [f for f in self.body_image_files if os.path.splitext(f)[0] not in failed_files]

        if shuffle:
            random.seed(seed)
            random.shuffle(self.body_image_files)

        # train / test split
        split_idx = int(len(self.body_image_files) * train_test_split)
        self.train_files = self.body_image_files[:split_idx]
        self.test_files = self.body_image_files[split_idx:]

        print(f"Total Image files in dataset are {len(self.body_image_files)}")
        print(f"Total train image files are {len(self.train_files)}")
        print(f"Total test image files are {len(self.test_files)}")

        self.body_caption_cache = {f: self._load_caption_file(self.body_captions_dir, f) for f in self.body_image_files}
        self.head_caption_cache = {f: self._load_caption_file(self.head_captions_dir, f) for f in self.body_image_files}

        self.current_position = self.batch_size * self.ddp_rank

    def _load_caption_file(self, dir: str, filename: str):
        path = os.path.join(dir, os.path.splitext(filename)[0] + ".txt")
        with open(path, "r") as f:
            return f.read().strip().lower()

    def load_caption(self, t: str, filename: str):
        if t == "body":
            return self.body_caption_cache[filename]
        else:
            return self.head_caption_cache[filename]

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

    def _soften_mask(self, mask: np.ndarray, expand_px=10, sigma=1.5):
        """
        Expand mask boundaries outward by 'expand_px' pixels,
        then soften edges slightly, returning a binary mask (0s, 1s).
        """
        # mask = (mask > 0.5).astype(float)
        for i in range(len(mask)):
            mask[i] = binary_dilation(mask[i], iterations=expand_px)
        blurred = gaussian_filter(mask.astype(float), sigma=sigma)
        softened_mask = (blurred > 0.5).astype(np.uint8)
        return softened_mask

    def make_head_mask(self,
                    body_hair_masks: np.ndarray,
                    body_face_masks: np.ndarray,
                    head_hair_masks: np.ndarray,
                    head_face_masks: np.ndarray
                    ):
        B, H, W = body_face_masks.shape
        combined_body = np.logical_or(body_face_masks, body_hair_masks)
        combined_head = np.zeros_like(body_face_masks)
        for i in range(B):
            head_combined = np.logical_or(head_face_masks[i], head_hair_masks[i])

            if combined_body[i].sum() > 0 and head_combined.sum() > 0:
                body_center = np.array(center_of_mass(combined_body[i]))
                head_center = np.array(center_of_mass(head_combined))

                shift = np.round(body_center - head_center).astype(int)
                aligned_head = self._shift_mask(head_combined, shift, H, W)
            else:
                aligned_head = head_combined

            combined_head[i] = aligned_head

        head_mask = np.logical_or(combined_body, combined_head).astype(np.uint8)
        head_mask = self._soften_mask(head_mask)

        downsampled_head_mask = np.array([
            np.array(
                Image.fromarray((m * 255).astype(np.uint8))
                .resize((128, 128), resample=Image.BILINEAR)
            ).astype(np.float32) / 255.0
            for m in head_mask
        ])

        return head_mask, downsampled_head_mask

    def _make_batch_from_files(self, body_files: list, head_files: list):
        body_images, body_hair_masks, body_face_masks, body_captions = [], [], [], []
        for f in body_files:
            body_images.append(Image.open(os.path.join(self.body_images_dir, f)).convert("RGB"))
            hair_mask, face_mask = self.process_mask(os.path.join(self.masks_dir, os.path.splitext(f)[0] + ".png"))
            body_hair_masks.append(hair_mask)
            body_face_masks.append(face_mask)
            body_captions.append(self.load_caption("body", f))
        body_hair_masks = np.stack(body_hair_masks)
        body_face_masks = np.stack(body_face_masks)

        head_images, head_hair_masks, head_face_masks, head_captions = [], [], [], []
        for f in head_files:
            head_images.append(Image.open(os.path.join(self.head_images_dir, f)).convert("RGB"))
            head_captions.append(self.load_caption("head", f))

        results = self.mask_model.get_masks(head_images, return_type = "numpy")
        _, head_face_masks, head_hair_masks = results
        head_hair_masks = np.array(head_hair_masks)/255.0
        head_face_masks = np.array(head_face_masks)/255.0

        head_masks, downsampled_head_masks = self.make_head_mask(body_hair_masks, body_face_masks, head_hair_masks, head_face_masks)

        return (
            body_images,
            head_images,
            head_hair_masks,
            body_hair_masks,
            body_captions,
            head_captions,
            head_masks,
            downsampled_head_masks,
        )

    def get_train_batch(self):
        n = len(self.train_files)
        selected_files = [self.train_files[(self.current_position + i) % n] for i in range(self.batch_size)]

        self.current_position += self.batch_size * self.ddp_world_size
        if self.current_position >= n:
            self.current_position %= n

        return self._make_batch_from_files(selected_files, selected_files)

    def get_test_batch(self):
        files = self.test_files
        # half_batch = self.batch_size // 2

        selected_body_files_1 = files[:self.batch_size]
        selected_body_files_2 = random.sample(files, self.batch_size)
        selected_head_files_2 = random.sample(files, self.batch_size)

        return self._make_batch_from_files(selected_body_files_1[self.batch_size // 2:] + selected_body_files_2[:self.batch_size // 2] + selected_body_files_1[:self.batch_size // 2] + selected_body_files_2[self.batch_size // 2:], selected_body_files_1[self.batch_size // 2:] + selected_head_files_2[:self.batch_size // 2] + files[1:self.batch_size // 2 + 1] + selected_head_files_2[self.batch_size // 2:])
    
        # selected_body_files_1 = files[:half_batch]
        # selected_body_files_2 = random.sample(files, self.batch_size - half_batch)
        # selected_head_files_2 = random.sample(files, self.batch_size - half_batch)

        # return self._make_batch_from_files(selected_body_files_1[:half_batch // 2] + selected_body_files_2[:half_batch // 2] + selected_body_files_1[half_batch // 2:] + selected_body_files_2[half_batch // 2:], selected_body_files_1[:half_batch // 2] + selected_head_files_2[:half_batch // 2] + files[1:half_batch // 2 + 1] + selected_head_files_2[half_batch // 2:])
    
    def state_dict(self):
        global_step = self.current_position // (self.batch_size * self.ddp_world_size)
        return {
            "global_step": global_step,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict):
        global_step = state["global_step"]
        self.seed = state["seed"]

        self.current_position = global_step * self.batch_size * self.ddp_world_size
        self.current_position += self.batch_size * self.ddp_rank

        random.seed(self.seed)

import queue
import threading

class PrefetchDataLoader:
    """
    Threaded batch prefetcher for custom DataLoader classes
    that implement get_train_batch() and get_test_batch().
    """

    def __init__(self, dataloader, mode: str = "train", num_workers: int = 2, prefetch_batches: int = 4):
        assert mode in ("train", "test"), "mode must be 'train' or 'test'"
        self.dataloader = dataloader
        self.mode = mode
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches

        self._queue = queue.Queue(maxsize=prefetch_batches)
        self._stop_event = threading.Event()
        self._workers = []
        self._initialized = False

    def _start_workers(self):
        """Start worker threads that fetch data asynchronously."""
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._workers.append(worker)
            worker.start()
        self._initialized = True

    def _worker_loop(self):
        """Worker thread: continuously fetch batches and enqueue them."""
        get_batch = (
            self.dataloader.get_train_batch
            if self.mode == "train"
            else self.dataloader.get_test_batch
        )

        while not self._stop_event.is_set():
            try:
                batch = get_batch()
                if batch is None:
                    # Treat None as end of data
                    self._queue.put(None)
                    break
                self._queue.put(batch)
            except StopIteration:
                self._queue.put(None)
                break
            except queue.Full:
                # Back off a bit when queue is full
                continue
            except Exception as e:
                print(f"[PrefetchDataLoader] Worker error: {e}")
                self._stop_event.set()
                break

    def __iter__(self):
        if not self._initialized:
            self._start_workers()
        return self

    def __next__(self):
        batch = self._queue.get()
        if batch is None:
            self.shutdown()
            raise StopIteration
        return batch

    def shutdown(self):
        """Gracefully stop worker threads and clear queue."""
        self._stop_event.set()
        for w in self._workers:
            w.join(timeout=1)
        self._workers.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._initialized = False