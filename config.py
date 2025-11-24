import yaml
import torch
from typing import List
from dataclasses import dataclass

@dataclass
class TrainConfig:
    cache_dir: str
    model_config_path: str
    dtype: torch.dtype
    output_dir: str
    total_steps: int
    
    use_wandb: bool
    wandb_project: str
    wandb_name: str

    body_images_dir: str
    head_images_dir: str
    masks_dir: str
    body_captions_dir: str
    head_captions_dir: str
    failed_captions_files: List[str]
    train_test_split: float
    batch_size: int
    num_dataloader_workers: int
    num_prefetch_data_batches: int

    grad_accum_steps: int
    checkpoint_saving_interval: int
    val_interval: int
    val_steps: int
    val_print: bool
    log_interval: int
    max_checkpoints: int

    invert_till: int
    num_inference_steps: int
    io_mask_head_condition_cfg: float

    noise_pred_loss_weight: float
    head_mask_loss_weight: float

    hopenet_model_path: str

    @staticmethod
    def from_yaml(path: str) -> "TrainConfig":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        # map dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        cfg["dtype"] = dtype_map.get(cfg.get("dtype", "float32"), torch.float32)

        return TrainConfig(**cfg)