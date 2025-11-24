import os
import time
import torch
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from config import TrainConfig
from dataloader import DataLoader, PrefetchDataLoader
from model import MyStableDiffusionXLPipeline

# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py
# CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train.py

config = TrainConfig.from_yaml("demo_train_config.yaml")

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get("RANK"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
    ddp_world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    ddp_world_size = 1
    ddp_rank = 0
    master_process = True
    seed_offset = 0
    device = "cuda:5" if torch.cuda.is_available() else "cpu"

if config.use_wandb and master_process:
    import wandb
    wandb.init(
    project = config.wandb_project,
    name = config.wandb_name
    )

os.makedirs(config.output_dir, exist_ok=True)
training_checkpoints_dir = os.path.join(config.output_dir, 'training_checkpoints')
os.makedirs(training_checkpoints_dir, exist_ok=True)
model_weights_dir = os.path.join(config.output_dir, 'weights')
os.makedirs(model_weights_dir, exist_ok=True)
val_outputs_dir = os.path.join(config.output_dir, 'val_outputs')
os.makedirs(val_outputs_dir, exist_ok=True)

os.environ["TORCH_HOME"] = config.cache_dir  # for PyTorch model zoo / torchvision / LPIPS
os.environ["TRANSFORMERS_CACHE"] = config.cache_dir  # for Hugging Face Transformers & Diffusers
os.environ["HF_HOME"] = config.cache_dir  # for Hugging Face Transformers & Diffusers

torch.manual_seed(2345 + seed_offset)

model = MyStableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype = config.dtype, use_safetensors = True, cache_dir = config.cache_dir, config_path = config.model_config_path).to(device)

for p in model.parameters():
    p.requires_grad = False
for p in model.trainable_parameters():
    p.requires_grad = True

raw_model = model
# model = torch.compile(model)

start_step = 0
best_val_loss = float('inf')
best_loss = float('inf')

checkpoint_files = glob.glob(os.path.join(training_checkpoints_dir, 'checkpoint_*.pth'))
if checkpoint_files:
    latest_checkpoint_path = max(checkpoint_files, key=os.path.getctime)
    step_str = os.path.basename(latest_checkpoint_path).split('_')[-1].split('.')[0]
    start_step = int(step_str) + 1

    # Load model adapters (before DDP)
    raw_model_path = os.path.join(model_weights_dir, f'checkpoint_{step_str}.pth')
    model.load_adapters(raw_model_path)

    checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    best_loss = checkpoint.get('best_loss', float('inf'))
    print(f"Resuming from step {start_step}, checkpoint: {latest_checkpoint_path}")
else:
    print("No checkpoint found, starting from scratch.")

if ddp:
    # model = DDP(model, device_ids=[ddp_local_rank])
    # raw_model = model.module
    model.id_encoder_adapter      = DDP(model.id_encoder_adapter,      device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    model.id_clip_encoder_adapter = DDP(model.id_clip_encoder_adapter, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    model.hair_encoder_adapter_1  = DDP(model.hair_encoder_adapter_1,  device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    model.hair_encoder_adapter_2  = DDP(model.hair_encoder_adapter_2,  device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    model.hair_clip_encoder_adapter = DDP(model.hair_clip_encoder_adapter, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    model.hair_fusion_module      = DDP(model.hair_fusion_module,      device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    model.id_fusion_module        = DDP(model.id_fusion_module,        device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    model.unet                    = DDP(model.unet,                    device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    raw_model = model

print(f"Rank {ddp_rank} sees device {device}")
print(f"UNET param device: {next(model.unet.parameters()).device}")

optimizer = AdamW(raw_model.trainable_parameters(), lr = 1e-5, weight_decay=0.01)
if checkpoint_files:
    optimizer.load_state_dict(checkpoint['optimizer_state'])

if master_process:
    raw_model.print_param_info()
    print(f"Total Trainable Parameters are {sum(p.numel() for p in raw_model.trainable_parameters()):,}")

dataloader = DataLoader(
    mask_model = raw_model.mask_model,
    body_images_dir = config.body_images_dir,
    head_images_dir = config.head_images_dir,
    masks_dir = config.masks_dir,
    body_captions_dir = config.body_captions_dir,
    head_captions_dir = config.head_captions_dir,
    batch_size = config.batch_size,
    ddp_rank = ddp_rank,
    ddp_world_size = ddp_world_size,
    train_test_split = config.train_test_split,
    failed_captions_files = config.failed_captions_files
    )
if checkpoint_files:
    dataloader.load_state_dict(checkpoint['dataloader_state'])

train_dataloader = PrefetchDataLoader(dataloader, mode = "train", num_workers = config.num_dataloader_workers, prefetch_batches = config.num_prefetch_data_batches)
train_dataloader = iter(train_dataloader)

test_dataloader = PrefetchDataLoader(dataloader, mode = "test", num_workers = config.num_dataloader_workers, prefetch_batches = config.num_prefetch_data_batches)
test_dataloader = iter(test_dataloader)

def save_validation_images(step, body_images, head_images, output_images, io_masks, head_masks):
    save_dir = os.path.join(val_outputs_dir, str(step))
    os.makedirs(save_dir, exist_ok=True)
    for i, (b, h, o, io, hm) in enumerate(zip(body_images, head_images, output_images, io_masks, head_masks)):
        sample_dir = os.path.join(save_dir, str(i))
        os.makedirs(sample_dir, exist_ok=True)

        b.save(os.path.join(sample_dir, "body_image.jpg"))
        h.save(os.path.join(sample_dir, "head_image.jpg"))
        o.save(os.path.join(sample_dir, "output_image.jpg"))

        io_img = Image.fromarray((io * 255).detach().cpu().byte().numpy())
        io_img.save(os.path.join(sample_dir, "io_mask.png"))

        hm_img = Image.fromarray((hm * 255).astype(np.uint8))
        hm_img.save(os.path.join(sample_dir, "head_mask.png"))

        b_resized = b.resize((512, 512), Image.BICUBIC)
        h_resized = h.resize((512, 512), Image.BICUBIC)
        o_resized = o.resize((512, 512), Image.BICUBIC)
        io_img = io_img.resize((512, 512), Image.BICUBIC)

        combined = Image.new("RGB", (1024, 1024))
        combined.paste(b_resized, (0, 0))        # top-left
        combined.paste(h_resized, (512, 0))      # top-right
        combined.paste(o_resized, (0, 512))      # bottom-left
        combined.paste(io_img, (512, 512))       # bottom-right

        combined.save(os.path.join(sample_dir, "combined.jpg"))

def delete_excess_checkpoints():
    def get_step_from_filename(filename):
        return int(os.path.basename(filename).split("_")[1].split(".")[0])
    
    for ckpt_dir in [model_weights_dir, training_checkpoints_dir]:
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pth"))
        if len(ckpt_files) > config.max_checkpoints:
            ckpt_files = sorted(ckpt_files, key=get_step_from_filename)
            num_to_delete = len(ckpt_files) - config.max_checkpoints
            files_to_delete = ckpt_files[:num_to_delete]
            
            for f in files_to_delete:
                os.remove(f)

def save_checkpoints(step):
    raw_model.save_adapters(os.path.join(model_weights_dir, f'checkpoint_{step}.pth'))
    checkpoint = {
        'optimizer_state': optimizer.state_dict(),
        'dataloader_state': dataloader.state_dict(),
        'step': step,
        'best_val_loss': best_val_loss,
        'best_loss': best_loss,
    }
    torch.save(checkpoint, os.path.join(training_checkpoints_dir, f'checkpoint_{step}.pth'))
    delete_excess_checkpoints()

def validate_model(step, best_val_loss):
    raw_model.set_eval_mode()
    val_loss_accum_dict = {
        "val_noise_pred_loss": 0.0,
        "val_head_mask_loss": 0.0,
        "val_total_loss": 0.0,
    }
    for _ in range(config.val_steps):
        body_images, head_images, head_images_hair_masks, body_images_hair_masks, body_image_captions, head_images_captions, head_masks, downsampled_head_masks = next(test_dataloader)
        # body_images, head_images, head_images_hair_masks, body_images_hair_masks, body_image_captions, head_images_captions, head_masks, downsampled_head_masks = dataloader.get_test_batch()
        with torch.no_grad():
            output_images, io_masks = raw_model.predict(
                body_images = body_images[:config.batch_size],
                body_images_hair_mask = body_images_hair_masks[:config.batch_size],
                head_images = head_images[:config.batch_size],
                head_images_hair_mask = head_images_hair_masks[:config.batch_size],
                body_image_captions = body_image_captions[:config.batch_size],
                head_images_captions = head_images_captions[:config.batch_size],
                io_mask_head_condition_cfg = config.io_mask_head_condition_cfg,
                _io_masks = downsampled_head_masks,
                invert_till = config.invert_till,
                num_inference_steps = config.num_inference_steps,
                show_progress = False,
                output_type = 'tensor_image'
            )

            head_mask_loss = F.mse_loss(io_masks, torch.tensor(downsampled_head_masks[:config.batch_size], device = device))
            noise_pred_loss = F.mse_loss(output_images, raw_model.from_pil_image_batch(body_images[:config.batch_size]))
            val_total_loss = config.head_mask_loss_weight * head_mask_loss + config.noise_pred_loss_weight * noise_pred_loss

            val_loss_accum_dict["val_noise_pred_loss"] += noise_pred_loss / config.val_steps
            val_loss_accum_dict["val_head_mask_loss"] += head_mask_loss / config.val_steps
            val_loss_accum_dict["val_total_loss"] += val_total_loss / config.val_steps

            output_images_2, io_masks_2 = raw_model.predict(
                body_images = body_images[config.batch_size:],
                body_images_hair_mask = body_images_hair_masks[config.batch_size:],
                head_images = head_images[config.batch_size:],
                head_images_hair_mask = head_images_hair_masks[config.batch_size:],
                body_image_captions = body_image_captions[config.batch_size:],
                head_images_captions = head_images_captions[config.batch_size:],
                io_mask_head_condition_cfg = config.io_mask_head_condition_cfg,
                invert_till = config.invert_till,
                num_inference_steps = config.num_inference_steps,
                show_progress = False,
                output_type = 'tensor_image'
            )

    output_images = raw_model.to_pil_image_batch(torch.concat([output_images , output_images_2]))
    save_validation_images(step, body_images, head_images, output_images, torch.concat([io_masks, io_masks_2]), head_masks)
    best_val_loss = min(best_val_loss, val_loss_accum_dict['val_total_loss'].item())
    if config.use_wandb and master_process:
        wandb.log(val_loss_accum_dict, step=step)
    if config.val_print:
        log_str = f"[Step {step}/{config.total_steps}]  "
        log_str += f"Total Avg. Val Loss: {val_loss_accum_dict['val_total_loss']:.4f}  "
        component_losses = ", ".join(
            [f"{k}: {v:.4f}" for k, v in val_loss_accum_dict.items() if k != "val_total_loss"]
        )
        log_str += f"Components: {component_losses}  "
        print(log_str)

    raw_model.set_train_mode()
    return best_val_loss

pbar = tqdm(range(start_step, config.total_steps), desc="Training")
raw_model.set_train_mode()

for step in pbar:
    loss_accum_dict = {
        "noise_pred_loss": 0.0,
        "head_mask_loss": 0.0,
        "total_loss": 0.0,
    }
    optimizer.zero_grad()
    for micro_step in range(config.grad_accum_steps):
        # t1 = time.time()
        body_images, head_images, head_images_hair_masks, body_images_hair_masks, body_image_captions, head_images_captions, head_masks, downsampled_head_masks = next(train_dataloader)
        # body_images, head_images, head_images_hair_masks, body_images_hair_masks, body_image_captions, head_images_captions, head_masks, downsampled_head_masks = dataloader.get_train_batch()
        # print(f"Time taken for dataloader is {time.time() - t1}")
        # t1 = time.time()

        head_mask_loss, noise_pred_loss = model(
            body_images = body_images,
            body_images_hair_mask = body_images_hair_masks,
            head_images = head_images,
            head_images_hair_mask = head_images_hair_masks,
            body_image_captions = body_image_captions,
            head_images_captions = head_images_captions,
            io_mask_head_condition_cfg = config.io_mask_head_condition_cfg,
            invert_till = config.invert_till,
            num_inference_steps = config.num_inference_steps,
            show_progress = False,
            head_mask = downsampled_head_masks,
        )

        # print(f"Time taken for model forward pass is {time.time() - t1}")
        # t1 = time.time()

        loss = head_mask_loss * config.head_mask_loss_weight + noise_pred_loss * config.noise_pred_loss_weight
        loss = loss / config.grad_accum_steps

        if ddp:
            model.require_backward_grad_sync = (micro_step == config.grad_accum_steps - 1)

        loss.backward()

        loss_accum_dict["total_loss"] += loss.detach()
        loss_accum_dict["head_mask_loss"] += head_mask_loss.detach() / config.grad_accum_steps
        loss_accum_dict["noise_pred_loss"] += noise_pred_loss.detach() / config.grad_accum_steps

    norm = torch.nn.utils.clip_grad_norm_(raw_model.trainable_parameters(), 1.0)
    loss_accum_dict["grad_norm"] = norm
    
    optimizer.step()
    optimizer.zero_grad()

    if ddp:
        with torch.no_grad():
            for k in loss_accum_dict:
                tensor_val = torch.tensor(loss_accum_dict[k], device=raw_model.parameters().__next__().device)
                dist.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
                loss_accum_dict[k] = (tensor_val / dist.get_world_size()).item()
    else:
        for k in loss_accum_dict:
            loss_accum_dict[k] = float(loss_accum_dict[k])
    best_loss = min(best_loss, loss_accum_dict['total_loss'])

    if master_process:
        if config.use_wandb:
            wandb.log(loss_accum_dict, step=step)
        pbar.set_postfix({
            # "elapsed_sec": int(0),
            "loss": loss_accum_dict["total_loss"],
            "best_loss": best_loss,
            "best_val_loss": best_val_loss,
        })

    if ((step % config.checkpoint_saving_interval == 0 and step != 0) or step == config.total_steps - 1) and master_process:
        save_checkpoints(step)

    if ((step % config.log_interval == 0) or step == config.total_steps - 1) and master_process:
        log_str = f"[Step {step}/{config.total_steps}]  "
        log_str += f"Total Loss: {loss_accum_dict['total_loss']:.4f}  "
        component_losses = ", ".join(
            [f"{k}: {v:.4f}" for k, v in loss_accum_dict.items() if k != "total_loss"]
        )
        log_str += f"Components: {component_losses}  "
        print(log_str)
    
    if ((step % config.val_interval == 0) or step == config.total_steps - 1) and master_process:
        best_val_loss = validate_model(step, best_val_loss)

train_dataloader.shutdown()
test_dataloader.shutdown()

if ddp:
    destroy_process_group()