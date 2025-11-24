import os
import time
import torch
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from loss_function import LossFunction
from config import TrainConfig
from dataloader import DataLoader, PrefetchDataLoader
from model import MyStableDiffusionXLPipeline

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

if config.use_wandb:
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
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

# print(raw_model)

optimizer = AdamW(raw_model.trainable_parameters(), lr = 1e-5, weight_decay=0.01)
if checkpoint_files:
    optimizer.load_state_dict(checkpoint['optimizer_state'])

if master_process:
    raw_model.print_param_info()
    print(f"Total Trainable Parameters are {sum(p.numel() for p in raw_model.trainable_parameters()):,}")

dataloader = DataLoader(
    images_dir = config.images_dir,
    masks_dir = config.masks_dir,
    captions_dir = config.captions_dir,
    batch_size = config.batch_size,
    ddp_rank = ddp_rank,
    ddp_world_size = ddp_world_size,
    train_test_split = config.train_test_split,
    failed_captions_file = config.failed_captions_file
    )
if checkpoint_files:
    dataloader.load_state_dict(checkpoint['dataloader_state'])

train_dataloader = PrefetchDataLoader(dataloader, mode = "train", num_workers = config.num_dataloader_workers, prefetch_batches = config.num_prefetch_data_batches)
test_dataloader = PrefetchDataLoader(dataloader, mode = "test", num_workers = config.num_dataloader_workers, prefetch_batches = config.num_prefetch_data_batches)

loss_function = LossFunction(
    model = raw_model,
    lpips_loss_weight = config.lpips_loss_weight,
    pose_loss_weight = config.pose_loss_weight,
    head_mask_loss_weight = config.head_mask_loss_weight,
    hair_sim_loss_weight = config.hair_sim_loss_weight,
    id_sim_loss_weight = config.id_sim_loss_weight,
    recon_loss_weight = config.recon_loss_weight,
    hopenet_model_path = config.hopenet_model_path,
    device = device
    )
# loss_function = torch.compile(loss_function)
# loss_function.to(device)

def save_validation_images(step, body_images, head_images, output_images, io_masks, head_masks):
    save_dir = os.path.join(val_outputs_dir, str(step))
    os.makedirs(save_dir, exist_ok=True)
    for i, (b, h, o, io, hm) in enumerate(zip(body_images, head_images, output_images, io_masks, head_masks)):
        sample_dir = os.path.join(save_dir, str(i))
        os.makedirs(sample_dir, exist_ok=True)

        b.save(os.path.join(sample_dir, "body_image.jpg"))
        h.save(os.path.join(sample_dir, "head_image.jpg"))
        o.save(os.path.join(sample_dir, "output_image.jpg"))

        io_img = Image.fromarray((io * 255).cpu().byte().numpy())
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

pbar = tqdm(range(start_step, config.total_steps), desc="Training")
raw_model.set_train_mode()

for step in pbar:
    loss_accum_dict = {
        "lpips_loss": 0.0,
        "pose_loss": 0.0,
        "head_mask_loss": 0.0,
        "hair_sim_loss": 0.0,
        "id_sim_loss": 0.0,
        "recon_loss": 0.0,
        "total_loss": 0.0,
    }
    for micro_step in range(config.grad_accum_steps):
        t1 = time.time()
        body_images, head_images, head_images_hair_masks, body_images_hair_masks, body_image_captions, head_images_captions, head_images_face_masks, head_masks, downsampled_head_masks = next(train_dataloader)
        output_images, io_masks = model(
            body_images = body_images,
            body_images_hair_mask = body_images_hair_masks,
            head_images = head_images,
            head_images_hair_mask = head_images_hair_masks,
            body_image_captions = body_image_captions,
            head_images_captions = head_images_captions,
            io_mask_head_condition_cfg = config.io_mask_head_condition_cfg,
            output_type = "tensor_image",
            invert_till = config.invert_till,
            num_inference_steps = config.num_inference_steps,
            show_progress = False,
        )
        # print(output_images.grad_fn)
        # print(io_masks.grad_fn)
        print(f"Time taken for model forward pass is {time.time() - t1}")
        t1 = time.time()

        loss, loss_dict = loss_function(body_images, head_images, output_images, head_images_hair_masks, io_masks, head_masks, downsampled_head_masks, return_dict = True)
        loss = loss / config.grad_accum_steps
        print(f"Time taken for loss calculation is {time.time() - t1}")
        t1 = time.time()
        for k, v in loss_dict.items():
            loss_accum_dict[k] += v.detach() / config.grad_accum_steps
        if ddp:
            model.require_backward_grad_sync = (micro_step == config.grad_accum_steps - 1)
        print(loss)
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(raw_model.trainable_parameters(), 1.0)
    print("norm", norm)
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
            "elapsed_sec": int(0),
            "loss": loss_accum_dict["total_loss"],
            "best_loss": best_loss,
            "best_val_loss": best_val_loss,
        })

    if ((step % config.checkpoint_saving_interval == 0 and step != 0) or step == config.total_steps - 1) and master_process:
        raw_model.save_adapters(os.path.join(model_weights_dir, f'checkpoint_{step}.pth'))
        checkpoint = {
            'optimizer_state': optimizer.state_dict(),
            'dataloader_state': dataloader.state_dict(),
            'step': step,
            'best_val_loss': best_val_loss,
            'best_loss': best_loss,
        }
        torch.save(checkpoint, os.path.join(training_checkpoints_dir, f'checkpoint_{step}.pth'))

    if ((step % config.log_interval == 0 and step != 0) or step == config.total_steps - 1) and master_process:
        log_str = f"[Step {step}/{config.total_steps}]  "
        log_str += f"Total Loss: {loss_accum_dict['total_loss']:.4f}  "
        component_losses = ", ".join(
            [f"{k}: {v:.4f}" for k, v in loss_accum_dict.items() if k != "total_loss"]
        )
        log_str += f"Components: {component_losses}  "
        print(log_str)
    
    if ((step % config.val_interval == 0 and step != 0) or step == config.total_steps - 1) and master_process:
        raw_model.set_eval_mode()
        val_loss_accum_dict = {
            "val_lpips_loss": 0.0,
            "val_pose_loss": 0.0,
            "val_head_mask_loss": 0.0,
            "val_hair_sim_loss": 0.0,
            "val_id_sim_loss": 0.0,
            "val_recon_loss": 0.0,
            "val_total_loss": 0.0,
        }
        for val_step in range(config.val_steps):
            body_images, head_images, head_images_hair_masks, body_images_hair_masks, body_image_captions, head_images_captions, head_images_face_masks, head_masks, downsampled_head_masks = next(test_dataloader)
            with torch.no_grad():
                output_images, io_masks = model(
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
                    output_type = 'tensor_image'
                )
                loss, loss_dict = loss_function(body_images, head_images, output_images, head_images_hair_masks, io_masks, head_masks, downsampled_head_masks, return_dict = True)

                for k, v in loss_dict.items():
                    val_loss_accum_dict["val_" + k] += v.detach() / config.val_steps

        output_images = raw_model.to_pil_image_batch(output_images)
        save_validation_images(step, body_images, head_images, output_images, io_masks, head_masks)
        best_val_loss = min(best_val_loss, val_loss_accum_dict['val_total_loss'].item())
        if config.use_wandb:
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

train_dataloader.shutdown()
test_dataloader.shutdown()

if ddp:
    destroy_process_group()