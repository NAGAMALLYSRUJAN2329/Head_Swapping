import lpips
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional, List
from PIL import Image

from modules.hopenet import HopeNet
from model import MyStableDiffusionXLPipeline

class LossFunction(nn.Module):
    def __init__(
            self, 
            model: MyStableDiffusionXLPipeline,
            lpips_loss_weight: float = 1.0,
            pose_loss_weight: float = 1.0,
            head_mask_loss_weight: float = 1.0,
            hair_sim_loss_weight: float = 1.0,
            id_sim_loss_weight: float = 1.0,
            recon_loss_weight: float = 1.0,
            hopenet_model_path: str = "module_models/hopenet_model.pkl",
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.bfloat16
            ):
        super().__init__()
        self.model = model
        self.device = device
        self.dtype = dtype
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to_tensor = transforms.ToTensor()
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device, self.dtype) # 2,470,848 Params
        self.hopenet_model = HopeNet(hopenet_model_path).to(self.device, self.dtype) # 23,919,890 Params

        self.lpips_loss_weight = lpips_loss_weight
        self.pose_loss_weight = pose_loss_weight
        self.head_mask_loss_weight = head_mask_loss_weight
        self.hair_sim_loss_weight = hair_sim_loss_weight
        self.id_sim_loss_weight = id_sim_loss_weight
        self.recon_loss_weight = recon_loss_weight

    def get_lpips_loss(self, body_images: List[Image.Image], output_images: torch.Tensor):
        body_images = self.model.from_pil_image_batch(body_images)
        dist = self.lpips_loss(body_images, output_images).view(-1)
        return sum(dist)/len(dist) # [B]
    
    def get_head_mask_loss(self, io_masks: torch.Tensor, downsampled_head_masks: np.ndarray):
        # io_masks already a tensor
        head_masks_tensor = torch.tensor(np.array(downsampled_head_masks), dtype=io_masks.dtype, device = io_masks.device)
        mask_loss = F.mse_loss(io_masks, head_masks_tensor)
        return mask_loss
    
    def get_pose_loss(self, body_images: List[Image.Image], output_images: torch.Tensor):
        # body_images = self.model.from_pil_image_batch(body_images)
        yaw1, pitch1, roll1 = self.hopenet_model(body_images)
        yaw2, pitch2, roll2 = self.hopenet_model(output_images)
        pose1 = torch.cat([yaw1, pitch1, roll1])
        pose2 = torch.cat([yaw2, pitch2, roll2])
        pose_loss = F.mse_loss(pose1, pose2)
        return pose_loss
    
    def non_mask_area_recon_loss(self, body_images: List[Image.Image], output_images: torch.Tensor, head_masks: np.ndarray):
        non_head_masks = 1 - head_masks
        body_images = self.model.from_pil_image_batch(body_images)
        body_images = self.model.apply_masks(body_images, non_head_masks)
        output_images = self.model.apply_masks(output_images, non_head_masks)
        recon_loss = F.mse_loss(output_images, body_images)
        return recon_loss
    
    def hair_sim_loss(self, head_images: List[Image.Image], output_images: torch.Tensor, head_images_hair_masks: np.ndarray):
        results = self.model.mask_model.get_masks(output_images, return_type = "numpy")
        _, _, output_image_hair_masks = results
        head_images = self.model.from_pil_image_batch(head_images)
        head_hair_images = self.model.apply_masks(head_images, head_images_hair_masks)
        output_hair_images = self.model.apply_masks(output_images, output_image_hair_masks)
        # self.model.to_pil_image(head_hair_images[0]).save("head_hair_image.jpg")
        # self.model.to_pil_image(output_hair_images[0]).save("output_hair_image.jpg")
        # head_hair_images = torch.stack([self.to_tensor(img) for img in head_hair_images]).to(self.device)
        # output_hair_images = torch.stack([self.to_tensor(img) for img in output_hair_images]).to(self.device)
        # hair_loss = F.mse_loss(head_hair_images, output_hair_images)
        with torch.no_grad():
            head_images_hair_embeddings = self.model.hair_encoder(head_hair_images)
            output_images_hair_embeddings = self.model.hair_encoder(output_hair_images)
        sim = F.cosine_similarity(head_images_hair_embeddings, output_images_hair_embeddings, dim=2)
        return 1 - sim.mean()
    
    def id_sim_loss(self, head_images: List[Image.Image], output_images: torch.Tensor):
        head_images = self.model.from_pil_image_batch(head_images)
        results = self.model.mask_model.get_masks(head_images, return_type = "numpy")
        _, head_images_face_masks, _ = results
        results = self.model.mask_model.get_masks(output_images, return_type = "numpy")
        _, output_image_face_masks, _ = results
        head_face_images = self.model.apply_masks(head_images, head_images_face_masks)
        output_face_images = self.model.apply_masks(output_images, output_image_face_masks)
        with torch.no_grad():
            head_images_face_embeddings = self.model.id_encoder(head_face_images)
            output_images_face_embeddings = self.model.id_encoder(output_face_images)
        sim = F.cosine_similarity(head_images_face_embeddings, output_images_face_embeddings, dim=1)
        return 1 - sim.mean()
    
    def clip_loss(self, head_images: List[Image.Image], output_images: torch.Tensor):
        # clip embeddings loss between head, output images, with full blur except face
        pass

    def forward(
            self,
            body_images: List[Image.Image],
            head_images: List[Image.Image],
            output_images: torch.Tensor,
            head_images_hair_masks: np.ndarray,
            io_masks: torch.Tensor,
            head_masks: np.ndarray,
            downsampled_head_masks: np.ndarray,
            return_dict: bool = False,
            return_weighted_loss: bool = True
            ):
        
        lpips_loss = self.get_lpips_loss(body_images, output_images)
        pose_loss = self.get_pose_loss(body_images, output_images)
        head_mask_loss = self.get_head_mask_loss(io_masks, downsampled_head_masks)
        hair_sim_loss = self.hair_sim_loss(head_images, output_images, head_images_hair_masks)
        id_sim_loss = self.id_sim_loss(head_images, output_images)
        recon_loss = self.non_mask_area_recon_loss(body_images, output_images, head_masks)

        total_loss = (
            self.lpips_loss_weight * lpips_loss +
            self.pose_loss_weight * pose_loss +
            self.head_mask_loss_weight * head_mask_loss +
            self.hair_sim_loss_weight * hair_sim_loss +
            self.id_sim_loss_weight * id_sim_loss +
            self.recon_loss_weight * recon_loss
        )

        if return_dict:
            if return_weighted_loss:
                return total_loss, {
                    "lpips_loss": self.lpips_loss_weight * lpips_loss,
                    "pose_loss": self.pose_loss_weight * pose_loss,
                    "head_mask_loss": self.head_mask_loss_weight * head_mask_loss,
                    "hair_sim_loss": self.hair_sim_loss_weight * hair_sim_loss,
                    "id_sim_loss": self.id_sim_loss_weight * id_sim_loss,
                    "recon_loss": self.recon_loss_weight * recon_loss,
                    "total_loss": total_loss,
                }
            else:
                return total_loss, {
                    "lpips_loss": lpips_loss,
                    "pose_loss": pose_loss,
                    "head_mask_loss": head_mask_loss,
                    "hair_sim_loss": hair_sim_loss,
                    "id_sim_loss": id_sim_loss,
                    "recon_loss": recon_loss,
                    "total_loss": total_loss,
                }
        return total_loss
