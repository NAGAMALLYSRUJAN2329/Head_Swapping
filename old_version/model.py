from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
import torch
import yaml
import itertools
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Optional, Union, List, Dict
import torch.nn.functional as F

from modules.adapter import Adapter
from modules.adapters import HairEncoderAdapter
from modules.clip_image_encoder import CLIPImageEncoder
from modules.hair_encoder import HairEncoder
from modules.identity_encoder import InsightFaceEncoder
from modules.qformer import QFormer
from modules.schp_model import SCHPModel


class MyStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        self.vae: Optional[AutoencoderKL] = None
        self.text_encoder: Optional[CLIPTextModel] = None
        self.text_encoder_2: Optional[CLIPTextModelWithProjection] = None
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.tokenizer_2: Optional[CLIPTokenizer] = None
        self.unet: Optional[UNet2DConditionModel] = None
        self.scheduler: Optional[KarrasDiffusionSchedulers] = None
        self.image_encoder: Optional[CLIPVisionModelWithProjection] = None
        self.feature_extractor: Optional[CLIPImageProcessor] = None
        self.id_encoder: Optional[InsightFaceEncoder] = None
        self.clip_img_encoder: Optional[CLIPImageEncoder] = None
        self.hair_encoder: Optional[HairEncoder] = None
        self.hair_encoder_adapter_1: Optional[HairEncoderAdapter] = None
        self.id_encoder_adapter: Optional[Adapter] = None
        self.id_clip_encoder_adapter: Optional[Adapter] = None
        self.hair_clip_encoder_adapter: Optional[Adapter] = None
        self.hair_encoder_adapter_2: Optional[Adapter] = None
        self.hair_fusion_module: Optional[QFormer] = None
        self.id_fusion_module: Optional[QFormer] = None
        self.mask_model: Optional[SCHPModel] = None

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, config_path=None, **kwargs):
        pipe = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if config_path:
            pipe._init_custom_modules(yaml.safe_load(open(config_path)))
        return pipe
    
    def _init_custom_modules(self, cfg):
        registry = {
            "InsightFaceEncoder": InsightFaceEncoder,
            "CLIPImageEncoder": CLIPImageEncoder,
            "HairEncoder": HairEncoder,
            "HairEncoderAdapter": HairEncoderAdapter,
            "QFormer": QFormer,
            "Adapter": Adapter,
            "SCHPModel": SCHPModel,
        }

        for name, entry in cfg.items():
            cls = registry[entry["class"]]
            params = entry.get("params", {})
            module = cls(**params)
            # type(self).__annotations__[name] = cls
            setattr(self, name, module.to(self.device, dtype=self.dtype))
        
        self.trainable_modules = [
            self.id_encoder_adapter,
            self.id_clip_encoder_adapter,
            self.hair_encoder_adapter_1,
            self.hair_encoder_adapter_2,
            self.hair_clip_encoder_adapter,
            self.hair_fusion_module,
            self.id_fusion_module,
        ]
        self.non_trainable_modules = [
            self.vae,
            self.unet,
            self.text_encoder,
            self.text_encoder_2,
            self.id_encoder,
            self.clip_img_encoder,
            self.hair_encoder,
            self.mask_model,
        ]
        # pipe.id_encoder # (B, 3, 1957, 1250) -> (B, 512)
        # pipe.id_encoder_adapter # (B, 512) -> (B, 1024)
        # pipe.clip_img_encoder # (B, 3, 224, 224) -> (B, 768)
        # pipe.id_clip_encoder_adapter # (B, 768) -> (B, 1024)
        # pipe.hair_encoder # (B, 3, 512, 512) -> (B, 18, 512)
        # pipe.hair_encoder_adapter_1 # (B, 18, 512) -> (B, 512)
        # pipe.hair_encoder_adapter_2 # (B, 512) -> (B, 1024)
        # pipe.hair_clip_encoder_adapter # (B, 768) -> (B, 1024)
        # pipe.hair_fusion_module # (B, 3, 2048) -> (B, 1, 2048)
        # pipe.id_fusion_module # (B, 3, 2048) -> (B, 1, 2048)

    def to(self, device):
        super().to(device)
        self.id_encoder.to(self.device, dtype = self.dtype)
        self.id_encoder_adapter.to(self.device, dtype = self.dtype)
        self.clip_img_encoder.to(self.device, dtype = self.dtype)
        self.id_clip_encoder_adapter.to(self.device, dtype = self.dtype)
        self.hair_encoder.to(self.device, dtype = self.dtype)
        self.hair_encoder_adapter_1.to(self.device, dtype = self.dtype)
        self.hair_encoder_adapter_2.to(self.device, dtype = self.dtype)
        self.hair_clip_encoder_adapter.to(self.device, dtype = self.dtype)
        self.hair_fusion_module.to(self.device, dtype = self.dtype)
        self.id_fusion_module.to(self.device, dtype = self.dtype)
        self.mask_model.to(self.device, dtype = self.dtype)
        return self

    def save_adapters(self, path):
        adapters = {
            "id_encoder_adapter": self.id_encoder_adapter.state_dict(),
            "id_clip_encoder_adapter": self.id_clip_encoder_adapter.state_dict(),
            "hair_encoder_adapter_1": self.hair_encoder_adapter_1.state_dict(),
            "hair_encoder_adapter_2": self.hair_encoder_adapter_2.state_dict(),
            "hair_clip_encoder_adapter": self.hair_clip_encoder_adapter.state_dict(),
            "hair_fusion_module": self.hair_fusion_module.state_dict(),
            "id_fusion_module": self.id_fusion_module.state_dict(),
        }
        torch.save(adapters, path)

    def load_adapters(self, path, strict=True):
        adapters = torch.load(path, map_location="cpu")
        self.id_encoder_adapter.load_state_dict(adapters["id_encoder_adapter"], strict=strict)
        self.id_clip_encoder_adapter.load_state_dict(adapters["id_clip_encoder_adapter"], strict=strict)
        self.hair_encoder_adapter_1.load_state_dict(adapters["hair_encoder_adapter_1"], strict=strict)
        self.hair_encoder_adapter_2.load_state_dict(adapters["hair_encoder_adapter_2"], strict=strict)
        self.hair_clip_encoder_adapter.load_state_dict(adapters["hair_clip_encoder_adapter"], strict=strict)
        self.hair_fusion_module.load_state_dict(adapters["hair_fusion_module"], strict=strict)
        self.id_fusion_module.load_state_dict(adapters["id_fusion_module"], strict=strict)

    def trainable_parameters(self):
        return itertools.chain(*[module.parameters() for module in self.trainable_modules])

    def parameters(self, recurse: bool = True):
        return itertools.chain(*[module.parameters() for module in (self.trainable_modules + self.non_trainable_modules)])
    
    def set_eval_mode(self):
        for module in (self.trainable_modules + self.non_trainable_modules):
            module.eval()

    def set_train_mode(self):
        for module in self.non_trainable_modules:
            module.eval()
        for module in self.trainable_modules:
            module.train()
    
    def print_param_info(self):
        id_encoder_params = self.id_encoder.num_parameters()
        id_encoder_adapter_params = sum(p.numel() for p in self.id_encoder_adapter.parameters())
        clip_img_encoder_params = sum(p.numel() for p in self.clip_img_encoder.parameters())
        id_clip_encoder_adapter_params = sum(p.numel() for p in self.id_clip_encoder_adapter.parameters())
        hair_encoder_params = sum(p.numel() for p in self.hair_encoder.parameters())
        hair_encoder_adapter_1_params = sum(p.numel() for p in self.hair_encoder_adapter_1.parameters())
        hair_encoder_adapter_2_params = sum(p.numel() for p in self.hair_encoder_adapter_2.parameters())
        hair_clip_encoder_adapter_params = sum(p.numel() for p in self.hair_clip_encoder_adapter.parameters())
        hair_fusion_module_params = sum(p.numel() for p in self.hair_fusion_module.parameters())
        id_fusion_module_params = sum(p.numel() for p in self.id_fusion_module.parameters())
        mask_model_params = sum(p.numel() for p in self.mask_model.parameters())
        
        vae_params = sum(p.numel() for p in self.vae.parameters())
        unet_params = sum(p.numel() for p in self.unet.parameters())
        text_encoder_1_params = sum(p.numel() for p in self.text_encoder.parameters())
        text_encoder_2_params = sum(p.numel() for p in self.text_encoder_2.parameters())

        total_params = (
            id_encoder_params
            + id_encoder_adapter_params
            + clip_img_encoder_params
            + id_clip_encoder_adapter_params
            + hair_encoder_params
            + hair_encoder_adapter_1_params
            + hair_encoder_adapter_2_params
            + hair_clip_encoder_adapter_params
            + hair_fusion_module_params
            + id_fusion_module_params
            + mask_model_params
            + vae_params
            + unet_params
            + text_encoder_1_params
            + text_encoder_2_params
        )

        print(f"vae: {vae_params:,}")
        print(f"unet: {unet_params:,}")
        print(f"text_encoder: {text_encoder_1_params:,}")
        print(f"text_encoder_2: {text_encoder_2_params:,}")
        print(f"id_encoder: {id_encoder_params:,}")
        print(f"id_encoder_adapter: {id_encoder_adapter_params:,}")
        print(f"clip_img_encoder: {clip_img_encoder_params:,}")
        print(f"id_clip_encoder_adapter: {id_clip_encoder_adapter_params:,}")
        print(f"hair_encoder: {hair_encoder_params:,}")
        print(f"hair_encoder_adapter_1: {hair_encoder_adapter_1_params:,}")
        print(f"hair_encoder_adapter_2: {hair_encoder_adapter_2_params:,}")
        print(f"hair_clip_encoder_adapter: {hair_clip_encoder_adapter_params:,}")
        print(f"hair_fusion_module: {hair_fusion_module_params:,}")
        print(f"id_fusion_module: {id_fusion_module_params:,}")
        print(f"mask_model: {mask_model_params:,}")
        print(f"Total parameters: {total_params:,}")

        # vae: 83,653,863
        # unet: 2,567,463,684
        # text_encoder: 123,060,480
        # text_encoder_2: 694,659,840
        # id_encoder: 43,590,976
        # id_encoder_adapter: 1,050,624
        # clip_img_encoder: 427,616,513
        # id_clip_encoder_adapter: 1,574,912
        # hair_encoder: 48,524,480
        # hair_encoder_adapter_1: 19
        # hair_encoder_adapter_2: 1,050,624
        # hair_clip_encoder_adapter: 1,574,912
        # hair_fusion_module: 67,160,064
        # id_fusion_module: 67,160,064
        # mask_model: 66,723,928
        # Total parameters: 4,194,864,983

    def updated_encode_prompt(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Union[str, List[str]] = "",
            update_prompt_embeddings: bool = False,
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = False,
            id_images: Optional[List[Image.Image]] = None,
            hair_images: Optional[List[Image.Image]] = None,
            ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt = negative_prompt,
                device = device or self.device,
                num_images_per_prompt = num_images_per_prompt,
                do_classifier_free_guidance = do_classifier_free_guidance,
                )
        
        if update_prompt_embeddings:
            assert all(('man' in p) or ('woman' in p) for p in prompt), "Each prompt must contain 'man' or 'woman'"
            assert all(('hairstyle' in p) for p in prompt), "Each prompt must contain 'hairstyle'"
            hairstyle_token = self.tokenizer("hairstyle").input_ids[1:-1][0] # 22324
            if "man" in prompt:
                gender_token = self.tokenizer("man").input_ids[1:-1][0] # 786
            else:
                gender_token = self.tokenizer("woman").input_ids[1:-1][0] # 2308
            prompt_tokens_ids = self.tokenizer(
                prompt,
                padding = "max_length",
                max_length = self.tokenizer.model_max_length,
                truncation = True,
                return_tensors = "pt",
            ).input_ids
            mask = (prompt_tokens_ids == hairstyle_token)
            idxs = torch.where(mask, torch.arange(prompt_tokens_ids.size(1)).expand_as(prompt_tokens_ids), prompt_tokens_ids.size(1))
            hairstyle_idxs = idxs.min(dim=1).values
            mask  = (prompt_tokens_ids == gender_token)
            idxs = torch.where(mask, torch.arange(prompt_tokens_ids.size(1)).expand_as(prompt_tokens_ids), prompt_tokens_ids.size(1))
            gender_idxs = idxs.min(dim=1).values
            if (hairstyle_idxs == 77).any() or (gender_idxs == 77).any():
                print("Gender or hairstyle token not found in required length of the prompt (77) as Truncation is set to True!, for these prompts")
                # print(prompt)
                return prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds
                raise ValueError("Gender or hairstyle token not found in required length of the prompt (77) as Truncation is set to True!")
            B = prompt_embeds.size(0)
            hairstyle_text_vectors = prompt_embeds[torch.arange(B), hairstyle_idxs]
            gender_text_vectors = prompt_embeds[torch.arange(B), gender_idxs]
            updated_vectors = self.fuse_embeddings(id_images, hair_images, gender_text_vectors, hairstyle_text_vectors)
            prompt_embeds[torch.arange(B), gender_idxs] = updated_vectors["updated_gender_text_vector"].squeeze(1)
            prompt_embeds[torch.arange(B), hairstyle_idxs] = updated_vectors["updated_hairstyle_text_vector"].squeeze(1)
        
        return prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds
    
    def fuse_embeddings(
        self,
        id_images: List[Image.Image],
        hair_images: List[Image.Image],
        gender_text_vectors: torch.Tensor, # (B, 2048)
        hairstyle_text_vectors: torch.Tensor, # (B, 2048)
        return_dict: bool = False,
        ):
        id_images = self.from_pil_image_batch(id_images) # (B, 3, 1024, 1024)
        hair_images = self.from_pil_image_batch(hair_images) # (B, 3, 1024, 1024)

        # with torch.no_grad():
        id_encoder_features = self.id_encoder(id_images) # (B, 512)
        clip_id_features = self.clip_img_encoder(id_images) # (B, 768)
        clip_hair_features = self.clip_img_encoder(hair_images) # (B, 768)
        hair_encoder_features = self.hair_encoder(hair_images) # (B, 18, 512)

        up_sampled_id_encoder_features = self.id_encoder_adapter(id_encoder_features) # (B, 2048)
        up_sampled_clip_id_features = self.id_clip_encoder_adapter(clip_id_features) # (B, 2048)
        up_sampled_clip_hair_features = self.hair_clip_encoder_adapter(clip_hair_features) # (B, 2048)
        compressed_hair_encoder_features = self.hair_encoder_adapter_1(hair_encoder_features) # (B, 512)
        up_sampled_hair_encoder_features = self.hair_encoder_adapter_2(compressed_hair_encoder_features) # (B, 2048)
        input_id_fusion_module = torch.stack([up_sampled_id_encoder_features, up_sampled_clip_id_features, gender_text_vectors], dim = 1) # (B, 3, 2048)
        input_hair_fusion_module = torch.stack([up_sampled_hair_encoder_features, up_sampled_clip_hair_features, hairstyle_text_vectors], dim = 1) # (B, 3, 2048)
        updated_gender_text_vector = self.id_fusion_module(input_id_fusion_module) # (B, 1, 2048)
        updated_hairstyle_text_vector = self.hair_fusion_module(input_hair_fusion_module) # (B, 1, 2048)

        if return_dict:
            return {
                "id_encoder_features": id_encoder_features,
                "up_sampled_id_encoder_features": up_sampled_id_encoder_features,
                "clip_id_features": clip_id_features,
                "up_sampled_clip_id_features": up_sampled_clip_id_features,
                "clip_hair_features": clip_hair_features,
                "up_sampled_clip_hair_features": up_sampled_clip_hair_features,
                "hair_encoder_features": hair_encoder_features,
                "compressed_hair_encoder_features": compressed_hair_encoder_features,
                "up_sampled_hair_encoder_features": up_sampled_hair_encoder_features,
                "input_id_fusion_module": input_id_fusion_module,
                "input_hair_fusion_module": input_hair_fusion_module,
                "updated_gender_text_vector": updated_gender_text_vector,
                "updated_hairstyle_text_vector": updated_hairstyle_text_vector,
                }
        else:
            return {
                "updated_gender_text_vector" : updated_gender_text_vector, 
                "updated_hairstyle_text_vector" : updated_hairstyle_text_vector,
                }

    def sample(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: str = "",
            start_step: int=0,
            start_latents: Optional[torch.Tensor]=None,
            guidance_scale: int = 1.0,
            num_inference_steps: int = 50,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = False,
            device: Optional[torch.device] = None,
            output_type: str = "pil_image",
            height: int = 1024,
            width: int = 1024,
            update_prompt_embeddings: bool = False,
            id_images: Optional[List[Image.Image]] = None,
            hair_images: Optional[List[Image.Image]] = None,
            apply_io_mask: bool = False,
            io_mask: Optional[torch.Tensor] = None,
            show_progress: bool = True,
            ):
        
        device=device or self.device
        if guidance_scale <= 1.0:
            do_classifier_free_guidance = False

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        if start_latents is None:
            start_latents = torch.randn((batch_size, 4, height//self.vae_scale_factor, width//self.vae_scale_factor), device=device, dtype=self.dtype)

        prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds = self.updated_encode_prompt(
            prompt,
            device = device,
            update_prompt_embeddings = update_prompt_embeddings,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance = do_classifier_free_guidance,
            negative_prompt = negative_prompt,
            id_images = id_images,
            hair_images = hair_images,
            )

        add_time_ids = torch.tensor([height, width, 0, 0, height, width], dtype=self.dtype, device=device)
        add_time_ids = add_time_ids.unsqueeze(0).repeat(batch_size * num_images_per_prompt, 1)
        add_text_embeds = pooled_prompt_embeds
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = start_latents.clone()

        for i in tqdm(range(start_step, num_inference_steps), disable=not show_progress):
            # t = self.scheduler.timesteps[i]
            t = torch.as_tensor(self.scheduler.timesteps[i], device=self.scheduler.timesteps.device)
            noise_pred = self.fp_one_step(
                latents=latents,
                timestep = t,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            next_latents = self.get_next_latent(
                noise_pred = noise_pred,
                latents = latents,
                timestep_idx = i
            )
            if apply_io_mask:
                if io_mask.ndim == 3:
                    io_mask = io_mask.unsqueeze(1)
                next_latents = latents * (1 - io_mask) + next_latents * (io_mask)
            latents = next_latents.to(dtype = latents.dtype)

        if output_type == "latent":
            return latents
        images = self.decode_images(latents)
        if output_type == "tensor_image":
            return images # in range of [-1, 1], shape of [B, 3, H, W]
        else:
            return self.to_pil_image_batch(images)
        
    def invert(
            self,
            images: List[Image.Image],
            prompt: Union[str, List[str]],
            start_latents: Optional[torch.Tensor] = None,
            negative_prompt: str = "",
            invert_till: Optional[int] = None,
            guidance_scale: int = 1.0,
            num_inference_steps: int = 50,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = False,
            device: Optional[torch.device] = None,
            return_dict: bool = False,
            show_progress: bool = True,
    ):
        device=device or self.device
        start_latents = self.encode_images(images)
        batch_size = start_latents.shape[0]
        prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds = self.updated_encode_prompt(
            prompt,
            device = device,
            update_prompt_embeddings = False,
            num_images_per_prompt = num_images_per_prompt,
            do_classifier_free_guidance = do_classifier_free_guidance,
            negative_prompt = negative_prompt,
            )
        
        height, width = start_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = (height, width)
        target_size = (height, width)
        add_time_ids = torch.tensor([original_size + (0, 0) + target_size], dtype=self.dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        add_text_embeds = pooled_prompt_embeds
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            batch_size = 1
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        if not invert_till:
            invert_till = num_inference_steps

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = reversed(self.scheduler.timesteps)  # -> [1, 21, 41, ...]
        latents = start_latents.clone()
        # intermediate_latents = []

        for i in tqdm(range(0, invert_till), disable = not show_progress):
            t = timesteps[i]
            noise_pred = self.fp_one_step(
                latents=latents,
                timestep = t,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            latents = self.get_prev_latent(
                noise_pred = noise_pred,
                latents = latents,
                timestep_idx = i
            )
            # intermediate_latents.append(latents)
        if return_dict:
            return latents, {"prompt_embeds": prompt_embeds, "added_cond_kwargs": added_cond_kwargs}
        
        return latents
        
    def encode_images(self, images: List[Image.Image]):
        images = torch.concat([self.from_pil_image(image).unsqueeze(0) for image in images])
        # with torch.no_grad():
        latents = self.vae.encode(images)
        latents = self.vae.config.scaling_factor * latents.latent_dist.sample()
        return latents
    
    def decode_images(self, latents: torch.Tensor):
        # with torch.no_grad():
        image_tensors = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        return image_tensors

    def get_next_latent(
            self, 
            noise_pred: torch.Tensor,
            latents: torch.Tensor,
            timestep_idx: Optional[int] = None,
            timestep: Optional[int] = None,
            num_inference_steps: Optional[int] = None,
            ):
        if timestep_idx is not None:
            t = self.scheduler.timesteps[timestep_idx]
            prev_t = self.scheduler.timesteps[min(self.scheduler.num_inference_steps - 1, timestep_idx+1)]
        else:
            t = timestep
            prev_t = timestep - (1000 // num_inference_steps)

        alpha_t = self.scheduler.alphas_cumprod[t]
        if t == 1:
            alpha_t_prev = torch.tensor(1.0, device=latents.device, dtype=latents.dtype)
        else:
            alpha_t_prev = self.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
        return latents

    def get_prev_latent(
            self, 
            noise_pred: torch.Tensor,
            latents: torch.Tensor,
            timestep_idx: Optional[int] = None,
            timestep: Optional[int] = None,
            num_inference_steps: Optional[int] = None,
            ):
        if timestep_idx is not None:
            t = reversed(self.scheduler.timesteps)[timestep_idx]
            last_t = reversed(self.scheduler.timesteps)[timestep_idx-1]
        else:
            t = timestep
            last_t = t - (1000 // num_inference_steps)
        alpha_t = self.scheduler.alphas_cumprod[t]
        if last_t == reversed(self.scheduler.timesteps)[-1]:
            alpha_t_last = self.scheduler.final_alpha_cumprod
        else:
            alpha_t_last = self.scheduler.alphas_cumprod[last_t]

        # latents = (latents - (1 - alpha_t_last).sqrt() * noise_pred) * (alpha_t.sqrt() / alpha_t_last.sqrt()) + (1 - alpha_t).sqrt() * noise_pred
        # sa * ((1 / sb) * x_tm1 + ((1 / a - 1) ** 0.5 - (1 / b - 1) ** 0.5) * eps_xt)
        latents = alpha_t.sqrt() * ((1/alpha_t_last.sqrt()) * latents + ((1/alpha_t - 1).sqrt() - (1/alpha_t_last - 1).sqrt()) * noise_pred)
        return latents

    def apply_masks(
            self,
            images: Union[List[Image.Image], torch.Tensor], # if torch.Tensor -> it will be of shape [B, C, H, W], Normal PIL image [H, W, C], later also add facility to take np.ndarray as input
            masks: Union[List[Image.Image], np.ndarray, torch.Tensor]):
        masked_images = []
        is_image_input = isinstance(images[0], Image.Image)
        for img, mask in zip(images, masks):
            if is_image_input:
                img = img.convert("RGB")
                img = np.array(img)

            if isinstance(mask, Image.Image):
                mask_np = np.array(mask)
            elif isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            elif isinstance(mask, np.ndarray):
                mask_np = mask
            else:
                raise TypeError("Mask must be PIL.Image, numpy array, or torch tensor.")

            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]

            mask_np = (mask_np > 0).astype(np.uint8)

            if is_image_input:
                img = img * mask_np[:, :, None]
            else:
                img = img * torch.from_numpy(mask_np).to(img.device).unsqueeze(0)

            if is_image_input:
                masked_images.append(Image.fromarray(img))
            else:
                masked_images.append(img)

        if is_image_input:
            return masked_images
        else:
            return torch.stack(masked_images)

    def get_io_mask(
            self,
            latents: torch.Tensor,
            t: int,
            body_images: List[Image.Image],
            body_h_masks: Union[Image.Image, np.ndarray, torch.Tensor],
            head_images: List[Image.Image],
            head_h_masks: Union[Image.Image, np.ndarray, torch.Tensor],
            head_condition_cfg: float,
            body_image_prompts: List[str],
            head_image_prompts: List[str],
            threshold: float = 0.6,
            sigma: float = 1.2,
            ):
        # latents -> no guidance scale used
        device = self.device
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.updated_encode_prompt(
            body_image_prompts,
            device = device,
            update_prompt_embeddings = True,
            do_classifier_free_guidance = False,
            id_images = body_images,
            hair_images = self.apply_masks(body_images, body_h_masks)
            )
        height, width = body_images[0].size
        original_size = (height, width)
        target_size = (height, width)
        add_time_ids = torch.tensor([original_size + (0, 0) + target_size], dtype=self.dtype, device=device)
        add_time_ids = add_time_ids.repeat(len(body_images), 1)
        add_text_embeds = pooled_prompt_embeds

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        body_images_noise = self.fp_one_step(
            latents = latents,
            timestep = t,
            prompt_embeds = prompt_embeds,
            added_cond_kwargs = added_cond_kwargs,
            guidance_scale = 1.0,
            do_classifier_free_guidance = False
            )
        
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.updated_encode_prompt(
            head_image_prompts,
            device = device,
            update_prompt_embeddings = True,
            do_classifier_free_guidance = True,
            id_images = head_images,
            hair_images = self.apply_masks(head_images, head_h_masks)
            )
        
        height, width = head_images[0].size
        original_size = (height, width)
        target_size = (height, width)
        add_time_ids = torch.tensor([original_size + (0, 0) + target_size], dtype=self.dtype, device=device)
        add_time_ids = add_time_ids.repeat(len(head_images), 1)
        add_text_embeds = pooled_prompt_embeds

        if True: # CFG
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        head_images_noise = self.fp_one_step(
            latents = latents,
            timestep = t,
            prompt_embeds = prompt_embeds,
            added_cond_kwargs = added_cond_kwargs,
            guidance_scale = head_condition_cfg,
            do_classifier_free_guidance = True
            )
        
        orth_vector = self.etheta_orth(body_images_noise, head_images_noise)
        iomask = self.apply_guassian_filter(orth_vector, threshold = threshold, sigma = sigma)
        
        return iomask

    def etheta_orth(
            self,
            etheta_cb: torch.Tensor,
            etheta_ch_phi: torch.Tensor):
        N, C, H, W = etheta_cb.shape
        cb_flat = etheta_cb.view(N, C, -1)  # (N, C, H*W)
        ch_flat = etheta_ch_phi.view(N, C, -1)
        
        dot = torch.sum(cb_flat * ch_flat, dim=1, keepdim=True)  # (N, 1, H*W)
        norm_cb_sq = torch.sum(cb_flat**2, dim=1, keepdim=True)  # (N, 1, H*W)
        
        norm_cb_sq = norm_cb_sq + 1e-8
        proj = dot / norm_cb_sq * cb_flat  # (N, C, H*W)
        
        # Orthogonalize
        orth_flat = ch_flat - proj
        orth = orth_flat.view(N, C, H, W)
        return orth

    def apply_guassian_filter(
            self,
            orth_vector: torch.Tensor,
            threshold: float = 0.6,
            sigma: float = 1.0):
        # Step 1: Compute magnitude along channel dimension

        magnitude = torch.norm(orth_vector, dim=1)  # shape (N, H, W)

        # Step 2: Normalize to [0, 1]
        mag_min = magnitude.amin(dim=(1,2), keepdim=True)
        mag_max = magnitude.amax(dim=(1,2), keepdim=True)
        normalized = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
        normalized = normalized.unsqueeze(1)  # shape (N, 1, H, W) for conv2d

        # Step 3: Create Gaussian kernel
        def gaussian_kernel(kernel_size=5, sigma=1.0):
            ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            kernel = kernel / kernel.sum()
            return kernel

        # Determine kernel size based on sigma
        kernel_size = int(2 * round(3 * sigma) + 1)
        kernel = gaussian_kernel(kernel_size, sigma).to(orth_vector.device, dtype = orth_vector.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,kH,kW)

        # Step 4: Apply Gaussian filter using conv2d
        padding = kernel_size // 2
        smoothed = F.conv2d(normalized, kernel, padding=padding)

        mask = smoothed.squeeze(1)  # shape (N, H, W)

        # Step 5: Threshold to get mask -> this step removes grad_fn from the mask, so if you need try thresholding in the output levl
        # mask = (smoothed >= threshold).float()
        # mask = torch.sigmoid(sharpness * (smoothed - threshold))

        return mask


    def fp_one_step(
            self, 
            latents: torch.Tensor,
            timestep: int,
            prompt_embeds: torch.Tensor,
            added_cond_kwargs: Dict[str, torch.Tensor],
            guidance_scale: int,
            do_classifier_free_guidance: bool,
            ):
        
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep) # does nothing for DDIM scheduler
        # with torch.no_grad():
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds, added_cond_kwargs = added_cond_kwargs)[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred
    
    def to_pil_image(self, image: torch.Tensor):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.to(dtype=torch.float32)
        image = image.permute(1, 2, 0).detach().cpu().numpy() # (3, 512, 512) -> (512, 512, 3)
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image)
        return image
    
    def to_pil_image_batch(self, image_tensors: torch.tensor):
        images = [self.to_pil_image(image) for image in image_tensors]
        return images

    def from_pil_image(self, image: Image.Image):
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1) # (512, 512, 3) -> (3, 512, 512)
        tensor = tensor * 2 - 1
        return tensor.to(dtype=self.dtype, device=self.device) # [-1 ,1]
    
    def from_pil_image_batch(self, images: List[Image.Image]):
        return torch.concat([self.from_pil_image(image).unsqueeze(0) for image in images], dim = 0) # could also use torch.stack
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
            self,
            body_images: List[Image.Image],
            body_images_hair_mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]],
            head_images: List[Image.Image],
            head_images_hair_mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]],
            body_image_captions: List[str],
            head_images_captions: List[str],
            io_mask_head_condition_cfg: float = 3.5,
            io_mask_threshold: float = 0.6,
            output_type: str = "pil_image",
            invert_till: int = 40,
            num_inference_steps: int = 50,
            show_progress: bool = True
            ):
        
        if body_images_hair_mask is None:
            results = self.mask_model.get_masks(body_images, return_type = "numpy")
            _, _, body_images_hair_mask = results
        if head_images_hair_mask is None:
            results = self.mask_model.get_masks(head_images, return_type = "numpy")
            _, _, head_images_hair_mask = results
        
        latents = self.invert(
            images = body_images,
            prompt = body_image_captions,
            invert_till = invert_till,
            num_inference_steps = num_inference_steps,
            show_progress = show_progress,
            )
        io_masks = self.get_io_mask(
            latents = latents,
            t = invert_till,
            body_images = body_images,
            body_h_masks = body_images_hair_mask,
            head_images = head_images,
            head_h_masks = head_images_hair_mask,
            head_condition_cfg = io_mask_head_condition_cfg,
            body_image_prompts = body_image_captions,
            head_image_prompts = head_images_captions,
            )
        thresholded_io_masks = (io_masks >= io_mask_threshold).float()
        start_step = 50 - invert_till
        # grad_timesteps = torch.randint(start_step, num_inference_steps, (batch_size,), device=self.device) # not required, as we are not training unet
        output_images = self.sample(
            prompt = body_image_captions,
            start_step = start_step,
            start_latents = latents,
            num_inference_steps = num_inference_steps,
            update_prompt_embeddings = True,
            id_images = head_images,
            hair_images = self.apply_masks(head_images, head_images_hair_mask),
            output_type = output_type,
            apply_io_mask = True,
            io_mask = thresholded_io_masks,
            show_progress = show_progress,
            )

        return output_images, io_masks
