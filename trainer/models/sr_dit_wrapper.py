import torch
import torch.nn as nn
from typing import List, Optional, Union, Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .base_wrapper import BaseWrapper
from .dinov3_feature_extractor import DinoV3FeatureExtractor
from ..si_loss import SILoss

class SRDiTPipeline(DiffusionPipeline):
    def __init__(self, transformer, vae, text_encoder, tokenizer, scheduler, vae_means=None, vae_stds=None):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler
        )
        self.vae_means = vae_means
        self.vae_stds = vae_stds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ):
        device = self.transformer.t_embedder.mlp[0].weight.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Correctly call text encoder with input_ids as positional or correct keyword
        # and handle different output types
        with torch.no_grad():
            outputs = self.text_encoder(
                text_inputs.input_ids.to(device),
                attention_mask=text_inputs.attention_mask.to(device),
                output_hidden_states=True
            )
            
            if hasattr(outputs, "hidden_states"):
                prompt_embeds = outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state"):
                prompt_embeds = outputs.last_hidden_state
            else:
                prompt_embeds = outputs[0]
            
            prompt_embeds = prompt_embeds.to(dtype=self.transformer.x_embedder.weight.dtype)

        latents = torch.randn(
            (batch_size, self.transformer.in_channels, height // 8, width // 8),
            generator=generator, device=device, dtype=prompt_embeds.dtype
        )
        
        cls_dim = getattr(self.transformer, "final_layer", None).linear_cls.out_features \
                  if hasattr(self.transformer, "final_layer") else 1024
        
        cls_latents = torch.randn((batch_size, cls_dim), generator=generator, device=device, dtype=prompt_embeds.dtype)

        # Manual Euler Sampling
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        
        for i in range(num_inference_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i+1]
            dt = t_next - t_curr
            
            t_batch = torch.full((batch_size,), t_curr, device=device, dtype=prompt_embeds.dtype)
            model_output, _, cls_output = self.transformer(latents, t_batch, prompt_embeds, cls_token=cls_latents)

            latents = latents + model_output * dt
            cls_latents = cls_latents + cls_output * dt

        # Inverse VAE normalization
        if self.vae_means is not None and self.vae_stds is not None:
            latents = latents * self.vae_stds.to(device=latents.device, dtype=latents.dtype) + \
                      self.vae_means.to(device=latents.device, dtype=latents.dtype)

        image = self.vae.decode(latents.to(dtype=self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return ImagePipelineOutput(images=self.numpy_to_pil(image))


class SRDiTWrapper(BaseWrapper):
    def __init__(self, transformer, vae, text_encoder, tokenizer, noise_scheduler, args=None):
        super().__init__(transformer, vae, text_encoder, tokenizer, noise_scheduler, args)
        self.dino = DinoV3FeatureExtractor(device='cpu') 
        self.si_loss = SILoss()
        
        # Flux VAE Normalization Constants (32 channels)
        means = [
            0.630743, -0.174794, 1.038644, 0.625652, -0.109543, -0.348630, 0.096801, -0.106808,
            -0.448995, 0.159081, -0.049913, -0.110386, 0.575314, 0.366637, -0.395181, -0.379614,
            0.010152, 0.812572, -0.138055, 0.251469, -0.046946, 0.340522, -0.010119, -0.014446,
            -0.341091, 0.493081, 0.361685, 0.238973, -0.410147, -0.577880, -0.152583, 0.007513
        ]
        stds = [
            1.595693, 1.707444, 1.352274, 1.801214, 1.487631, 1.612388, 1.553012, 1.412901,
            1.877872, 1.498435, 1.486805, 1.459984, 1.440838, 1.450051, 1.596210, 1.898887,
            1.520757, 1.634902, 1.503450, 1.818565, 1.615305, 1.576635, 1.584312, 1.532588,
            1.611043, 1.409536, 1.784642, 1.439150, 1.493293, 1.689721, 1.592106, 1.825309
        ]
        self.register_buffer("vae_means", torch.tensor(means).view(1, -1, 1, 1))
        self.register_buffer("vae_stds", torch.tensor(stds).view(1, -1, 1, 1))

    def encode_images(self, pixel_values, weight_dtype):
        with torch.no_grad():
            posterior = self.vae.encode(pixel_values.to(dtype=self.vae.dtype))
            latents = posterior.latent_dist.sample() if hasattr(posterior, "latent_dist") else posterior.latent
            
            # Apply per-channel normalization: (x - mean) / std
            latents = (latents - self.vae_means) / self.vae_stds
            return latents.to(dtype=weight_dtype)

    def forward(self, pixel_values, prompts, device, weight_dtype=torch.float32, **kwargs):
        if self.dino.device != device:
             self.dino.model.to(device)
             self.dino.device = device

        prompt_embeds = self.encode_text(prompts, device, weight_dtype)
        latents = self.encode_images(pixel_values, weight_dtype)
        
        # Get crop coords from kwargs if present
        crop_coords = kwargs.get("crop_coords", None)
        if crop_coords is not None:
            crop_coords = crop_coords.to(device=device, dtype=weight_dtype)
        
        with torch.no_grad():
            cls_target, patch_tokens = self.dino.extract_features(pixel_values)
            cls_target, patch_tokens = cls_target.to(dtype=weight_dtype), patch_tokens.to(dtype=weight_dtype)
            zs_target = [patch_tokens]

        loss_dict = self.si_loss(
            model=self.transformer,
            images=latents,
            model_kwargs={'context': prompt_embeds, 'y': crop_coords},
            zs=zs_target,
            cls_token=cls_target
        )
        
        denoising_loss, proj_loss, _, _, denoising_loss_cls, cfm_loss, cfm_loss_cls = loss_dict
        
        loss = denoising_loss + denoising_loss_cls + proj_loss + cfm_loss * 0.05 + cfm_loss_cls * 0.05
        
        return {
            "loss": loss,
            "loss_img": denoising_loss,
            "loss_cls": denoising_loss_cls,
            "loss_proj": proj_loss,
            "loss_cfm_img": cfm_loss,
            "loss_cfm_cls": cfm_loss_cls
        }

    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=20, guidance_scale=4.0, num_images=1, seed=None, device=None):
        if device is None:
            device = next(self.transformer.parameters()).device

        was_training = self.transformer.training
        self.transformer.eval() 
        
        pipeline = SRDiTPipeline(
            transformer=self.transformer, vae=self.vae, text_encoder=self.text_encoder,
            tokenizer=self.tokenizer, scheduler=self.noise_scheduler,
            vae_means=self.vae_means, vae_stds=self.vae_stds
        )
        
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None
        prompt = [prompt] * num_images if isinstance(prompt, str) else prompt
            
        images = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images
        if was_training: self.transformer.train()
        return images, prompt

