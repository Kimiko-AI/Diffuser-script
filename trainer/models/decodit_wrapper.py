import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Any
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .base_wrapper import BaseWrapper

class DecoDiTPipeline(DiffusionPipeline):
    def __init__(self, transformer, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler
        )

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
            
            # Ensure correct dtype
            prompt_embeds = prompt_embeds.to(dtype=self.transformer.s_embedder.weight.dtype)

        # Pixel space random latents (images)
        latents = torch.randn(
            (batch_size, self.transformer.in_channels, height, width),
            generator=generator, device=device, dtype=prompt_embeds.dtype
        )
        
        # Crop coords for inference? 
        # For inference we usually want the full image, so coords might be 0,0,1,1?
        # The model expects y of shape [B, 4].
        # Let's assume full crop: left=0, top=0, w=1, h=1
        y = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=prompt_embeds.dtype).repeat(batch_size, 1)

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for t in timesteps:
            # expand latents if doing classifier free guidance
            # (Simplification: implement standard CFG loop if needed, here just basic guidance placeholder logic)
            # But the loop below does manual Euler in previous code. 
            # Let's switch to scheduler.step which is standard.
            
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
            t_batch = torch.full((latent_model_input.shape[0],), t, device=device, dtype=prompt_embeds.dtype)
            
            prompt_embeds_input = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]) if guidance_scale > 1 else prompt_embeds
            y_input = torch.cat([y, y]) if guidance_scale > 1 else y
            
            noise_pred = self.transformer(latent_model_input, t_batch, prompt_embeds_input, y=y_input)
            
            if guidance_scale > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        image = (latents / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return ImagePipelineOutput(images=self.numpy_to_pil(image))


class DecoDiTWrapper(BaseWrapper):
    def __init__(self, transformer, vae, text_encoder, tokenizer, noise_scheduler, args=None):
        # vae is passed as None from loader
        super().__init__(transformer, vae, text_encoder, tokenizer, noise_scheduler, args)

    def forward(self, pixel_values, prompts, device, weight_dtype=torch.float32, **kwargs):
        prompt_embeds = self.encode_text(prompts, device, weight_dtype)
        
        # Pixel values are already images [B, 3, H, W]
        # Normalize to [-1, 1] if not already (Loader usually gives [-1, 1])
        # WebDataset bucket_batcher: transforms.Normalize([0.5], [0.5]) -> results in [-1, 1]
        
        clean_images = pixel_values.to(device=device, dtype=weight_dtype)
        
        # Sample noise
        noise = torch.randn_like(clean_images)
        bsz = clean_images.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # Get crop coords from kwargs if present
        crop_coords = kwargs.get("crop_coords", None)
        if crop_coords is not None:
            crop_coords = crop_coords.to(device=device, dtype=weight_dtype)
        
        # Predict the noise residual
        model_output = self.transformer(noisy_images, timesteps, prompt_embeds, y=crop_coords)
        
        # Loss calculation (assuming noise prediction)
        target = noise
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(clean_images, noise, timesteps)
        
        loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
        
        return {
            "loss": loss,
        }

    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=20, guidance_scale=4.0, num_images=1, seed=None, device=None):
        if device is None:
            device = next(self.transformer.parameters()).device

        was_training = self.transformer.training
        self.transformer.eval() 
        
        pipeline = DecoDiTPipeline(
            transformer=self.transformer, text_encoder=self.text_encoder,
            tokenizer=self.tokenizer, scheduler=self.noise_scheduler
        )
        
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None
        prompt = [prompt] * num_images if isinstance(prompt, str) else prompt
            
        images = pipeline(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images
        if was_training: self.transformer.train()
        return images, prompt