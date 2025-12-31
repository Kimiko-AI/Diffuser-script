import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Optional, Union, Any
from diffusers import Lumina2Pipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig

class Lumina2Wrapper(nn.Module):
    def __init__(self, transformer, vae, text_encoder, tokenizer, noise_scheduler, args=None):
        super().__init__()
        self.transformer = transformer
        self._vae = [vae]
        self._text_encoder = [text_encoder]
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.args = args
        self.caption_dropout_prob = getattr(args, "caption_dropout_prob", 0.0)

        # Freeze frozen components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False) # Freeze base transformer first

        # Setup LoRA
        lora_rank = getattr(args, "lora_rank", 4)
        if lora_rank > 0:
            lora_alpha = getattr(args, "lora_alpha", lora_rank)
            lora_dropout = getattr(args, "lora_dropout", 0.0)
            lora_target_modules = getattr(args, "lora_target_modules", ["to_k", "to_q", "to_v", "to_out.0"])
            
            if isinstance(lora_target_modules, str):
                lora_target_modules = [x.strip() for x in lora_target_modules.split(",")]

            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights="gaussian",
                target_modules=lora_target_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)
            # Make sure adapter is trainable
            # add_adapter usually sets requires_grad for adapter params to True
        else:
            # If not LoRA, unfreeze transformer
            self.transformer.requires_grad_(True)
            if getattr(args, "gradient_checkpointing", False):
                self.transformer.enable_gradient_checkpointing()

        # Helper pipeline for encoding text
        # We initialize it with None for components we don't need immediately to save memory/time
        # But we need text_encoder and tokenizer
        self.text_encoding_pipeline = Lumina2Pipeline(
            vae=self.vae, # Needed for structure? No, from_pretrained usually handles it.
            # We construct it manually
            transformer=self.transformer, 
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler
        )

    @property
    def vae(self):
        return self._vae[0]

    @property
    def text_encoder(self):
        return self._text_encoder[0]
        
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32, device=None):
        # Lumina2 logic for sigmas from scheduler
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(
            self,
            pixel_values,
            prompts,
            device,
            weight_dtype=torch.float32,
            **kwargs
    ):
        # --- 1. Encode Text ---
        with torch.no_grad():
            if self.training and self.caption_dropout_prob > 0:
                prompts_in = ["" if random.random() < self.caption_dropout_prob else p for p in prompts]
            else:
                prompts_in = prompts

            max_sequence_length = getattr(self.args, "max_sequence_length", 256)
            system_prompt = getattr(self.args, "system_prompt", None)

            # Lumina2Pipeline.encode_prompt
            prompt_embeds, prompt_attention_mask, _, _ = self.text_encoding_pipeline.encode_prompt(
                prompt=prompts_in,
                max_sequence_length=max_sequence_length,
                system_prompt=system_prompt,
                device=device
            )

            prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
            prompt_attention_mask = prompt_attention_mask.to(device=device)

        # --- 2. Encode Images (VAE) ---
        with torch.no_grad():
            # AutoencoderKL
            posterior = self.vae.encode(pixel_values.to(dtype=self.vae.dtype)).latent_dist
            latents = posterior.sample()
            
            # Lumina2 scaling/shifting
            # (model_input - vae_config_shift_factor) * vae_config_scaling_factor
            # Check if shift_factor exists (SD3/Lumina logic usually has it)
            shift_factor = getattr(self.vae.config, "shift_factor", None)
            scaling_factor = self.vae.config.scaling_factor
            
            if shift_factor is not None:
                latents = (latents - shift_factor) * scaling_factor
            else:
                latents = latents * scaling_factor
                
            latents = latents.to(dtype=weight_dtype)

        # --- 3. Prepare Flow Matching ---
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)

        # Timestep sampling
        weighting_scheme = getattr(self.args, "weighting_scheme", "none") # Default none per script?
        # Script says default="none"
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=getattr(self.args, "logit_mean", 0.0),
            logit_std=getattr(self.args, "logit_std", 1.0),
            mode_scale=getattr(self.args, "mode_scale", 1.29),
        ).to(device)

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=device)

        # Add noise
        # zt = (1 - texp) * x + texp * z1
        # Lumina2 reverses lerp: sigma of 1.0 means `model_input`?
        # Script: noisy_model_input = (1.0 - sigmas) * noise + sigmas * model_input
        # Let's verify sigma definition.
        # If sigma=1 -> model_input (clean). If sigma=0 -> noise.
        # This is opposite to standard SD.
        
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=device)
        noisy_latents = (1.0 - sigmas) * noise + sigmas * latents

        # --- 4. Model Prediction ---
        # Scale timesteps 
        timesteps_norm = timesteps / self.noise_scheduler.config.num_train_timesteps
        
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps_norm,
            return_dict=False,
        )[0]

        # --- 5. Calculate Loss ---
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        
        # Target = model_input - noise
        target = latents - noise

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return loss

    @torch.no_grad()
    def generate(
            self,
            prompt: Union[str, List[str]],
            num_inference_steps: int = 20,
            guidance_scale: float = 4.0, # Default per Lumina usually
            num_images: int = 1,
            seed: Optional[int] = None,
            device: Optional[torch.device] = None,
            height: int = 512, # Lumina2 resolution
            width: int = 512,
            **kwargs
    ) -> List[Any]:
        if device is None:
            device = next(self.transformer.parameters()).device

        pipeline = Lumina2Pipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        generator = torch.Generator(device=device).manual_seed(seed) if seed else None
        
        images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            system_prompt=getattr(self.args, "system_prompt", None)
        ).images

        return images, prompt
