import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Optional, Union, Dict, Any
from diffusers import SanaPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

class SanaWrapper(nn.Module):
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
        
        # Ensure transformer is in train mode
        self.transformer.train()

        # Helper pipeline for encoding text
        self.text_encoding_pipeline = SanaPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer, # Pipeline requires transformer for init, though we won't use it for gen here
            scheduler=self.noise_scheduler
        )

    @property
    def vae(self):
        return self._vae[0]

    @property
    def text_encoder(self):
        return self._text_encoder[0]

    def forward(
        self, 
        pixel_values, 
        prompts, 
        device, 
        paraphrased_prompts: Optional[List[str]] = None, 
        weight_dtype=torch.float32,
        consistency_lambda: float = 0.0
    ):
        # --- 1. Encode Text ---
        with torch.no_grad():
            if self.training and self.caption_dropout_prob > 0:
                 prompts_in = ["" if random.random() < self.caption_dropout_prob else p for p in prompts]
            else:
                prompts_in = prompts

            # Sana specific text encoding parameters
            max_sequence_length = getattr(self.args, "max_sequence_length", 300)
            complex_human_instruction = getattr(self.args, "complex_human_instruction", None)
            
            prompt_embeds, prompt_attention_mask, _, _ = self.text_encoding_pipeline.encode_prompt(
                prompts_in,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=complex_human_instruction,
                device=device
            )
            
            # Cast to weight_dtype (likely bf16)
            prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
            prompt_attention_mask = prompt_attention_mask.to(device=device)

            # Optional: Paraphrased prompts for consistency loss (if adapted to Sana)
            prompt_embeds_pos = None
            prompt_attention_mask_pos = None
            if paraphrased_prompts is not None and consistency_lambda > 0:
                 prompt_embeds_pos, prompt_attention_mask_pos, _, _ = self.text_encoding_pipeline.encode_prompt(
                    paraphrased_prompts,
                    max_sequence_length=max_sequence_length,
                    complex_human_instruction=complex_human_instruction,
                    device=device
                )
                 prompt_embeds_pos = prompt_embeds_pos.to(dtype=weight_dtype)
                 prompt_attention_mask_pos = prompt_attention_mask_pos.to(device=device)

        # --- 2. Encode Images (VAE) ---
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(dtype=weight_dtype)).latent
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

        # --- 3. Prepare Flow Matching ---
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        
        # Timestep sampling
        weighting_scheme = getattr(self.args, "weighting_scheme", "logit_normal")
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=getattr(self.args, "logit_mean", 0.0),
            logit_std=getattr(self.args, "logit_std", 1.0),
            mode_scale=getattr(self.args, "mode_scale", 1.29),
        ).to(device)
        
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=device)
        
        # Get sigmas
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        # --- 4. Model Prediction ---
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            timestep=timesteps,
            return_dict=False,
        )[0]

        # --- 5. Calculate Loss ---
        # Weighting
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        
        target = noise - latents # flow matching target: v_t = x_1 - x_0
        
        # Basic loss
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        # --- 6. Optional: Consistency Loss ---
        if prompt_embeds_pos is not None and consistency_lambda > 0:
             model_pred_pos = self.transformer(
                hidden_states=noisy_latents, # Same noise
                encoder_hidden_states=prompt_embeds_pos, # Different text
                encoder_attention_mask=prompt_attention_mask_pos,
                timestep=timesteps,
                return_dict=False,
            )[0]
             
             loss_consistency = F.mse_loss(
                 model_pred.float(), 
                 model_pred_pos.float()
             )
             loss = loss + (consistency_lambda * loss_consistency)

        return loss

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 20, # Sana defaults to fewer steps usually
        guidance_scale: float = 4.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        height: int = 1024,
        width: int = 1024
    ) -> List[Any]:
        if device is None:
            device = next(self.transformer.parameters()).device
            
        was_training = self.transformer.training
        self.transformer.eval()
        
        pipeline = SanaPipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None
        
        if isinstance(prompt, str):
            prompt = [prompt] * num_images
            
        images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            complex_human_instruction=getattr(self.args, "complex_human_instruction", None),
        ).images
        
        if was_training:
            self.transformer.train()
            
        return images
