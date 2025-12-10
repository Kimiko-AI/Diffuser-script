import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Optional, Union, Dict, Any
from diffusers import ZImagePipeline
from diffusers.training_utils import compute_density_for_timestep_sampling


class ZImageWrapper(nn.Module):
    def __init__(
            self,
            transformer: nn.Module,
            vae: nn.Module,
            text_encoder: nn.Module,
            tokenizer: Any,
            noise_scheduler: Any,
            timestep_sampling_config: Optional[Dict[str, Any]] = None,
            caption_dropout_prob: float = 0.0,
            afm_lambda: float = 0.05,  # Default lambda from the AFM paper [cite: 167, 249]
    ):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.caption_dropout_prob = caption_dropout_prob
        self.afm_lambda = afm_lambda

        # Configuration defaults
        self.sampling_config = timestep_sampling_config or {
            "weighting_scheme": "cosmap",
            "logit_mean": 0.0,
            "logit_std": 1.0,
            "mode_scale": 1.29,
        }

        # Freeze static components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Enable training for transformer
        self.transformer.train().to(torch.bfloat16)

        # Helper pipeline for text encoding
        self.text_encoding_pipeline = ZImagePipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=None,
            scheduler=None
        )

    def forward(
            self,
            pixel_values: torch.Tensor,
            prompts: List[str],
            device: torch.device,
            weight_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:

        # --- 1. Encode Text ---
        with torch.no_grad():
            if self.training and self.caption_dropout_prob > 0:
                prompts = ["" if random.random() < self.caption_dropout_prob else p for p in prompts]

            prompt_embeds, _ = self.text_encoding_pipeline.encode_prompt(
                prompts, max_sequence_length=64, device=device, do_classifier_free_guidance=False
            )

        # --- 2. Encode Images (VAE) ---
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

        # --- 3. Prepare Flow Matching ---
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)

        # Sample timesteps using the config
        u = compute_density_for_timestep_sampling(
            batch_size=bsz,
            **self.sampling_config
        ).to(device)

        # Interpolate: x_t = (1 - u) * noise + u * x_1
        # This matches the paper's optimal transport path where u goes 0 -> 1
        sigmas_view = u.view(bsz, 1, 1, 1)
        noisy_latents = (1.0 - sigmas_view) * noise + sigmas_view * latents

        # --- 4. Model Prediction ---
        # Reshape for transformer input as needed
        noisy_latents_input = list(noisy_latents.unsqueeze(2).unbind(dim=0))

        model_pred = self.transformer(
            noisy_latents_input,
            u.flatten(),
            prompt_embeds,
            return_dict=False
        )[0]
        model_pred = torch.stack(model_pred, dim=0).squeeze(2)

        # --- 5. Calculate Loss (AFM) ---
        # Target velocity v = x_1 - x_0 (latents - noise)
        target = latents - noise

        # Standard Flow Matching Loss (Minimize distance to correct flow)
        loss_fm = F.mse_loss(model_pred.float(), target.float())

        # Contrastive Regularization (Maximize distance from incorrect flows)
        if self.afm_lambda > 0 and bsz > 1:
            # We must sample neg_latents != latents.
            # rolling the batch by 1 guarantees every sample is paired with a different sample.
            neg_latents = torch.roll(latents, shifts=1, dims=0)
            neg_noise = torch.roll(noise, shifts=1, dims=0)

            # The flow vector for the negative pair
            neg_target = neg_latents - neg_noise

            # Calculate MSE for negative pairs
            loss_contrastive = F.mse_loss(model_pred.float(), neg_target.float())

            # Objective: Minimize FM loss, Maximize Contrastive Loss
            # L_total = L_fm - lambda * L_contrastive
            loss = loss_fm - (self.afm_lambda * loss_contrastive)
        else:
            loss = loss_fm

        return loss

    @torch.no_grad()
    def generate(
            self,
            prompt: Union[str, List[str]],
            num_inference_steps: int = 50,
            guidance_scale: float = 4.0,
            num_images: int = 1,
            seed: Optional[int] = None,
            device: Optional[torch.device] = None
    ) -> List[Any]:

        if device is None:
            device = next(self.transformer.parameters()).device

        # Ensure eval mode
        was_training = self.transformer.training
        self.transformer.eval()

        pipeline = ZImagePipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        generator = torch.Generator(device=device).manual_seed(seed) if seed else None

        # Handle single string prompt
        if isinstance(prompt, str):
            prompt = [prompt] * num_images

        images = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images

        # Restore training state
        if was_training:
            self.transformer.train()

        return images