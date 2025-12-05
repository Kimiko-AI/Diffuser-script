import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ZImagePipeline
from diffusers.training_utils import compute_density_for_timestep_sampling

class ZImageWrapper(nn.Module):
    def __init__(self, transformer, vae, text_encoder, tokenizer, noise_scheduler, timestep_sampling_config=None):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        
        # Default sampling config if none provided
        self.timestep_sampling_config = timestep_sampling_config or {"weighting_scheme": "cosmap"}

        # Freeze frozen components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Ensure transformer is in train mode by default
        self.transformer.train()

        # Helper pipeline for encoding text during training
        # We pass transformer=None because we only use it for encode_prompt
        self.text_encoding_pipeline = ZImagePipeline(
            vae=self.vae, 
            text_encoder=self.text_encoder, 
            tokenizer=self.tokenizer, 
            transformer=None, 
            scheduler=None
        )

    def forward(self, pixel_values, prompts, device, weight_dtype=torch.float32):
        """
        Args:
            pixel_values: Tensor of shape (B, C, H, W)
            prompts: List of strings or pre-computed embeddings
            device: torch.device
            weight_dtype: torch.dtype (target dtype for transformer)
        Returns:
            loss: scalar tensor
        """
        # 1. Text Encode
        with torch.no_grad():
            prompt_embeds, _ = self.text_encoding_pipeline.encode_prompt(
                prompts, max_sequence_length=64, device=device
            )

        # 2. VAE Encode
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

        # 3. Noise & Flow Matching Logic
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        
        # Sample timesteps/density
        # Unpack config for clarity
        scheme = self.timestep_sampling_config.get("weighting_scheme", "cosmap")
        logit_mean = self.timestep_sampling_config.get("logit_mean", 0.0)
        logit_std = self.timestep_sampling_config.get("logit_std", 1.0)
        mode_scale = self.timestep_sampling_config.get("mode_scale", 1.29)
        
        u = compute_density_for_timestep_sampling(
            batch_size=bsz, 
            weighting_scheme=scheme,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale
        ).to(device)
        
        # Interpolate (Flow Matching)
        sigmas = u
        sigmas_view = sigmas.view(bsz, 1, 1, 1)
        noisy_latents = (1.0 - sigmas_view) * noise + sigmas_view * latents
        
        noisy_latents_list = list(noisy_latents.unsqueeze(2).unbind(dim=0))

        # 4. Predict
        model_pred = self.transformer(
            noisy_latents_list,
            u.flatten(),
            prompt_embeds,
            return_dict=False
        )[0]
        
        model_pred = torch.stack(model_pred, dim=0).squeeze(2)

        # 5. Loss
        target = latents - noise
        loss = F.mse_loss(model_pred.float(), target.float())
        
        return loss

    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=50, guidance_scale=4.0, num_images=1, seed=None, device=None):
        """
        Run inference (generation) using the current models.
        """
        if device is None:
            # Try to infer device from transformer parameters
            try:
                device = next(self.transformer.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        # Temporarily switch transformer to eval mode
        was_training = self.transformer.training
        self.transformer.eval()

        # Create a fresh pipeline to ensure it uses the current state of the transformer
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
        images = []
        
        # Generate images
        for _ in range(num_images):
            img = pipeline(
                prompt=prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            images.append(img)

        # Restore original training state
        if was_training:
            self.transformer.train()
            
        # Clean up to free memory if needed
        del pipeline
        
        return images