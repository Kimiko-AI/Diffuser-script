import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Optional, Union, Dict, Any
from diffusers import ZImagePipeline
from diffusers.training_utils import compute_density_for_timestep_sampling


class ZImageWrapper(nn.Module):
    def __init__(self, transformer, vae, text_encoder, tokenizer, noise_scheduler, timestep_sampling_config=None,
                 caption_dropout_prob=0.0, afm_lambda=0.0, consistency_lambda=1.0):
        super().__init__()
        self.transformer = transformer
        # Use lists to prevent nn.Module from registering them as submodules
        # This prevents them from being saved in the state_dict
        self._vae = [vae]
        self._text_encoder = [text_encoder]
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.caption_dropout_prob = caption_dropout_prob
        self.afm_lambda = afm_lambda
        self.consistency_lambda = consistency_lambda

        # Default sampling config if none provided
        self.timestep_sampling_config = timestep_sampling_config or {"weighting_scheme": "cosmap"}

        # Freeze frozen components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Ensure transformer is in train mode by default
        self.transformer.train()

        # Initialize Refiner
        # Latent channels usually 4 or 16. transformer output matches it.
        latent_channels = self.vae.config.latent_channels

        # Helper pipeline for encoding text during training
        # We pass transformer=None because we only use it for encode_prompt
        self.text_encoding_pipeline = ZImagePipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=None,
            scheduler=None
        )

    @property
    def vae(self):
        return self._vae[0]

    @property
    def text_encoder(self):
        return self._text_encoder[0]

    def load_state_dict(self, state_dict, strict=True):
        # Filter out vae and text_encoder keys from the state_dict
        # This ensures that resuming from checkpoints that included them (old behavior)
        # doesn't cause "Unexpected key(s)" errors.
        new_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("vae.") and not k.startswith("text_encoder.") and
               not k.startswith("_vae.") and not k.startswith("_text_encoder.")
        }
        return super().load_state_dict(new_state_dict, strict=strict)

    def forward(
            self,
            pixel_values,
            prompts,
            device,
            paraphrased_prompts: Optional[List[str]] = None,
            weight_dtype=torch.float32,
            consistency_lambda: float = None,  # Allow override
            **kwargs
    ):
        # Use instance default if not provided
        if consistency_lambda is None:
            consistency_lambda = self.consistency_lambda

        # --- 1. Encode Text (Original) ---
        with torch.no_grad():
            if self.training and self.caption_dropout_prob > 0:
                # Apply dropout to main prompts
                prompts_in = ["" if random.random() < self.caption_dropout_prob else p for p in prompts]
            else:
                prompts_in = prompts

            prompt_embeds, _ = self.text_encoding_pipeline.encode_prompt(
                prompts_in, max_sequence_length=64, device=device, do_classifier_free_guidance=False
            )

            # --- Encode Paraphrased Text (Positive Pair) ---
            prompt_embeds_pos = None
            if paraphrased_prompts is not None:
                # Usually we don't apply dropout to the positive pair consistency check
                prompt_embeds_pos, _ = self.text_encoding_pipeline.encode_prompt(
                    paraphrased_prompts, max_sequence_length=64, device=device, do_classifier_free_guidance=False
                )

        # --- 2. Encode Images (VAE) ---
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

        # --- 3. Prepare Flow Matching ---
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)

        u = compute_density_for_timestep_sampling(
            batch_size=bsz,
            **self.timestep_sampling_config
        ).to(device)

        sigmas_view = u.view(bsz, 1, 1, 1)
        noisy_latents = (1.0 - sigmas_view) * noise + sigmas_view * latents
        noisy_latents_input = list(noisy_latents.unsqueeze(2).unbind(dim=0))

        # --- 4. Model Prediction (Anchor) ---
        model_pred = self.transformer(
            noisy_latents_input,
            u.flatten(),
            prompt_embeds,
            return_dict=False
        )[0]
        model_pred = torch.stack(model_pred, dim=0).squeeze(2)

        # --- 5. Calculate Standard Losses ---
        target = latents - noise
        loss_fm = F.mse_loss(model_pred.float(), target.float())

        loss = loss_fm

        # --- 6. Positive Pair Consistency (The requested feature) ---
        # If we have paraphrases, force the model to predict similar flows for both prompts
        if prompt_embeds_pos is not None:
            model_pred_pos = self.transformer(
                noisy_latents_input,  # Same noisy input
                u.flatten(),  # Same timestep
                prompt_embeds_pos,  # Different (paraphrased) text
                return_dict=False
            )[0]
            model_pred_pos = torch.stack(model_pred_pos, dim=0).squeeze(2)

            # Minimize distance between Anchor Flow and Positive Flow
            loss_consistency = F.mse_loss(
                model_pred.detach().float(),
                model_pred_pos.float()
            )
            loss = loss + (consistency_lambda * loss_consistency)

        # --- 7. Negative Pair Contrastive (AFM from Paper) ---
        # The paper emphasizes this part: steering AWAY from wrong conditions
        if self.afm_lambda > 0 and bsz > 1:
            neg_latents = torch.roll(latents, shifts=1, dims=0)
            neg_noise = torch.roll(noise, shifts=1, dims=0)
            neg_target = (neg_latents - neg_noise)

            # Note: The paper maximizes distance to NEGATIVE flow [cite: 48, 128]
            # This is mathematically equivalent to minimizing the negative of the distance
            loss_contrastive = F.mse_loss(model_pred.detach().float(), neg_target.float())
            loss = loss - (self.afm_lambda * loss_contrastive)

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
        prompt = random.sample(prompt, 4)
        pipeline = ZImagePipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=False)

        generator = torch.Generator(device=device).manual_seed(seed) if seed else None

        # Handle single string prompt
        if isinstance(prompt, str):
            prompt = [prompt] * num_images

        images = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=20,
            guidance_scale=4
        ).images

        # Restore training state
        if was_training:
            self.transformer.train()

        return images, prompt
