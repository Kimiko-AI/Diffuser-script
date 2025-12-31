import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
from typing import List, Optional, Union, Any, Tuple
from diffusers import Lumina2Pipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

logger = logging.getLogger(__name__)

def _encode_prompt_lumina2_monkeypatch(
    self,
    prompt: Union[str, List[str]],
    *args,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # Extract args with defaults, prioritizing kwargs
    device = kwargs.get("device", None)
    # If device was passed positionally (very unlikely for device, but possible if signature mismatch)
    # we ignore positional args for device if it's in kwargs.
    # If it's NOT in kwargs, we default to self._execution_device
    
    device = device or self._execution_device

    do_classifier_free_guidance = kwargs.get("do_classifier_free_guidance", True)
    negative_prompt = kwargs.get("negative_prompt", None)
    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
    prompt_embeds = kwargs.get("prompt_embeds", None)
    negative_prompt_embeds = kwargs.get("negative_prompt_embeds", None)
    prompt_attention_mask = kwargs.get("prompt_attention_mask", None)
    negative_prompt_attention_mask = kwargs.get("negative_prompt_attention_mask", None)
    max_sequence_length = kwargs.get("max_sequence_length", 256)
    system_prompt = kwargs.get("system_prompt", None)

    # Default system prompt if not provided (Lumina2 specific)
    if system_prompt is None:
        system_prompt = getattr(self, "system_prompt", "You are an assistant designed to generate high-quality images.")

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Apply system prompt formatting if prompts are raw
    if prompt is not None:
        prompt = [system_prompt + " <Prompt Start> " + p for p in prompt]

    def _get_gemma_prompt_embeds(
        prompt_in: Union[str, List[str]],
        device_in: Optional[torch.device] = None,
        max_len_in: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device_in = device_in or self.text_encoder.device
        prompt_in = [prompt_in] if isinstance(prompt_in, str) else prompt_in
        
        text_inputs = self.tokenizer(
            prompt_in,
            padding="max_length",
            max_length=max_len_in,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device_in)
        prompt_mask = text_inputs.attention_mask.to(device_in)

        encoder_outputs = self.text_encoder(
            text_input_ids, attention_mask=prompt_mask, output_hidden_states=True, use_cache=False
        )
        
        # Drop embedding layer
        hidden_states = encoder_outputs.hidden_states[1:]

        # Stack: (num_layers, batch, seq_len, hidden_size)
        H = torch.stack(hidden_states, dim=0)
        # L2 normalize across hidden_size dim
        H_norm = torch.nn.functional.normalize(H, p=2, dim=-1)
        # Result: (batch, seq_len, hidden_size)
        emb = H_norm.mean(dim=0)

        if self.text_encoder is not None:
            dtype_enc = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype_enc = self.transformer.dtype
        else:
            dtype_enc = None

        emb = emb.to(dtype=dtype_enc, device=device_in)
        return emb, prompt_mask

    if prompt_embeds is None:
        prompt_embeds, prompt_attention_mask = _get_gemma_prompt_embeds(
            prompt_in=prompt,
            device_in=device,
            max_len_in=max_sequence_length,
        )

    batch_size, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings and attention mask for each generation per prompt
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
    prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

    # Get negative embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt if negative_prompt is not None else ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if prompt is not None and type(prompt) is not type(negative_prompt):
             raise TypeError(f"Type mismatch: {type(negative_prompt)} != {type(prompt)}.")

        negative_prompt_embeds, negative_prompt_attention_mask = _get_gemma_prompt_embeds(
            prompt_in=negative_prompt,
            device_in=device,
            max_len_in=max_sequence_length,
        )

        batch_size, seq_len, _ = negative_prompt_embeds.shape
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

# Monkeypatch Lumina2Pipeline
Lumina2Pipeline.encode_prompt = _encode_prompt_lumina2_monkeypatch


class Lumina2Wrapper(nn.Module):
    def __init__(self, transformer, vae, text_encoder, tokenizer, noise_scheduler, timestep_sampling_config=None,
                 caption_dropout_prob=0.0, afm_lambda=0.0, consistency_lambda=1.0, args=None):
        super().__init__()
        self.transformer = transformer
        self._vae = [vae]
        self._text_encoder = [text_encoder]
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.caption_dropout_prob = caption_dropout_prob
        self.afm_lambda = afm_lambda
        self.consistency_lambda = consistency_lambda
        self.args = args

        # Default sampling config if none provided
        self.timestep_sampling_config = timestep_sampling_config or {"weighting_scheme": "logit_normal"}
        
        # Ensure defaults for logit-normal parameters if scheme is logit_normal
        if self.timestep_sampling_config.get("weighting_scheme") == "logit_normal":
             self.timestep_sampling_config.setdefault("logit_mean", 0.0)
             self.timestep_sampling_config.setdefault("logit_std", 1.0)
             self.timestep_sampling_config.setdefault("mode_scale", 1.29)

        # Freeze frozen components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Ensure transformer is in train mode
        self.transformer.train()
        if args and getattr(args, "gradient_checkpointing", False):
            self.transformer.enable_gradient_checkpointing()

        # Helper pipeline for encoding text during training
        self.text_encoding_pipeline = Lumina2Pipeline(
            transformer=self.transformer, 
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
            vae=self.vae
        )
        
        # Set system prompt default on the pipeline instance
        self.text_encoding_pipeline.system_prompt = getattr(args, "system_prompt", "You are an assistant designed to generate high-quality images.") if args else "You are an assistant designed to generate high-quality images."

    @property
    def vae(self):
        return self._vae[0]

    @property
    def text_encoder(self):
        return self._text_encoder[0]

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("vae.") and not k.startswith("text_encoder.") and
               not k.startswith("_vae.") and not k.startswith("_text_encoder.")
        }
        return super().load_state_dict(new_state_dict, strict=strict)

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32, device=None):
        # Keep this helper as it's specific to Flow Matching sigma retrieval
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
            paraphrased_prompts: Optional[List[str]] = None,
            weight_dtype=torch.float32,
            consistency_lambda: float = None,
            **kwargs
    ):
        # Use instance default if not provided
        if consistency_lambda is None:
            consistency_lambda = self.consistency_lambda

        # --- 1. Encode Text ---
        with torch.no_grad():
            if self.training and self.caption_dropout_prob > 0:
                prompts_in = ["" if random.random() < self.caption_dropout_prob else p for p in prompts]
            else:
                prompts_in = prompts

            max_sequence_length = getattr(self.args, "max_sequence_length", 256) if self.args else 256
            
            # Using the monkeypatched pipeline method
            prompt_embeds, prompt_attention_mask, _, _ = self.text_encoding_pipeline.encode_prompt(
                prompt=prompts_in,
                do_classifier_free_guidance=False,
                device=device,
                max_sequence_length=max_sequence_length
            )

            prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
            prompt_attention_mask = prompt_attention_mask.to(device=device)

            # --- Encode Paraphrased Text (Positive Pair) ---
            prompt_embeds_pos = None
            prompt_attention_mask_pos = None
            if paraphrased_prompts is not None:
                prompt_embeds_pos, prompt_attention_mask_pos, _, _ = self.text_encoding_pipeline.encode_prompt(
                    prompt=paraphrased_prompts,
                    do_classifier_free_guidance=False,
                    device=device,
                    max_sequence_length=max_sequence_length
                )
                prompt_embeds_pos = prompt_embeds_pos.to(dtype=weight_dtype)
                prompt_attention_mask_pos = prompt_attention_mask_pos.to(device=device)

        # --- 2. Encode Images (VAE) ---
        with torch.no_grad():
            # AutoencoderKL
            posterior = self.vae.encode(pixel_values.to(dtype=self.vae.dtype)).latent_dist
            latents = posterior.sample()
            
            # Lumina2 scaling/shifting
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
        u = compute_density_for_timestep_sampling(
            batch_size=bsz,
            **self.timestep_sampling_config
        ).to(device)

        # Map u to timesteps
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        indices = indices.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
        
        # Check if timesteps is present
        if hasattr(self.noise_scheduler, "timesteps") and self.noise_scheduler.timesteps is not None:
             sched_timesteps = self.noise_scheduler.timesteps.to(device)
             timesteps = sched_timesteps[indices]
        else:
             # Fallback
             timesteps = torch.linspace(0, self.noise_scheduler.config.num_train_timesteps - 1, self.noise_scheduler.config.num_train_timesteps, device=device).flip(0)
             timesteps = timesteps[indices]

        # Add noise
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
        # Using weighting from config if possible, but ZImage uses hardcoded logic or fm. 
        # Lumina2 generally uses uniform or logit-normal with specific weighting.
        # We stick to compute_loss_weighting_for_sd3 as it was there.
        weighting_scheme = self.timestep_sampling_config.get("weighting_scheme", "logit_normal")
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        
        target = latents - noise

        loss_fm = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss_fm.mean()

        # --- 6. Positive Pair Consistency ---
        if prompt_embeds_pos is not None:
             model_pred_pos = self.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds_pos,
                encoder_attention_mask=prompt_attention_mask_pos,
                timestep=timesteps_norm,
                return_dict=False,
            )[0]
             
             loss_consistency = F.mse_loss(
                 model_pred.detach().float(),
                 model_pred_pos.float()
             )
             loss = loss + (consistency_lambda * loss_consistency)

        # --- 7. Negative Pair Contrastive (AFM) ---
        if self.afm_lambda > 0 and bsz > 1:
            neg_latents = torch.roll(latents, shifts=1, dims=0)
            neg_noise = torch.roll(noise, shifts=1, dims=0)
            neg_target = (neg_latents - neg_noise)

            # We use simple MSE for contrastive loss as in ZImage
            loss_contrastive = F.mse_loss(model_pred.detach().float(), neg_target.float())
            loss = loss - (self.afm_lambda * loss_contrastive)

        return loss

    @torch.no_grad()
    def generate(
            self,
            prompt: Union[str, List[str]],
            num_inference_steps: int = 20,
            guidance_scale: float = 4.0, 
            num_images: int = 1,
            seed: Optional[int] = None,
            device: Optional[torch.device] = None,
            height: int = 256,
            width: int = 256,
            **kwargs
    ) -> List[Any]:
        
        if device is None:
            device = next(self.transformer.parameters()).device
        was_training = self.transformer.training
        self.transformer.eval()
        # Reuse internal pipeline if possible, or create new one like ZImage
        pipeline = Lumina2Pipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=False)
        
        # System prompt is handled by the monkeypatched encode_prompt via instance attribute or default
        pipeline.system_prompt = getattr(self.args, "system_prompt", "You are an assistant designed to generate high-quality images.") if self.args else "You are an assistant designed to generate high-quality images."

        device_type = device.type if device else "cuda"
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None
        
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            images = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
            ).images
        if was_training:
            self.transformer.train()
        return images, prompt