import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
from typing import List, Optional, Union, Any, Tuple
from diffusers import Lumina2Pipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

logger = logging.getLogger(__name__)

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
        
        # Unfreeze transformer for full fine-tuning
        self.transformer.requires_grad_(True)
        self.transformer.train()
        if getattr(args, "gradient_checkpointing", False):
            self.transformer.enable_gradient_checkpointing()

        # Helper pipeline for encoding text
        # We initialize it with None for components we don't need immediately to save memory/time
        self.text_encoding_pipeline = Lumina2Pipeline(
            transformer=self.transformer, 
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
            vae=self.vae
        )
        
        # Set system prompt default
        self.system_prompt = getattr(args, "system_prompt", "You are an assistant designed to generate high-quality images.")

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

    # --- Custom Text Encoding Logic ---

    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.text_encoder.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because Gemma can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        encoder_outputs = self.text_encoder(
            text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        
        # Adaptation: Layer-wise pooling
        # Drop embedding layer
        hidden_states = encoder_outputs.hidden_states[1:]

        # Stack: (num_layers, batch, seq_len, hidden_size)
        H = torch.stack(hidden_states, dim=0)
        # L2 normalize across hidden_size dim
        H_norm = torch.nn.functional.normalize(H, p=2, dim=-1)
        # Result: (batch, seq_len, hidden_size)
        prompt_embeds = H_norm.mean(dim=0)

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.text_encoder.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if system_prompt is None:
            system_prompt = self.system_prompt
        
        # Apply system prompt formatting if prompts are raw
        if prompt is not None:
            prompt = [system_prompt + " <Prompt Start> " + p for p in prompt]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
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

            # Normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
            )

            batch_size, seq_len, _ = negative_prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                batch_size * num_images_per_prompt, -1
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask


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

            # Use local encode_prompt
            prompt_embeds, prompt_attention_mask, _, _ = self.encode_prompt(
                prompt=prompts_in,
                do_classifier_free_guidance=False,
                device=device,
                max_sequence_length=max_sequence_length,
                system_prompt=system_prompt
            )

            prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
            prompt_attention_mask = prompt_attention_mask.to(device=device)

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
        # Use args directly if flat or nested dict if structured
        # config.yaml structure: timestep_sampling: {weighting_scheme: ...}
        # But args flattening in train.py keeps subdicts? No, I flattened it recursively.
        # But for 'timestep_sampling' I said I'd flatten. 
        # Let's check train.py again. 
        # "timestep_sampling" key in args? 
        # I added exception in flatten(): k != "timestep_sampling"
        # So args.timestep_sampling should be a dict.
        
        ts_config = getattr(self.args, "timestep_sampling", {})
        weighting_scheme = ts_config.get("weighting_scheme", "uniform")
        
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=ts_config.get("logit_mean", 0.0),
            logit_std=ts_config.get("logit_std", 1.0),
            mode_scale=ts_config.get("mode_scale", 1.29),
        ).to(device)

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        # Ensure indices and timesteps are on the same device
        # We clamp indices just in case, though u is usually [0, 1)
        indices = indices.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
        
        # Check if timesteps is present (some schedulers need set_timesteps called)
        if hasattr(self.noise_scheduler, "timesteps") and self.noise_scheduler.timesteps is not None:
             sched_timesteps = self.noise_scheduler.timesteps.to(device)
             timesteps = sched_timesteps[indices]
        else:
             # Fallback if timesteps is not populated (though error implies it was a CPU tensor)
             # Usually FlowMatch scheduler has reversed 1000 -> 0 or 0 -> 1000
             # Default to linspace if missing
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
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
        
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
            guidance_scale: float = 4.0, 
            num_images: int = 1,
            seed: Optional[int] = None,
            device: Optional[torch.device] = None,
            height: int = 512, 
            width: int = 512,
            **kwargs
    ) -> List[Any]:
        if device is None:
            device = next(self.transformer.parameters()).device

        # Patch pipeline with our custom encoder methods
        # To avoid patching the class globally or creating a new subclass dynamically, 
        # we can just assign the methods to the instance.
        pipeline = Lumina2Pipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler,
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        
        # Bind methods to the pipeline instance
        # We simply assign the bound methods from the wrapper to the pipeline.
        # This ensures 'self' inside these methods remains the wrapper instance,
        # which holds the correct tokenizer, text_encoder, and configuration.
        pipeline._get_gemma_prompt_embeds = self._get_gemma_prompt_embeds
        pipeline.encode_prompt = self.encode_prompt
        
        # We also ensure the pipeline has access to system_prompt if it needs it 
        # (though our wrapper.encode_prompt uses self.system_prompt from wrapper)
        pipeline.system_prompt = self.system_prompt

        generator = torch.Generator(device=device).manual_seed(seed) if seed else None
        
        images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            system_prompt=self.system_prompt
        ).images

        return images, prompt