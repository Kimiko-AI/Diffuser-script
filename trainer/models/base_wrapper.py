import torch
import torch.nn as nn
from typing import List, Optional, Union, Any

class BaseWrapper(nn.Module):
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
        if self.vae:
             self.vae.requires_grad_(False)
        if self.text_encoder:
             self.text_encoder.requires_grad_(False)

        # Ensure transformer is in train mode
        if self.transformer:
            self.transformer.train()

    @property
    def vae(self):
        return self._vae[0]

    @property
    def text_encoder(self):
        return self._text_encoder[0]

    def encode_text(self, prompts, device, weight_dtype):
        """
        Common text encoding logic.
        """
        import random
        with torch.no_grad():
            if self.training and self.caption_dropout_prob > 0:
                 prompts = ["" if random.random() < self.caption_dropout_prob else p for p in prompts]
            
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)

            # Request hidden states to support CausalLM models (where [0] is logits)
            # and standard models (where [0] or last_hidden_state is embeddings)
            outputs = self.text_encoder(
                text_input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            
            if hasattr(outputs, "hidden_states"):
                prompt_embeds = outputs.hidden_states[-1]
            elif hasattr(outputs, "last_hidden_state"):
                prompt_embeds = outputs.last_hidden_state
            else:
                prompt_embeds = outputs[0]

            prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
            
            return prompt_embeds

    def encode_images(self, pixel_values, weight_dtype):
        """
        Common image encoding logic (VAE).
        """
        with torch.no_grad():
            # Check if VAE is AutoencoderKL or DC, methods might differ slightly but usually .encode().latent_dist.sample() for KL
            # Sana uses .latent
            # We assume AutoencoderKL behavior as default or handle specific in subclasses if needed.
            # But let's try to be generic if possible or leave abstract.
            # ZImage/SRDiT: encode().latent_dist.sample() * scaling_factor
            # Sana: encode().latent * scaling_factor
            
            # Subclasses should probably implement this if VAE types differ significantly.
            # Or we check attribute existence.
            # We assume AutoencoderKL behavior as default.
            posterior = self.vae.encode(pixel_values.to(dtype=weight_dtype))
            latents = posterior.latent_dist.sample()
                 
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)
            return latents

    @torch.no_grad()
    def generate(self, prompt, **kwargs):
        """
        Generic generate method. Subclasses should override or configure the pipeline used.
        """
        raise NotImplementedError("Subclasses must implement generate()")
