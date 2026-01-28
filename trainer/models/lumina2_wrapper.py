import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import timm
from typing import List, Optional, Union, Dict, Any, Tuple
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.image_processor import VaeImageProcessor
from tqdm.auto import tqdm
import copy


# -----------------------------------------------------------------------------
# Loss Functions: Charbonnier, SSIM, DINOv3 Perceptual
# -----------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-6, reduction='mean'):
        """
        Args:
            alpha (float): Balancing weight (0.5 = equal weight).
            eps (float): Stability term.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target, weights=None, **kwargs):
        """
        Args:
            pred (Tensor): (B, C, H, W) or (B, D)
            target (Tensor): (B, C, H, W) or (B, D)
            weights (Tensor, optional): (B,) or (B, 1, 1, 1) - Timestep/SNR weights.
            **kwargs: Catches other unexpected args (like 'split' or 'mask') safely.
        """
        # --- 1. Magnitude Loss (Charbonnier) ---
        # Shape: (B, C, H, W)
        diff = pred - target
        loss_mag_pixel = torch.sqrt(diff * diff + self.eps ** 2)

        # Reduce Magnitude loss to (B,) so we can combine it with Directional loss
        # We take the mean over all dimensions except Batch
        loss_mag_sample = loss_mag_pixel.view(loss_mag_pixel.shape[0], -1).mean(dim=1)

        # --- 2. Directional Loss (Cosine Similarity) ---
        # Shape: (B,)
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8)
        loss_dir_sample = 1.0 - cosine_sim

        # --- 3. Weighted Combination (Per Sample) ---
        # Shape: (B,)
        loss_combined = ((1 - self.alpha) * loss_mag_sample) + (self.alpha * loss_dir_sample)

        # --- 4. Apply Training Weights (Timestep/SNR) ---
        if weights is not None:
            # Ensure weights are (B,) to match loss_combined
            if weights.ndim > 1:
                weights = weights.reshape(weights.shape[0])
            loss_combined = loss_combined * weights

        # --- 5. Final Reduction ---
        if self.reduction == 'mean':
            return loss_combined.mean()
        elif self.reduction == 'sum':
            return loss_combined.sum()
        else:
            return loss_combined


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM Loss for image reconstruction training.

    Args:
        window_size (int): Size of the gaussian window. Default: 11
        size_average (bool): If True, returns average SSIM. If False, returns SSIM per sample. Default: True

    Forward expects inputs in range [0, 1]
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = None

    def _gaussian(self, window_size, sigma):
        """Create 1D Gaussian kernel."""
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        """Create 2D Gaussian window for all channels."""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average):
        """Calculate SSIM between two images."""
        # Calculate means
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        """
        Calculate SSIM loss between two images.

        Args:
            img1 (torch.Tensor): First image tensor of shape (B, C, H, W) in range [0, 1]
            img2 (torch.Tensor): Second image tensor of shape (B, C, H, W) in range [0, 1]

        Returns:
            torch.Tensor: SSIM loss (1 - SSIM similarity)
        """
        (_, channel, _, _) = img1.size()

        # Create or recreate window if necessary
        if (self.window is None or
                self.window.size(0) != channel or
                self.window.data.type() != img1.data.type()):
            self.window = self._create_window(self.window_size, channel)

        # Move window to correct device if needed
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        # Return loss (1 - SSIM similarity)
        return 1.0 - self._ssim(img1, img2, self.window, self.window_size,
                                channel, self.size_average)


class ConvNeXtPerceptualLoss(nn.Module):
    """
    LPIPS-style perceptual loss using ConvNeXt-DINOv3 features.
    """

    def __init__(
            self,
            model_name="convnext_large.dinov3_lvd1689m",
            layer_weights=None, devices="cpu"
    ):
        super().__init__()

        # Create feature extractor
        # We assume the device handling is done by the training wrapper moving this module
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
        ).eval().to(devices)

        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.layer_weights = layer_weights

        # Resolve transforms (normalization constants mainly)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.mean = torch.tensor(data_config['mean']).view(1, 3, 1, 1)
        self.std = torch.tensor(data_config['std']).view(1, 3, 1, 1)

    def forward(self, img1, img2):
        """
        img1, img2: Tensor (B, 3, H, W) in range [0, 1]
        """
        # Manual Normalization for TIMM model using registered buffers if possible,
        # or simple tensor math. Note: self.mean/std should be moved to device in the forward pass
        # or registered as buffers in init if strictly part of state_dict.
        # Here we do it on the fly to handle device movement automatically.

        mean = self.mean.to(img1.device)
        std = self.std.to(img1.device)

        in1 = (img1 - mean) / std
        in2 = (img2 - mean) / std

        feats1 = self.model(in1)
        feats2 = self.model(in2)

        if self.layer_weights is None:
            weights = [1.0] * len(feats1)
        else:
            weights = self.layer_weights

        loss = 0.0
        weight_sum = 0.0

        for w, f1, f2 in zip(weights, feats1, feats2):
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            dist = (f1 - f2).pow(2).sum(dim=1)
            loss = loss + w * dist.mean()
            weight_sum += w

        return loss / weight_sum


# -----------------------------------------------------------------------------
# Flux 2 Logic
# -----------------------------------------------------------------------------

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


class Flux2TrainingWrapper(nn.Module):
    def __init__(self, transformer, vae, text_encoder, tokenizer, noise_scheduler,
                 timestep_sampling_config=None, caption_dropout_prob=0.0,
                 afm_lambda=0.0, consistency_lambda=1.0,
                 ssim_lambda=0, perceptual_lambda=1,  # New lambda
                 perceptual_model_name="convnext_large.dinov3_lvd1689m", device="",
                 **kwargs):
        super().__init__()
        self.transformer = transformer
        self._vae = [vae]
        self._text_encoder = [text_encoder]
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.device = next(transformer.parameters()).device
        # Training Hyperparameters
        self.caption_dropout_prob = caption_dropout_prob
        self.afm_lambda = afm_lambda
        self.consistency_lambda = consistency_lambda
        self.ssim_lambda = ssim_lambda
        self.perceptual_lambda = perceptual_lambda
        self.timestep_sampling_config = timestep_sampling_config or {"weighting_scheme": "uniform"}
        self.target_transformer = copy.deepcopy(self.transformer)
        self.target_transformer.requires_grad_(False)
        self.target_transformer.eval()
        self.ema_decay = 0.99
        # Initialize Losses
        self.charbonnier = CharbonnierLoss().to(self.device)
        self.ssim_loss = SSIMLoss().to(self.device)

        # Initialize Perceptual Loss
        print(f"Initializing Perceptual Loss with {perceptual_model_name}...")
        try:
            self.perceptual_loss = ConvNeXtPerceptualLoss(model_name=perceptual_model_name, devices=self.device)
        except Exception as e:
            print(f"Warning: Failed to load DINOv3 model: {e}. Perceptual loss will be disabled.")
            self.perceptual_loss = None
            self.perceptual_lambda = 0.0

        # Freeze components
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.train()

        # Ensure perceptual model is frozen
        if self.perceptual_loss:
            self.perceptual_loss.eval()
            self.perceptual_loss.requires_grad_(False)

        # Flux 2 Logic Constants
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.text_encoder_out_layers = (10, 20, 30)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

    @property
    def vae(self):
        return self._vae[0]

    def step_ema(self, decay=None):
        """
        Updates the target model parameters using Exponential Moving Average.
        Call this after every optimizer.step() during training.
        """
        if decay is None:
            decay = self.ema_decay

        with torch.no_grad():
            for param_t, param_s in zip(self.target_transformer.parameters(),
                                        self.transformer.parameters()):
                param_t.data.mul_(decay).add_(param_s.data, alpha=1 - decay)

    @property
    def text_encoder(self):
        return self._text_encoder[0]

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("vae.") and not k.startswith("text_encoder.") and
               not k.startswith("_vae.") and not k.startswith("_text_encoder.") and
               not k.startswith("perceptual_loss.")  # Exclude perceptual weights from loading
        }
        return super().load_state_dict(new_state_dict, strict=strict)

    # ... [Static Helpers omitted for brevity, identical to previous] ...
    @staticmethod
    def _prepare_text_ids(x: torch.Tensor):
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1);
            h = torch.arange(1);
            w = torch.arange(1);
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor):
        batch_size, _, height, width = latents.shape
        t = torch.arange(1);
        h = torch.arange(height);
        w = torch.arange(width);
        l = torch.arange(1)
        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    @staticmethod
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // 4, height * 2, width * 2)
        return latents

    @staticmethod
    def _pack_latents(latents):
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width):
        batch_size, seq_len, num_channels = latents.shape
        latents = latents.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
        return latents

    def _get_qwen3_prompt_embeds(self, prompt: List[str], device, max_length=128):
        # ... [Same as previous] ...
        all_input_ids = []
        all_attention_masks = []
        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        with torch.no_grad():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            out = torch.stack([output.hidden_states[k] for k in self.text_encoder_out_layers], dim=1)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds

    def forward(
            self,
            pixel_values,
            prompts,
            device,
            paraphrased_prompts: Optional[List[str]] = None,
            weight_dtype=torch.float32,
            consistency_lambda: float = None,
            global_step=0,
            **kwargs
    ):
        if consistency_lambda is None:
            consistency_lambda = self.consistency_lambda

        # 1. Text Encoding
        if self.training and self.caption_dropout_prob > 0:
            prompts_in = ["" if random.random() < self.caption_dropout_prob else p for p in prompts]
        else:
            prompts_in = prompts

        prompt_embeds = self._get_qwen3_prompt_embeds(prompts_in, device)
        txt_ids = self._prepare_text_ids(prompt_embeds).to(device)

        # 2. Image Encoding & Preprocessing
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
            latents = (latents - 0.0) * 0.62
            latents_clean = self._patchify_latents(latents)

        img_ids = self._prepare_latent_ids(latents_clean).to(device)

        # 3. Flow Matching Setup
        bsz = latents_clean.shape[0]
        noise = torch.randn_like(latents_clean)
        u = compute_density_for_timestep_sampling(batch_size=bsz, **self.timestep_sampling_config).to(device)

        sigmas_view = u.view(bsz, 1, 1, 1)
        noisy_latents = (1.0 - sigmas_view) * latents_clean + sigmas_view * noise
        target = noise - latents_clean

        # 4. Packing
        packed_noisy_latents = self._pack_latents(noisy_latents)
        packed_target = self._pack_latents(target)

        # --- NEW: Calculate Drop Ratio based on global_step ---
        # Seed a generator with global_step for reproducibility
        # This ensures the decision is deterministic for a given step.
        rng_gen = torch.Generator(device='cpu').manual_seed(global_step)

        # 90% chance to drop 75% tokens (0.75), 10% chance to drop nothing (0.0)
        if torch.rand(1, generator=rng_gen).item() < 0.5:
            current_drop_ratio = 0.75
        else:
            current_drop_ratio = 0.0
        # ------------------------------------------------------

        # 5. Transformer Call
        model_pred = self.transformer(
            hidden_states=packed_noisy_latents,
            timestep=u.flatten(),
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
            token_drop_ratio=current_drop_ratio,  # <--- Pass the ratio here
        )[0]

        # 6. Loss Calculation

        # --- A. Latent Space Loss (Charbonnier) ---
        loss = self.charbonnier(model_pred.float(), packed_target.float())

        # --- D. AFM (Contrastive) ---
        if self.afm_lambda > 0 and bsz > 1:
            neg_latents = torch.roll(latents_clean, shifts=1, dims=0)
            neg_noise = torch.roll(noise, shifts=1, dims=0)
            neg_target = self._pack_latents(neg_latents - neg_noise)

            loss_contrastive = self.charbonnier(model_pred.detach().float(), neg_target.float())
            loss = loss - (self.afm_lambda * loss_contrastive)

        return loss, loss

    # ... [Generate method remains the same] ...
    @torch.no_grad()
    def generate(
            self,
            prompt: Union[str, List[str]],
            num_inference_steps: int = 20,
            guidance_scale: float = 3.5,
            height: int = 512,
            width: int = 512,
            num_images: int = 1,
            seed: Optional[int] = None,
            device: Optional[torch.device] = None,
            drop=0
    ) -> List[Any]:

        if device is None:
            device = next(self.transformer.parameters()).device

        was_training = self.transformer.training
        self.transformer.eval()

        if isinstance(prompt, str):
            prompt = [prompt] * num_images

        batch_size = len(prompt)
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None

        # 1. Prepare Positive Prompts
        prompt_embeds = self._get_qwen3_prompt_embeds(prompt, device)
        txt_ids = self._prepare_text_ids(prompt_embeds).to(device)

        # 2. Prepare Negative Prompts (Custom Logic)
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg_prompt = []
            for p in prompt:
                # Split using space
                words = p.split(' ')
                # Drop everything beside first 4 words
                truncated_p = ""
                neg_prompt.append(truncated_p)

            # Debug print to verify behavior (optional, can be removed)
            # print(f"CFG Negative Prompts: {neg_prompt}")

            neg_prompt_embeds = self._get_qwen3_prompt_embeds(neg_prompt, device)
            neg_txt_ids = self._prepare_text_ids(neg_prompt_embeds).to(device)

        # 3. Prepare Latents
        num_channels_latents = self.transformer.config.in_channels // 4
        h_latent = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_latent = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents * 4, h_latent // 2, w_latent // 2)

        latents = torch.randn(shape, generator=generator, device=device, dtype=self.transformer.dtype)
        img_ids = self._prepare_latent_ids(latents).to(device)
        latents = self._pack_latents(latents)

        # 4. Scheduler Setup
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
        timesteps = self.noise_scheduler.timesteps

        # 5. Denoising Loop
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            vec_t = t.expand(latents.shape[0]).to(latents.dtype)

            # Positive prediction
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=vec_t / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                token_drop_ratio=0,
                return_dict=False,
            )[0]

            if do_cfg:
                # Negative prediction using the truncated prompts
                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=vec_t / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=neg_txt_ids,
                    img_ids=img_ids,
                    token_drop_ratio=0.75,
                    return_dict=False,
                )[0]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 6. Post-processing
        latents = self._unpack_latents(latents, h_latent // 2, w_latent // 2)
        latents = latents / 0.62
        latents = self._unpatchify_latents(latents)

        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")

        if was_training:
            self.transformer.train()

        return image, prompt