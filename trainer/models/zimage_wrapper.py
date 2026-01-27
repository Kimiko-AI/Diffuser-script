import torch
import torch.nn as nn
import copy
import random
import numpy as np
import ot
from collections import deque
import torch.nn.functional as F
from diffusers import ZImagePipeline
from diffusers.training_utils import compute_density_for_timestep_sampling


def compute_dual_infonce_loss(text_embeds, image_embeds, temperature=0.07):
    """
    Computes InfoNCE both ways with specific gradient stops.
    
    Args:
        text_embeds:  (B, D) Tensor
        image_embeds: (B, D) Tensor
    """
    # 1. Normalize
    text_embeds = F.normalize(text_embeds, dim=-1)
    image_embeds = F.normalize(image_embeds, dim=-1)
    
    batch_size = text_embeds.shape[0]
    labels = torch.arange(batch_size, device=text_embeds.device)
    
    # --- Direction 1: Text -> Image (Stop grad on Image) ---
    # Image acts as the target/teacher
    logits_t2i = torch.matmul(text_embeds, image_embeds.t()) / temperature
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    
    # --- Direction 2: Image -> Text (Stop grad on Text) ---
    # Text acts as the target/teacher
    logits_i2t = torch.matmul(image_embeds, text_embeds.t()) / temperature
    loss_i2t = F.cross_entropy(logits_i2t, labels)
    
    # Total Symmetric Loss
    return (loss_t2i + loss_i2t) / 2

    
class AlignedCharbonnierLoss(nn.Module):
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
        loss_mag_pixel = torch.sqrt(diff * diff + self.eps**2)
        
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


class ZImageWrapper(nn.Module):
    """
    ZImageWrapper with SigLip (stable CLIP-style) contrastive loss and EMA-enabled queue memory.
    """

    def __init__(
        self, 
        transformer, 
        vae, 
        text_encoder, 
        tokenizer, 
        noise_scheduler, 
        timestep_sampling_config=None, 
        caption_dropout_prob=0.0, 
        uot_tau=2.0, 
        uot_k=10.0, 
        contrastive_temperature=0.07, 
        contrastive_queue_batches=8, 
        ema_decay=0.999,  contrastive_warmup_steps=10,
        **kwargs
    ):
        super().__init__()
        self.transformer = transformer
        self._vae = [vae]
        self._text_encoder = [text_encoder]
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        
        # --- UOT-RFM Config ---
        self.uot_tau = uot_tau
        self.uot_k = uot_k
        
        # --- EMA / Target Model ---
        self.target_transformer = copy.deepcopy(self.transformer)
        self.target_transformer.requires_grad_(False)
        self.target_transformer.eval()
        self.ema_decay = ema_decay

        self.caption_dropout_prob = caption_dropout_prob
        self.timestep_sampling_config = timestep_sampling_config or {"weighting_scheme": "cosmap"}
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.charbonnier = AlignedCharbonnierLoss()
        self.transformer.train()

        # Helper pipeline for encoding
        self.text_encoding_pipeline = ZImagePipeline(
            vae=self.vae, text_encoder=self.text_encoder, 
            tokenizer=self.tokenizer, transformer=None, scheduler=None
        )
        self.snr_gamma = 5

        # ---- Contrastive / SigLip settings ----
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_queue_batches = contrastive_queue_batches
        self.text_queue = deque(maxlen=self.contrastive_queue_batches)
        self.image_queue = deque(maxlen=self.contrastive_queue_batches)
        self.global_step = 0
        self.contrastive_warmup_steps = contrastive_warmup_steps
        # small epsilon to avoid divide-by-zero
        self._eps = 1e-8

    def get_contrastive_memory(self):
        """
        Returns concatenated text/image embeddings from previous batches.
        Gradients must NOT flow through memory.
        """
        if len(self.text_queue) == 0:
            return None, None
    
        text_mem = torch.cat(list(self.text_queue), dim=0)
        image_mem = torch.cat(list(self.image_queue), dim=0)
    
        return text_mem, image_mem

    @property
    def vae(self): return self._vae[0]
    
    @property
    def text_encoder(self): return self._text_encoder[0]

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

    def compute_uot_weights(self, x0, x1, device):
        """Phase 1: Compute UOT-RFM majority score weights."""
        bsz = x0.shape[0]
        x0_np = x0.view(bsz, -1).detach().cpu().float().numpy()
        x1_np = x1.view(bsz, -1).detach().cpu().float().numpy()

        M = ot.dist(x0_np, x1_np, metric='sqeuclidean')
        M /= (M.max() + 1e-8)
        
        # Source-fixed UOT approximation
        pi_u = ot.unbalanced.sinkhorn_knopp_unbalanced(
            np.ones(bsz)/bsz, np.ones(bsz)/bsz, M, reg=0.05, reg_m=self.uot_tau
        )
        
        pi_u = torch.from_numpy(pi_u).to(device)
        weights = torch.pow(bsz * pi_u.sum(dim=0) + 1e-8, -self.uot_k)
        return weights / weights.mean()

    def compute_snr_weights(self, timesteps):
        """
        Computes Min-SNR weights for Flow Matching.
        timesteps (u): 0 (noise) -> 1 (data)
        """
        snr = (timesteps ** 2) / ((1 - timesteps) ** 2 + 1e-8)
        clamped_snr = torch.stack([snr, torch.full_like(snr, self.snr_gamma)], dim=1).min(dim=1)[0]
        weights = clamped_snr / (snr + 1e-8)
        return weights

    def compute_oat_pairing(self, x0, x1, prompt_embeds, device):
        """Phase 2: Re-pair noise(x0) to data(x1) minimizing acceleration cost."""
        bsz = x0.shape[0]

        # 1. Get velocities from the STABLE EMA model
        with torch.no_grad():
            t0 = torch.zeros(bsz, device=device)
            t1 = torch.ones(bsz, device=device)

            x0_input = list(x0.unsqueeze(2).unbind(dim=0))
            x1_input = list(x1.unsqueeze(2).unbind(dim=0))

            v0_raw = self.target_transformer(
                x0_input, t0, prompt_embeds, return_dict=False
            )[0]
            v0 = torch.stack(v0_raw, dim=0).squeeze(2)

            v1_raw = self.target_transformer(
                x1_input, t1, prompt_embeds, return_dict=False
            )[0]
            v1 = torch.stack(v1_raw, dim=0).squeeze(2)

        # 2. Compute OAT Cost Matrix (Vectorized)
        x0_f = x0.flatten(start_dim=1).cpu().double().numpy()
        x1_f = x1.flatten(start_dim=1).cpu().double().numpy()
        v0_f = v0.flatten(start_dim=1).cpu().double().numpy()
        v1_f = v1.flatten(start_dim=1).cpu().double().numpy()

        x1_b = x1_f[:, None, :]
        v1_b = v1_f[:, None, :]
        x0_b = x0_f[None, :, :]
        v0_b = v0_f[None, :, :]

        term1 = x1_b - x0_b - 0.5 * (v0_b + v1_b)
        dist_term1 = np.sum(term1**2, axis=2)

        term2 = v1_b - v0_b
        dist_term2 = np.sum(term2**2, axis=2)

        M = 12 * dist_term1 + dist_term2
        M /= (M.max() + 1e-8)

        T = ot.emd(ot.unif(bsz), ot.unif(bsz), M)
        pair_indices = np.argmax(T, axis=1) 
        
        return x0[pair_indices]

    def compute_siglip_loss(
        self,
        student_embeds,
        teacher_embeds,
        memory_queue=None,
        temperature=None,
        bias=-10.0, # Standard SigLIP bias init to keep probs low at start
    ):
        """
        True Sigmoid Loss (SigLIP) for Image-to-Image alignment.
        
        Args:
            student_embeds: (B, D) from Online Model (Short Prompt)
            teacher_embeds: (B, D) from EMA Model (Full Prompt)
            memory_queue:   (M, D) tensor of past teacher embeddings (Negatives)
        """
        if temperature is None:
            temperature = self.contrastive_temperature

        # Normalize
        student_embeds = F.normalize(student_embeds, dim=-1)
        teacher_embeds = F.normalize(teacher_embeds, dim=-1)
        
        device = student_embeds.device
        bsz = student_embeds.shape[0]

        # 1. Prepare Targets (Teacher + Memory)
        # We want to match the current batch teacher (Diagonals are positives)
        # Everything else (Off-diagonals + Memory) are negatives.
        if memory_queue is not None:
             # memory_queue should be (M, D)
            targets = torch.cat([teacher_embeds, memory_queue.to(device)], dim=0)
        else:
            targets = teacher_embeds

        # 2. Compute Logits
        # Shape: (B, B + M)
        logits = torch.matmul(student_embeds, targets.transpose(0, 1)) 
        logits = logits / temperature + bias

        # 3. Create Labels
        # Only the diagonals (0,0), (1,1)... of the first B columns are 1.
        # Everything else is 0.
        labels = 2 * torch.eye(bsz, targets.shape[0], device=device) - 1
        # labels is now 1 for positive, -1 for negative
        
        # 4. Sigmoid Loss
        # loss = -log(sigmoid(logits * labels))
        # strictly: log(1 + exp(-logits * labels))
        loss = -F.logsigmoid(logits * labels)
        
        # Sum over all pairs, normalize by batch size
        return torch.sum(loss) / bsz

    
    def forward(
        self,
        pixel_values,
        prompts,
        full_prompt,  # <--- Input the high-quality captions here
        device,
        phase="uot",
        weight_dtype=torch.float32,
        **kwargs
    ):
        if self.training:
            self.global_step += 1

        with torch.no_grad():
            # A. Encode SHORT prompts for Student
            prompts_in = [
                "" if (self.training and random.random() < self.caption_dropout_prob)
                else p for p in prompts
            ]
            prompt_embeds, _ = self.text_encoding_pipeline.encode_prompt(
                prompts_in, max_sequence_length=64, device=device, do_classifier_free_guidance=False
            )

            latents = self.vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)

        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        uot_weights = self.compute_uot_weights(noise, latents, device)
        
        # ... [OAT Pairing logic unchanged] ...
        
        u = compute_density_for_timestep_sampling(batch_size=bsz, **self.timestep_sampling_config).to(device)
        snr_weights = self.compute_snr_weights(u.flatten())
        final_weights = uot_weights * snr_weights
        
        sigmas = u.view(bsz, 1, 1, 1)
        noisy_latents = (1 - sigmas) * noise + sigmas * latents
        noisy_latents_input = list(noisy_latents.unsqueeze(2).unbind(dim=0))

        # --- STUDENT (Online) ---
        # Conditioned on: Noisy/Short Prompts
        model_pred, img_online = self.transformer(
            noisy_latents_input, u.flatten(), prompt_embeds, return_dict=True
        )
        
        # --- Reconstruction Loss ---
        model_pred = torch.stack(model_pred, dim=0).squeeze(2)
        model_pred2 = torch.stack(img_online, dim=0).squeeze(2)

        target = latents - noise
        loss_recon = self.charbonnier(model_pred.float(), target.float(), weights=final_weights)
        loss_contrastive = self.charbonnier(model_pred2.float(), target.float(), weights=final_weights)
        # --- SigLIP Image-to-Image Alignment ---
        #loss_contrastive = torch.zeros((), device=device, dtype=loss_recon.dtype)
        
        return loss_recon, loss_contrastive
    
    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=50, use_ema=True, seed=None, device=None, **kwargs):
        """
        Generates images. 
        Args:
            use_ema (bool): If True, uses the stable target_transformer (Recommended for OAT).
        """
        if device is None: 
            device = next(self.transformer.parameters()).device
        
        # Select Model for Inference
        inference_model = self.target_transformer if use_ema else self.transformer
        inference_model.eval()
        
        pipeline = ZImagePipeline(
            transformer=inference_model,
            vae=self.vae, 
            text_encoder=self.text_encoder, 
            tokenizer=self.tokenizer,
            scheduler=self.noise_scheduler
        )
        pipeline.to(device)
        
        generator = torch.Generator(device=device).manual_seed(seed) if seed else None
        if isinstance(prompt, str): prompt = [prompt]
        
        images = pipeline(
                prompt=prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale= 4
            ).images
        
        return images, prompt