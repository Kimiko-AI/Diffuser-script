#!/usr/bin/env python
import argparse
import yaml
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from diffusers import Lumina2Pipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

from trainer.dataset import get_wds_loader
from trainer.model import load_models
from trainer.utils import log_validation, save_model_card

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args, unknown = parser.parse_known_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            parser.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
        else:
            parser.add_argument(f"--{k}", default=v)
            
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers("lumina2-training", config=vars(args))

    set_seed(args.seed)
    
    # Load Models
    noise_scheduler, tokenizer, text_encoder, vae, transformer = load_models(args)
    
    transformer.train()
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        
    # Optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    
    # Dataset (WebDataset)
    dataloader = get_wds_loader(
        url_pattern=args.data_url,
        batch_size=args.train_batch_size,
        num_workers=4, # Adjustable
        is_train=True
    )
    
    # Scheduler
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, 
        num_warmup_steps=args.lr_warmup_steps, 
        num_training_steps=args.max_train_steps
    )
    
    # Prepare
    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.bfloat16)

    # Text Encoding Helper
    text_encoding_pipeline = Lumina2Pipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=None, scheduler=None
    )
    
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    # Training Loop
    data_iter = iter(dataloader)
    
    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        with accelerator.accumulate(transformer):
            images = batch["pixels"].to(accelerator.device, dtype=weight_dtype)
            prompts = batch["prompts"]

            # 1. Text Encode
            with torch.no_grad():
                    prompt_embeds, prompt_mask, _, _ = text_encoding_pipeline.encode_prompt(
                    prompts, max_sequence_length=256, device=accelerator.device
                )
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
            
            # 2. VAE Encode
            with torch.no_grad():
                latents = vae.encode(images.to(dtype=torch.float32)).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)
            
            # 3. Noise & Flow Matching
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            u = compute_density_for_timestep_sampling(batch_size=bsz).to(latents.device)
            
            # Interpolate
            sigmas = u
            noisy_latents = (1.0 - sigmas.view(bsz, 1, 1, 1)) * noise + sigmas.view(bsz, 1, 1, 1) * latents
            
            # 4. Predict
            model_pred = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                timestep=u.flatten(), 
                return_dict=False
            )[0]
            
            # 5. Loss
            target = latents - noise
            loss = F.mse_loss(model_pred.float(), target.float())
            
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)
            
            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    save_model_card(
                        repo_id=f"lumina2-step-{global_step}",
                        base_model=args.pretrained_model_name_or_path or "scratch",
                        repo_folder=save_path
                    )
                
                if global_step % args.validation_steps == 0:
                    log_validation(
                        accelerator=accelerator,
                        transformer=transformer,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        scheduler=noise_scheduler,
                        args=args,
                        global_step=global_step
                    )
        
        logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
        accelerator.log(logs, step=global_step)
        
        if global_step >= args.max_train_steps: break

    accelerator.end_training()

if __name__ == "__main__":
    main()
