#!/usr/bin/env python
import argparse
import yaml
import os
import shutil
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, EMAModel
from trainer.data import get_dataloader
from trainer.models import load_models
from trainer.utils import log_validation, save_model_card
from trainer.models.zimage_wrapper import ZImageWrapper
from trainer.models.sana_wrapper import SanaWrapper
import bitsandbytes as bnb

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args, unknown = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=args.config) # Re-add config to avoid error
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
        print(args)
        accelerator.init_trackers("zimage-training", config=vars(args))

    set_seed(args.seed)

    # Load Models
    noise_scheduler, tokenizer, text_encoder, vae, transformer = load_models(args)

    # Prepare Wrapper
    # Move frozen models to device/dtype first if needed, or let accelerator handle it.
    # For VAE/TextEncoder, we often want specific dtypes.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    
    # Use bf16 for text encoder if mixed precision is bf16, otherwise float32 for stability
    text_encoder_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    text_encoder.to(accelerator.device, dtype=text_encoder_dtype)
    
    # Create Wrapper
    timestep_sampling_config = getattr(args, "timestep_sampling", None)
    model_type = getattr(args, "model_type", "zimage")
    
    if model_type == "sana":
        model_wrapper = SanaWrapper(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            args=args
        )
    elif model_type == "zimage":
        model_wrapper = ZImageWrapper(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            timestep_sampling_config=timestep_sampling_config,
            caption_dropout_prob=getattr(args, "caption_dropout_prob", 0.0),
            afm_lambda=getattr(args, "afm_lambda", 0.0),
            consistency_lambda=getattr(args, "consistency_lambda", 1.0)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if args.gradient_checkpointing:
        model_wrapper.transformer.enable_gradient_checkpointing()

    # EMA
    ema_model = None
    if getattr(args, "use_ema", False):
        ema_model = EMAModel(
            model_wrapper.transformer.parameters(),
            decay=getattr(args, "ema_decay", 0.9999),
            update_after_step=getattr(args, "ema_update_after_step", 0),
            model_cls=type(model_wrapper.transformer),
            model_config=model_wrapper.transformer.config,
        )
        ema_model.to(accelerator.device)
        accelerator.register_for_checkpointing(ema_model)

    # Optimizer (optimize transformer and refiner parameters)
    # ZImageWrapper only registers transformer and refiner as submodules (vae/text_encoder are hidden in lists)
    optimizer = torch.optim.AdamW(
        model_wrapper.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    # Dataset (WebDataset)
    dataloader = get_dataloader(args)

    # Scheduler
    lr_scheduler = get_scheduler(
        getattr(args, "lr_scheduler_type", "constant"),
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    # Prepare
    # Accelerator will handle the wrapper (DDP, etc.)
    model_wrapper, optimizer, lr_scheduler = accelerator.prepare(
        model_wrapper, optimizer, lr_scheduler
    )

    # Resume from checkpoint
    global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting from scratch."
            )
        else:
            accelerator.print(f"Resuming model weights from checkpoint {path}")
            load_path = os.path.join(args.output_dir, path)
            unwrapped_model = accelerator.unwrap_model(model_wrapper)
            
            # 1. Load Model Weights
            safetensors_file = os.path.join(load_path, "diffusion_pytorch_model.safetensors")
            bin_file = os.path.join(load_path, "diffusion_pytorch_model.bin")
            
            state_dict = None
            if os.path.exists(safetensors_file):
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_file)
            elif os.path.exists(bin_file):
                state_dict = torch.load(bin_file, map_location="cpu")
            
            if state_dict is not None:
                m, u = unwrapped_model.transformer.load_state_dict(state_dict, strict=False)
                accelerator.print(f"Weights loaded. Missing keys: {len(m)}, Unexpected keys: {len(u)}")
            else:
                accelerator.print(f"No weights found at {load_path}")

            # 2. Load Training State (Optimizer, Scheduler, Step)
            state_file = os.path.join(load_path, "training_state.pt")
            if os.path.exists(state_file):
                accelerator.print(f"Loading training state from {state_file}")
                # Map location to cpu to avoid potential OOM or device mismatch during load
                training_state = torch.load(state_file, map_location="cpu")
                global_step = training_state.get("global_step", 0)
                
                if "scheduler" in training_state:
                    lr_scheduler.load_state_dict(training_state["scheduler"])
                
                accelerator.print(f"Resumed training state at global step {global_step}")
            else:
                accelerator.print("No training state file found. Resuming with fresh optimizer/scheduler.")

            # Ensure EMA model is on the correct device after loading state
            if getattr(args, "use_ema", False) and ema_model is not None:
                ema_model.to(accelerator.device)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Training Loop
    data_iter = iter(dataloader)
    
    # Fast-forward dataloader
    if global_step > 0:
        accelerator.print(f"Skipping {global_step} batches to resume dataset position...")
        # Only main process logs progress, but all processes must consume the iterator to stay in sync
        # if using distributed sampler or wds with same split
        for _ in tqdm(range(global_step), desc="Skipping batches", disable=not accelerator.is_local_main_process):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                next(data_iter)

    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model_wrapper):
            images = batch["pixels"].to(accelerator.device, dtype=weight_dtype)
            prompts = batch["prompts"]
            full_prompts = batch["full_prompts"] # Get full prompts
            
            # Forward pass through wrapper
            loss = model_wrapper(
                pixel_values=images, 
                prompts=prompts,
                device=accelerator.device, 
               # paraphrased_prompts=full_prompts, # Pass full prompts as paraphrases
                weight_dtype=weight_dtype
            )

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                # Clip gradients for all trainable parameters (transformer + refiner)
                params_to_clip = model_wrapper.module.parameters() if hasattr(model_wrapper, "module") else model_wrapper.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if getattr(args, "use_ema", False) and ema_model is not None:
            # Unwrap model if needed to access transformer
            unwrapped_model = accelerator.unwrap_model(model_wrapper)
            ema_model.step(unwrapped_model.transformer.parameters())

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # accelerator.save_state(save_path)
                    
                    # Access unwrapped transformer for saving model card info if needed
                    unwrapped_transformer = accelerator.unwrap_model(model_wrapper).transformer
                    unwrapped_transformer.save_pretrained(save_path)
                    
                    # Save training state (Step, Optimizer, Scheduler)
                    # We save this manually because we aren't using accelerator.save_state() anymore
                    torch.save({
                        "global_step": global_step,
                        "scheduler": lr_scheduler.state_dict()
                    }, os.path.join(save_path, "training_state.pt"))
                    
                    save_model_card(
                        repo_id=f"lumina2-step-{global_step}",
                        base_model=args.pretrained_model_name_or_path or "scratch",
                        repo_folder=save_path
                    )

                    # Rotation: Keep only the latest N checkpoints
                    if getattr(args, "checkpoints_total_limit", None) is not None:
                        limit = int(args.checkpoints_total_limit)
                        checkpoints = os.listdir(args.output_dir)
                        # Filter for checkpoint folders
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
                        # Sort by step number
                        try:
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) > limit:
                                num_to_remove = len(checkpoints) - limit
                                removing_checkpoints = checkpoints[:num_to_remove]
                                for rc in removing_checkpoints:
                                    full_path = os.path.join(args.output_dir, rc)
                                    if os.path.isdir(full_path):
                                        shutil.rmtree(full_path)
                                        logger.info(f"Removed old checkpoint {rc} to respect total_limit={limit}")
                        except Exception as e:
                            logger.warning(f"Error during checkpoint rotation: {e}")

                if global_step % args.validation_steps == 0:
                    # Unwrap for validation not needed here as log_validation handles unwrap or receives wrapper
                    # We pass the wrapper directly (it might be wrapped by accelerator, log_validation handles unwrapping if needed)
                    log_validation(
                        accelerator=accelerator,
                        model_wrapper=model_wrapper,
                        args=args,
                        global_step=global_step,
                        ema_model=ema_model
                    )

        logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps: break

    accelerator.end_training()


if __name__ == "__main__":
    main()
