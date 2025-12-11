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
from trainer.dataset import get_wds_loader
from trainer.model import load_models
from trainer.utils import log_validation, save_model_card
from trainer.ZImage import ZImageWrapper
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
    model_wrapper = ZImageWrapper(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        timestep_sampling_config=timestep_sampling_config,
        caption_dropout_prob=getattr(args, "caption_dropout_prob", 0.0),
        afm_lambda=getattr(args, "afm_lambda", 0.0)
    )
    
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

    # Optimizer (optimize only transformer parameters)
    optimizer = bnb.optim.Adam8bit(
        model_wrapper.transformer.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    # Dataset (WebDataset)
    dataloader = get_wds_loader(
        url_pattern=args.data_url,
        batch_size=args.train_batch_size,
        num_workers=getattr(args, "dataloader_num_workers", 8),
        is_train=True,
        base_resolution=getattr(args, "resolution", 256),
        bucket_step_size=getattr(args, "bucket_step_size", 32)
    )

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
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            
            # Ensure EMA model is on the correct device after loading state
            if getattr(args, "use_ema", False) and ema_model is not None:
                ema_model.to(accelerator.device)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Training Loop
    data_iter = iter(dataloader)

    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model_wrapper):
            images = batch["pixels"].to(accelerator.device, dtype=weight_dtype)
            prompts = batch["prompts"]
            
            # Forward pass through wrapper
            loss = model_wrapper(images, prompts, accelerator.device, weight_dtype)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model_wrapper.module.transformer.parameters() if hasattr(model_wrapper, "module") else model_wrapper.transformer.parameters(), args.max_grad_norm)

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
                    accelerator.save_state(save_path)
                    
                    # Access unwrapped transformer for saving model card info if needed
                    unwrapped_transformer = accelerator.unwrap_model(model_wrapper).transformer
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
