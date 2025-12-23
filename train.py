#!/usr/bin/env python
import argparse
import yaml
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import logging
import math
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from trainer.data import get_dataloader
from trainer.models import load_models
from trainer.utils import log_validation, save_model_card
from trainer.models.zimage_wrapper import ZImageWrapper
from trainer.models.sana_wrapper import SanaWrapper
from contextlib import nullcontext

# WandB check
try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args, unknown = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=args.config)
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            parser.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
        else:
            parser.add_argument(f"--{k}", default=v)

    # Standard DDP arguments
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    return parser.parse_args()


def main():
    args = parse_args()

    # DDP Initialization
    # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        print("Distributed environment not detected, running in single process mode.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Setup logging only on main process
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Running on {world_size} processes.")
        print(args)

        # Initialize trackers
        if args.report_to == "wandb" and _has_wandb:
            wandb.init(project="zimage-training", config=vars(args), dir=args.output_dir)

    else:
        logger.setLevel(logging.ERROR)

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)

    # Determine dtypes
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load Models
    # Models are loaded directly to device using device_map and low_cpu_mem_usage
    noise_scheduler, tokenizer, text_encoder, vae, transformer = load_models(args, device=device, weight_dtype=weight_dtype)

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

    # Move wrapper to device (components are already on device but wrapper itself needs move)
    model_wrapper = model_wrapper.to(device)

    # Wrap DDP
    if world_size > 1:
        model_wrapper = DDP(model_wrapper, device_ids=[local_rank], output_device=local_rank)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model_wrapper.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    # Dataset
    # WebDataset automatically handles splitting with split_by_node if dist is initialized
    dataloader = get_dataloader(args)

    # Scheduler
    lr_scheduler = get_scheduler(
        getattr(args, "lr_scheduler_type", "constant"),
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    # Scaler for FP16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    # === RESUME LOGIC ===
    global_step = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if path and os.path.exists(path):
            if rank == 0:
                print(f"Resuming from checkpoint {path}")
            
            # Load transformer weights
            transformer_path = os.path.join(path, "transformer")
            if os.path.exists(transformer_path):
                # Load state dict into transformer directly
                # diffusers models have from_pretrained, but here we already have the object
                # So we use load_state_dict or a similar mechanism if we want to avoid re-instantiation
                # However, for simplicity and ensuring correct loading:
                if hasattr(transformer, "load_state_dict"):
                    from diffusers.models.modeling_utils import load_state_dict
                    state_dict = load_state_dict(os.path.join(transformer_path, "diffusion_pytorch_model.safetensors"))
                    transformer.load_state_dict(state_dict)
                    del state_dict
                
            global_step = int(path.split("-")[-1])

    # Training Loop
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=(rank != 0))
    progress_bar.set_description("Steps")

    data_iter = iter(dataloader)

    # Mixed Precision Context
    amp_context = torch.amp.autocast('cuda', dtype=weight_dtype)

    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Gradient Accumulation Logic
        accum_loss = 0.0

        for i in range(args.gradient_accumulation_steps):
            # Fetch data if not the first step in accumulation
            if i > 0:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

            is_last_accum = (i == args.gradient_accumulation_steps - 1)

            # Sync gradients only on the last step of accumulation
            if world_size > 1 and not is_last_accum:
                context = model_wrapper.no_sync()
            else:
                context = nullcontext()

            with context:
                images = batch["pixels"].to(device, dtype=weight_dtype)
                prompts = batch["prompts"]

                with amp_context:
                    loss = model_wrapper(
                        pixel_values=images,
                        prompts=prompts,
                        device=device,
                        weight_dtype=weight_dtype
                    )
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                scaler.scale(loss).backward()
                accum_loss += loss.item()

        # Step
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), args.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)

        # Logs
        if rank == 0:
            current_lr = lr_scheduler.get_last_lr()[0]
            logs = {"loss": accum_loss, "lr": current_lr}
            if _has_wandb and wandb.run:
                wandb.log(logs, step=global_step)

            progress_bar.set_postfix(**logs)

        lr_scheduler.step()

        # === VALIDATION & SAVING ===
        # We should sync before saving/validating

        if global_step % args.checkpointing_steps == 0 or global_step % args.validation_steps == 0:
            if world_size > 1:
                dist.barrier()

        # Save
        if global_step % args.checkpointing_steps == 0:
            if rank == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)

                try:
                    # Unwrap
                    if hasattr(model_wrapper, "module"):
                        unwrapped = model_wrapper.module
                    else:
                        unwrapped = model_wrapper

                    # Save Transformer
                    # We usually want to save it such that it can be loaded by diffusers
                    # The original code used unwrapped.transformer.save_pretrained
                    unwrapped.transformer.save_pretrained(
                        os.path.join(save_path, "transformer")
                    )

                    save_model_card(
                        repo_id=f"lumina2-step-{global_step}",
                        base_model=args.pretrained_model_name_or_path or "scratch",
                        repo_folder=save_path
                    )

                    # Rotation Logic
                    if getattr(args, "checkpoints_total_limit", None) is not None:
                        limit = int(args.checkpoints_total_limit)
                        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                        try:
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) > limit:
                                num_to_remove = len(checkpoints) - limit
                                removing_checkpoints = checkpoints[:num_to_remove]
                                for rc in removing_checkpoints:
                                    full_path = os.path.join(args.output_dir, rc)
                                    if os.path.isdir(full_path):
                                        shutil.rmtree(full_path)
                                        logger.info(f"Removed old checkpoint {rc}")
                        except Exception as e:
                            logger.warning(f"Rotation error: {e}")

                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")

        # Validate
        if global_step % args.validation_steps == 0:
            if rank == 0:
                with torch.no_grad():
                    # We pass the wrapper. log_validation handles unwrapping now.
                    log_validation(
                        model_wrapper=model_wrapper,
                        args=args,
                        global_step=global_step,
                        device=device
                    )

            if world_size > 1:
                dist.barrier()

    if rank == 0:
        print("Training finished.")
        if _has_wandb and wandb.run:
            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
