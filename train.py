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
from datetime import timedelta
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from trainer.data import get_dataloader
from trainer.models import load_models
from trainer.utils import log_validation, save_model_card
from trainer.models.zimage_wrapper import ZImageWrapper
from trainer.models.sana_wrapper import SanaWrapper
from contextlib import nullcontext
from pytorch_optimizer.optimizer import ScheduleFreeAdamW

# WandB check
try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_config(config, parent_key='', sep='_'):
    items = []
    for k, v in config.items():
        # Special handling for lists like validation_prompt to keep them as is
        if isinstance(v, dict) and not (k == "timestep_sampling"): # Keep timestep_sampling as dict if needed by wrapper
             # But wait, original code accessed args.timestep_sampling as a dict? 
             # Let's check usage. 
             # usage: timestep_sampling_config = getattr(args, "timestep_sampling", None)
             # So timestep_sampling should remain a dict in args.
             # However, simple flattening would make it timestep_sampling_weighting_scheme etc.
             # We should probably flatten but ALSO keep sub-dicts if they represent coherent config objects.
             
             # Let's just flatten everything recursively for now, but also check how to handle
             # things that were top-level before. 
             # Actually, the best approach for this refactor without breaking code is:
             # 1. Load config
             # 2. Add keys to parser.
             # If we have sections, we can flatten them into the top level namespace
             # e.g. training.learning_rate -> args.learning_rate
             
             items.extend(flatten_config(v, parent_key='', sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args, unknown = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Flatten the config for argument parsing
    # This allows sections in yaml (e.g. model: learning_rate) to be accessed as args.learning_rate
    # which matches current code expectation.
    flat_config = {}
    
    def flatten(d):
        for k, v in d.items():
            if isinstance(v, dict) and k != "timestep_sampling" and k != "model_config": # Exception for known dict args
                 flatten(v)
            else:
                flat_config[k] = v
    
    flatten(config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=args.config)
    
    for k, v in flat_config.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            parser.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
        elif isinstance(v, list):
             # Handle lists (like validation prompts)
             # argparse doesn't handle lists well by default in this dynamic way without 'nargs'
             # but since we set default=v, it works for internal usage.
             # If passed via CLI, user might need to pass multiple times or we need specific handling.
             # For now, we assume these are mostly set in config.
             parser.add_argument(f"--{k}", default=v)
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
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=2))
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

    # Use the factory from trainer/models/__init__.py
    from trainer.models import get_model_wrapper
    
    # Common kwargs
    wrapper_kwargs = {
        "transformer": transformer,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
        "args": args
    }
    
    # Add model specific args if needed
    if model_type == "zimage":
        wrapper_kwargs.update({
            "timestep_sampling_config": timestep_sampling_config,
            "caption_dropout_prob": getattr(args, "caption_dropout_prob", 0.0),
            "afm_lambda": getattr(args, "afm_lambda", 0.0),
            "consistency_lambda": getattr(args, "consistency_lambda", 1.0)
        })
    
    model_wrapper = get_model_wrapper(model_type, **wrapper_kwargs)

    if args.gradient_checkpointing:
        model_wrapper.transformer.enable_gradient_checkpointing()

    # Move wrapper to device (components are already on device but wrapper itself needs move)
    model_wrapper = model_wrapper.to(device)

    # Wrap DDP
    if world_size > 1:
        model_wrapper = DDP(model_wrapper, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Optimizer
    optimizer = ScheduleFreeAdamW(
        model_wrapper.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    # Dataset
    # WebDataset automatically handles splitting with split_by_node if dist is initialized
    dataloader = get_dataloader(args)

    # Scheduler
    #lr_scheduler = get_scheduler(
    #    getattr(args, "lr_scheduler_type", "constant"),
    #    optimizer=optimizer,
    #    num_warmup_steps=args.lr_warmup_steps,
    #    num_training_steps=args.max_train_steps
    #)

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
                if hasattr(transformer, "load_state_dict"):
                    st_path = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
                    bin_path = os.path.join(transformer_path, "diffusion_pytorch_model.bin")
                    
                    state_dict = None
                    if os.path.exists(st_path):
                        try:
                            from safetensors.torch import load_file
                            state_dict = load_file(st_path)
                        except ImportError:
                            from diffusers.models.modeling_utils import load_state_dict
                            state_dict = load_state_dict(st_path)
                    elif os.path.exists(bin_path):
                        state_dict = torch.load(bin_path, map_location="cpu")
                    
                    if state_dict is not None:
                        # Load with strict=False to ignore shape mismatches
                        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
                        if rank == 0:
                            if len(missing) > 0:
                                print(f"Missing keys when loading transformer: {len(missing)}")
                            if len(unexpected) > 0:
                                print(f"Unexpected keys when loading transformer: {len(unexpected)}")
                        del state_dict
                    else:
                        if rank == 0:
                            print(f"No model weights found in {transformer_path}, skipping weight load.")
                
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
        accum_logs = {}

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
                crop_coords = batch.get("crop_coords", None)

                with amp_context:
                    model_output = model_wrapper(
                        pixel_values=images,
                        prompts=prompts,
                        crop_coords=crop_coords,
                        device=device,
                        weight_dtype=weight_dtype
                    )
                    
                    if isinstance(model_output, dict):
                        loss = model_output["loss"]
                        # Accumulate other metrics
                        for k, v in model_output.items():
                            if k not in accum_logs:
                                accum_logs[k] = 0.0
                            accum_logs[k] += v.item() / args.gradient_accumulation_steps
                    else:
                        loss = model_output
                        accum_logs["loss"] = accum_logs.get("loss", 0.0) + loss.item() / args.gradient_accumulation_steps

                    loss = loss / args.gradient_accumulation_steps

                # Backward
                scaler.scale(loss).backward()
                accum_loss += loss.item()

        # Step
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), args.max_grad_norm)
        else:
            # Calculate grad norm even if not clipping for logging
            total_norm = 0.0
            for p in model_wrapper.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norm = torch.tensor(grad_norm) # consistency

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)

        # Logs
        if rank == 0:
            current_lr = 0.0002
            logs = {"lr": current_lr, "grad_norm": grad_norm.item()}
            logs.update(accum_logs) # Add all accumulated losses
            
            if _has_wandb and wandb.run:
                wandb.log(logs, step=global_step)

            progress_bar.set_postfix(**logs)

        #lr_scheduler.step()

        # === VALIDATION & SAVING ===
        # We should sync before saving/validating

        if global_step % args.checkpointing_steps == 0 or global_step % args.validation_steps == 0 or global_step == 1:
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
        if global_step % args.validation_steps == 0 or global_step == 1:
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
