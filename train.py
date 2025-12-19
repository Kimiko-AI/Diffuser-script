#!/usr/bin/env python
import argparse
import yaml
import os
import shutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import logging
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from trainer.data import get_dataloader
from trainer.models import load_models
from trainer.utils import log_validation, save_model_card
from trainer.models.zimage_wrapper import ZImageWrapper
from trainer.models.sana_wrapper import SanaWrapper
from contextlib import nullcontext

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

        writer = None
    else:
        logger.setLevel(logging.ERROR)
        writer = None

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)

    # Load Models
    # Models are loaded to CPU initially by default in most diffusers pipelines or custom loaders
    # We need to move them to the correct device
    noise_scheduler, tokenizer, text_encoder, vae, transformer = load_models(args)

    # Determine dtypes
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device
    vae.to(device, dtype=torch.float32)  # VAE usually kept in fp32 to avoid NaN

    text_encoder_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    text_encoder.to(device, dtype=text_encoder_dtype)

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

    # === RESUME LOGIC ===
    global_step = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None

        if path and os.path.exists(path):
            if rank == 0:
                print(f"Resuming from checkpoint {path}")

            # Load model state
            # We need to load onto CPU or correct device
            # DDP wraps model in .module
            model_to_load = model_wrapper.module if hasattr(model_wrapper, "module") else model_wrapper

            # Check for safetensors or bin
            if os.path.exists(os.path.join(path, "transformer", "diffusion_pytorch_model.safetensors")):
                from safetensors.torch import load_file
                state_dict = load_file(os.path.join(path, "transformer", "diffusion_pytorch_model.safetensors"))
            else:
                # Fallback/standard bin
                state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=device)

            # This logic assumes the checkpoint structure matches what we save.
            # If we saved via save_pretrained, we need to load into transformer specifically.
            # The previous code saved: unwrapped_model.transformer.save_pretrained(...)
            # So we should load into model_wrapper.transformer

            # However, standard diffusers loading usually handles this via from_pretrained.
            # Since we instantiated transformer separately in load_models, we can try loading weights.
            # But wait, save_pretrained saves config + weights.

            # Let's rely on standard diffusers loading mechanism if possible,
            # BUT we already created the model instance.
            # We should probably use `load_state_dict`.
            # If the checkpoint is a full diffusers dump (config+weights), we can load it.

            # Simpler approach: If we saved using transformer.save_pretrained, we can reload it into the transformer.
            try:
                from diffusers.models import Transformer2DModel
                # Re-load weights into the transformer
                # Note: This is inefficient (double load), but safe.
                # Actually, load_models already loaded the pretrained path.
                # If resuming, we want to overwrite with the checkpoint weights.

                # Check if it's a diffusers compatible folder
                if os.path.isdir(os.path.join(path, "transformer")):  # It was saved as subfolder
                    loaded_transformer = Transformer2DModel.from_pretrained(os.path.join(path, "transformer")).to(
                        device, dtype=weight_dtype)
                    if hasattr(model_wrapper, "module"):
                        model_wrapper.module.transformer.load_state_dict(loaded_transformer.state_dict())
                    else:
                        model_wrapper.transformer.load_state_dict(loaded_transformer.state_dict())
                    del loaded_transformer

                # Try to determine global step
                try:
                    global_step = int(os.path.basename(path).split("-")[1])
                except:
                    pass
            except Exception as e:
                if rank == 0:
                    print(f"Failed to load checkpoint weights: {e}")

        else:
            if rank == 0:
                print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting from scratch.")

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
        # We need to sync gradients only on the last step of accumulation
        # Current step count relative to global step isn't tracked perfectly by simple enumeration
        # because WDS is infinite. We'll simulate accumulation locally.

        # NOTE: standard accumulation usually works by stepping optimizer every N steps.
        # But global_step increments every optimizer step.

        # We process 'gradient_accumulation_steps' micro-batches per global step.
        # So we can run a mini-loop or just use a counter.
        # To match the original logic roughly:

        # We'll do the standard way: accumulate gradients, then step.
        # For DDP, we use no_sync() except for the last accumulation step.

        # However, getting "last step" in an infinite loop context with `iter` is tricky if we don't track it.
        # We will iterate accumulation_steps times for one global_step.

        accum_loss = 0.0

        for i in range(args.gradient_accumulation_steps):
            # Fetch data
            if i > 0:  # We already fetched one batch before the loop if we restructure...
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

            is_last_accum = (i == args.gradient_accumulation_steps - 1)

            # Context for DDP sync
            # If it's NOT the last accumulation step, we DO NOT sync gradients -> no_sync()
            # If it IS the last step, we sync -> default context
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
                loss.backward()
                accum_loss += loss.item()

        # Step
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)

        # Logs
        if rank == 0:
            current_lr = lr_scheduler.get_last_lr()[0]
            logs = {"loss": accum_loss, "lr": current_lr}
            if writer:
                writer.add_scalar("train/loss", accum_loss, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
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
                    log_validation(
                        model_wrapper=model_wrapper,
                        args=args,
                        global_step=global_step,
                        device=device,
                        writer=writer
                    )
                torch.cuda.empty_cache()
                gc.collect()

            if world_size > 1:
                dist.barrier()

    if rank == 0:
        print("Training finished.")
        if writer:
            writer.close()
        if _has_wandb and wandb.run:
            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
