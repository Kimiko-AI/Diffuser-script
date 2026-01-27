#!/usr/bin/env python
import argparse
import yaml
import os
import shutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from datetime import timedelta
from tqdm.auto import tqdm
from trainer.data import get_dataloader
from trainer.models import load_models
from trainer.utils import log_validation, save_model_card
from contextlib import nullcontext
from pytorch_optimizer.optimizer import ScheduleFreeAdamW

def is_valid_prompt(p):
    return p is not None and isinstance(p, str) and p.strip() != ""


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
        if isinstance(v, dict) and not (k == "timestep_sampling"): 
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

    flat_config = {}
    
    def flatten(d):
        for k, v in d.items():
            if isinstance(v, dict) and k != "timestep_sampling" and k != "model_config": 
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
             parser.add_argument(f"--{k}", default=v)
        else:
            parser.add_argument(f"--{k}", default=v)

    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    return parser.parse_args()


# --- NEW HELPER FUNCTION ---
def select_validation_prompts(dataloader, keyword="general", count=16):
    """
    Selects validation prompts:
    - ~50% containing `keyword` (general prompts)
    - ~50% from caption_detailed or caption_long (prefer detailed),
      regardless of keyword presence.
    """

    logger.info(
        f"Selecting {count} validation prompts "
        f"(~50% '{keyword}', ~50% captions)..."
    )

    target_general = count // 2
    target_caption = count - target_general

    general_prompts = []
    caption_prompts = []

    temp_iter = iter(dataloader)

    def is_valid(p):
        return isinstance(p, str) and p.strip() != ""

    with tqdm(total=count, desc="Finding Prompts", unit="p") as pbar:
        while len(general_prompts) < target_general or len(caption_prompts) < target_caption:
            try:
                batch = next(temp_iter)
            except StopIteration:
                logger.warning("Dataset exhausted before reaching target counts.")
                break

            batch_size = len(batch.get("prompts", []))

            for i in range(batch_size):
                # ---- General prompts (keyword-based) ----
                if len(general_prompts) < target_general:
                    for key in ("full_prompts", "prompts"):
                        p = batch.get(key, [None] * batch_size)[i]
                        if is_valid(p) and keyword.lower() in p.lower():
                            if p not in general_prompts:
                                general_prompts.append(p)
                                pbar.update(1)
                            break

                # ---- Caption prompts (preferred detailed) ----
                if len(caption_prompts) < target_caption:
                    p = batch.get("caption_detailed", [None] * batch_size)[i]
                    if not is_valid(p):
                        p = batch.get("caption_long", [None] * batch_size)[i]

                    if is_valid(p) and p not in caption_prompts:
                        caption_prompts.append(p)
                        pbar.update(1)

                if len(general_prompts) >= target_general and len(caption_prompts) >= target_caption:
                    break

    selected = general_prompts[:target_general] + caption_prompts[:target_caption]

    logger.info(
        f"Selected {len(selected)} validation prompts "
        f"({len(general_prompts)} general, {len(caption_prompts)} captions)."
    )

    return selected

# ---------------------------


def main():
    args = parse_args()

    # DDP Initialization
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="nccl", 
                    init_method="env://", 
                    timeout=timedelta(hours=2),
                    device_id=torch.device(f"cuda:{local_rank}")
                )
            except TypeError:
                dist.init_process_group(
                    backend="nccl", 
                    init_method="env://", 
                    timeout=timedelta(hours=2)
                )
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print("Distributed environment not detected, running in single process mode.")

    # Setup logging only on main process
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Running on {world_size} processes.")
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
    args.device = device
    
    # Load Models
    noise_scheduler, tokenizer, text_encoder, vae, transformer = load_models(args, device=device, weight_dtype=weight_dtype)

    # Create Wrapper
    timestep_sampling_config = getattr(args, "timestep_sampling", None)
    model_type = getattr(args, "model_type", "zimage")

    from trainer.models import get_model_wrapper
    
    wrapper_kwargs = {
        "transformer": transformer,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "noise_scheduler": noise_scheduler,
        "args": args
    }
    
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

    model_wrapper = model_wrapper.to(device)

    if world_size > 1:
        model_wrapper = DDP(model_wrapper, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = ScheduleFreeAdamW(
        model_wrapper.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    # Dataset
    dataloader = get_dataloader(args)

    # === DYNAMIC PROMPT SELECTION LOGIC ===
    # We do this here, before the training loop starts, so we have the prompts ready for all validations.
    if rank == 0:
        # We only scan on Rank 0
        dynamic_prompts = select_validation_prompts(dataloader, keyword="general", count=32)
        if dynamic_prompts:
            args.validation_prompt = dynamic_prompts
            logger.info(f"Updated validation prompts: {args.validation_prompt}")
        else:
            logger.warning("Could not find enough prompts with keyword 'general'. Keeping config prompts.")
    # ======================================

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    # === RESUME LOGIC ===
    global_step = 0
    path = args.resume_from_checkpoint
    
    if path:
        if rank == 0:
            if path == "latest":
                if os.path.exists(args.output_dir):
                    dirs = os.listdir(args.output_dir)
                    dirs = [d for d in dirs if d.startswith("checkpoint-")]
                    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                    path = os.path.join(args.output_dir, dirs[-1]) if len(dirs) > 0 else None
                else:
                    path = None
        
        if world_size > 1:
            object_list = [path]
            dist.broadcast_object_list(object_list, src=0)
            path = object_list[0]

        if path and os.path.exists(path):
            if rank == 0:
                print(f"Resuming from checkpoint {path}")
            
            # transformer_path = os.path.join(path, "transformer")
            # unwrapped.transformer.load_pretrained... (Assuming handled by framework or manually here)
                
            global_step = int(path.split("-")[-1]) 

    # Training Loop
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=(rank != 0))
    progress_bar.set_description("Steps")

    data_iter = iter(dataloader)
    amp_context = torch.amp.autocast('cuda', dtype=weight_dtype)

    while global_step < args.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        accum_loss = 0.0
        accum_logs = {}

        for i in range(args.gradient_accumulation_steps):
            if i > 0:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

            is_last_accum = (i == args.gradient_accumulation_steps - 1)

            if world_size > 1 and not is_last_accum:
                context = model_wrapper.no_sync()
            else:
                context = nullcontext()
            
            with context:
                batch_size = len(batch["prompts"])
                
                gen = torch.Generator(device="cpu").manual_seed(int(global_step + 432))
                
                selected_prompts = []
                
                for i in range(batch_size):
                    candidates = []
                
                    for key in (
                        "prompts",
                        "full_prompts",
                        "caption_detailed",
                        "caption_long",
                        "caption_short",
                    ):
                        p = batch[key][i]
                        if is_valid_prompt(p):
                            candidates.append(p)
                
                    if len(candidates) == 0:
                        raise ValueError(f"No valid prompts found for sample {i}")
                
                    idx = torch.randint(
                        0,
                        len(candidates),
                        (1,),
                        generator=gen,
                    ).item()
                
                    selected_prompts.append(candidates[idx])

                images = batch["pixels"].to(device, dtype=weight_dtype)
                crop_coords = batch.get("crop_coords", None)
                    
                with amp_context:
                    model_output, metric = model_wrapper(
                        pixel_values=images,
                        prompts=selected_prompts,
                        full_prompt=selected_prompts,
                        crop_coords=crop_coords,
                        device=device,
                        weight_dtype=weight_dtype,
                        global_step=global_step,
                    )
                    
                    if isinstance(model_output, dict):
                        loss = model_output["loss"]
                        for k, v in model_output.items():
                            if k not in accum_logs:
                                accum_logs[k] = 0.0
                            accum_logs[k] += v.item() / args.gradient_accumulation_steps
                    else:
                        loss = model_output
                        accum_logs["loss"] = accum_logs.get("loss", 0.0) + loss.item() / args.gradient_accumulation_steps
                        accum_logs["metric"] = accum_logs.get("metric", 0.0) + metric.item() / args.gradient_accumulation_steps
                    loss = (loss + metric * 0.5) / args.gradient_accumulation_steps
                    

                scaler.scale(loss).backward()
                accum_loss += loss.item()

        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), args.max_grad_norm)
        else:
            total_norm = 0.0
            for p in model_wrapper.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            grad_norm = torch.tensor(grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        model_wrapper.module.step_ema(decay=0.999)
        global_step += 1
        progress_bar.update(1)

        if rank == 0:
            current_lr = 0.0002 # Should likely fetch this from optimizer.param_groups[0]['lr']
            logs = {"lr": current_lr, "grad_norm": grad_norm.item()}
            logs.update(accum_logs)
            
            if _has_wandb and wandb.run:
                wandb.log(logs, step=global_step)

            progress_bar.set_postfix(**logs)

        # === VALIDATION & SAVING ===
        if global_step % args.checkpointing_steps == 0 or global_step % args.validation_steps == 0 or global_step == 1:
            if world_size > 1:
                dist.barrier(device_ids=[local_rank])

        # Save
        if global_step % args.checkpointing_steps == 0:
            if rank == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)

                try:
                    if hasattr(model_wrapper, "module"):
                        unwrapped = model_wrapper.module
                    else:
                        unwrapped = model_wrapper

                    unwrapped.target_transformer.save_pretrained(
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
                    log_validation(
                        model_wrapper=model_wrapper,
                        args=args,
                        global_step=global_step,
                        device=device
                    )

            if world_size > 1:
                dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print("Training finished.")
        if _has_wandb and wandb.run:
            wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()