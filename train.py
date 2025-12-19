#!/usr/bin/env python
import argparse
import yaml
import os
import shutil
import torch
import torch.nn.functional as F
import gc
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
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
    parser.add_argument("--config", type=str, default=args.config)
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
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)

    # Use bf16 for text encoder if mixed precision is bf16, otherwise float32
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model_wrapper.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )

    # Dataset
    dataloader = get_dataloader(args)

    # Scheduler
    lr_scheduler = get_scheduler(
        getattr(args, "lr_scheduler_type", "constant"),
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    # Prepare with Accelerator
    model_wrapper, optimizer, lr_scheduler = accelerator.prepare(
        model_wrapper, optimizer, lr_scheduler
    )

    # === RESUME LOGIC ===
    global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
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
            load_path = os.path.join(args.output_dir, path)
            accelerator.load_state(load_path)

            try:
                global_step = int(path.split("-")[1])
                accelerator.print(f"Resumed at global step {global_step}")
            except ValueError:
                accelerator.print("Could not determine global step from checkpoint name. Starting step count at 0.")

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

            loss = model_wrapper(
                pixel_values=images,
                prompts=prompts,
                device=accelerator.device,
                weight_dtype=weight_dtype
            )

            accelerator.backward(loss)
            # if accelerator.sync_gradients:
            #    params_to_clip = model_wrapper.module.parameters() if hasattr(model_wrapper, "module") else model_wrapper.parameters()
            #    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)

            # === SAVING CHECKPOINTS ===
            if global_step % args.checkpointing_steps == 0:

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                try:
                    # Warning: This is safe for DDP.
                    # If using DeepSpeed/FSDP, this line will hang because all ranks must call save_state.
                    accelerator.save_state(save_path)
                    unwrapped_model = accelerator.unwrap_model(model_wrapper)
                    unwrapped_model.transformer.save_pretrained(
                        save_path,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    save_model_card(
                        repo_id=f"lumina2-step-{global_step}",
                        base_model=args.pretrained_model_name_or_path or "scratch",
                        repo_folder=save_path
                    )
                    if accelerator.is_main_process:
                        # Rotation Logic
                        if getattr(args, "checkpoints_total_limit", None) is not None:
                            limit = int(args.checkpoints_total_limit)
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint-")]
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
                                logger.warning(f"Error during checkpoint rotation: {e}")

                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {global_step}: {e}")

            # === VALIDATION ===
            if global_step % args.validation_steps == 0:
                with torch.no_grad():
                    log_validation(
                        accelerator=accelerator,
                        model_wrapper=model_wrapper,
                        args=args,
                        global_step=global_step
                    )

                torch.cuda.empty_cache()
                gc.collect()

                # Sync everyone after validation ends
                accelerator.wait_for_everyone()

        logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps: break

    accelerator.end_training()


if __name__ == "__main__":
    main()