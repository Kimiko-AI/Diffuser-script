import os
import logging
import torch
import numpy as np
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils import is_wandb_available

if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)


def save_model_card(repo_id, base_model=None, repo_folder=None):
    # Simplified model card saving without validation images
    model_description = f"""
# Lumina2 Training - {repo_id}
## Model description
Trained on {base_model}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="apache-2.0",
        base_model=base_model,
        model_description=model_description,
    )
    populate_model_card(model_card, tags=["text-to-image", "lumina2"]).save(os.path.join(repo_folder, "README.md"))

@torch.no_grad()
def log_validation(model_wrapper, args, global_step, device):
    validation_prompts = args.validation_prompt
    if validation_prompts is None:
        validation_prompts = []
    elif isinstance(validation_prompts, str):
        validation_prompts = [validation_prompts]
    else:
        validation_prompts = list(validation_prompts)

    logger.info(f"Running validation... Prompts: {validation_prompts}")

    # Unwrap the wrapper if it's wrapped by DDP
    if hasattr(model_wrapper, "module"):
        unwrapped_wrapper = model_wrapper.module
    else:
        unwrapped_wrapper = model_wrapper

    all_images = []

    # 1. Conditional Generation (User prompts)
    if validation_prompts:
        images_cond, _ = unwrapped_wrapper.generate(
            prompt=validation_prompts,
            num_inference_steps=getattr(args, "validation_num_inference_steps", 20),
            guidance_scale=getattr(args, "validation_guidance_scale", 4.0),
            num_images=1,
            seed=args.seed,
            device=device
        )
        for i, img in enumerate(images_cond):
            all_images.append((validation_prompts[i], img))

    uncond_seed = args.seed + 1 if args.seed is not None else None
    
    uncond_prompts = [""] * 4
    images_uncond, _ = unwrapped_wrapper.generate(
        prompt=uncond_prompts,
        num_inference_steps=getattr(args, "validation_num_inference_steps", 20),
        guidance_scale=1.0,  # Force CFG off
        num_images=1,
        seed=uncond_seed,
        device=device
    )

    for i, img in enumerate(images_uncond):
        all_images.append((f"Unconditional_{i}", img))

    # Consolidated WandB logging
    if is_wandb_available() and wandb.run is not None:
        wandb.log(
            {
                "validation": [
                    wandb.Image(image, caption=f"{i}: {prompt}")
                    for i, (prompt, image) in enumerate(all_images)
                ]
            },
            step=global_step
        )

    return [img for _, img in all_images]
