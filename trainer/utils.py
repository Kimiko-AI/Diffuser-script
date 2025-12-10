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


def log_validation(accelerator, model_wrapper, args, global_step, ema_model=None):
    validation_prompts = args.validation_prompt
    if isinstance(validation_prompts, str):
        validation_prompts = [validation_prompts]

    logger.info(f"Running validation... Prompts: {validation_prompts}")

    # Unwrap the wrapper if it's wrapped by accelerator (DDP)
    unwrapped_wrapper = accelerator.unwrap_model(model_wrapper)

    if ema_model is not None:
        ema_model.store(unwrapped_wrapper.transformer.parameters())
        ema_model.copy_to(unwrapped_wrapper.transformer.parameters())

    all_images = []
    
    # Generate for each prompt
    for prompt_idx, prompt in enumerate(validation_prompts):
        # Run inference using the wrapper's generate method
        # Infer num_validation_images from number of prompts (1 per prompt)
        images = unwrapped_wrapper.generate(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=4.0,
            num_images=1,
            seed=args.seed,
            device=accelerator.device
        )
        
        # Collect for consolidated logging
        for img in images:
            all_images.append((prompt, img))

        # Tensorboard logging per prompt group
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(f"validation/prompt_{prompt_idx}", np_images, global_step, dataformats="NHWC")

    if ema_model is not None:
        ema_model.restore(unwrapped_wrapper.transformer.parameters())

    # Consolidated WandB logging
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {prompt}") 
                        for i, (prompt, image) in enumerate(all_images)
                    ]
                }
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [img for _, img in all_images]
