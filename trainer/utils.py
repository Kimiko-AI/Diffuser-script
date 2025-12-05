import os
import logging
import torch
import numpy as np
from diffusers import Lumina2Pipeline
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

def log_validation(accelerator, transformer, vae, text_encoder, tokenizer, scheduler, args, global_step):
    logger.info(f"Running validation... Prompt: {args.validation_prompt}")
    
    # Create pipeline
    # We need to unwrap the transformer if it's wrapped by accelerator
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    
    pipeline = Lumina2Pipeline(
        transformer=unwrapped_transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    # Run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    
    images = []
    # Use autocast for inference to match training precision if needed, or just float32/bf16 logic
    # The pipeline usually handles mixed precision if the components are in that dtype.
    # We'll assume components are already on device and in correct dtype (except maybe VAE which is float32)
    
    with torch.no_grad():
        for _ in range(args.num_validation_images):
            img = pipeline(
                prompt=args.validation_prompt, 
                generator=generator,
                num_inference_steps=20, # Default or configurable
                guidance_scale=4.0      # Default for flow match often higher or specific
            ).images[0]
            images.append(img)
            
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, global_step, dataformats="NHWC")
        if tracker.name == "wandb":
             tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
            
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images
