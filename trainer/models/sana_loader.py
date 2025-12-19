import torch
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_sana_components(args):
    """
    Loads components specifically for Sana model.
    """
    # Load scheduler
    if args.pretrained_model_name_or_path:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler",
        )
    else:
        noise_scheduler = FlowMatchEulerDiscreteScheduler()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        revision=args.revision, 
        variant=args.variant
    )

    # Load VAE (AutoencoderDC)
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    # Load Transformer (SanaTransformer2DModel)
    if args.model_config:
        # If training from scratch with config
        print(f"Initializing SanaTransformer2DModel from config: {args.model_config}")
        config = {k: v for k, v in args.model_config.items() if not k.startswith("_")}
        transformer = SanaTransformer2DModel(**config)
    else:
        transformer = SanaTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="transformer", 
            revision=args.revision, 
            variant=args.variant
        )

    return noise_scheduler, tokenizer, text_encoder, vae, transformer
