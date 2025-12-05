import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    Lumina2Transformer2DModel,
)
from transformers import AutoTokenizer, Gemma2Model

def load_models(args):
    # Load scheduler
    if args.pretrained_model_name_or_path:
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
        )
    else:
        # Default scheduler config if scratch
        noise_scheduler = FlowMatchEulerDiscreteScheduler() # Use default config

    # Load Text Encoder & Tokenizer (Usually pretrained even if model is scratch)
    # Unless we are pretraining text encoder too, which is rare for this context. 
    # We assume text encoder is always pretrained or provided via path.
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_encoder_path or args.pretrained_model_name_or_path,
        subfolder="tokenizer", revision=args.revision
    )
    text_encoder = Gemma2Model.from_pretrained(
        args.text_encoder_path or args.pretrained_model_name_or_path, 
        subfolder="text_encoder", revision=args.revision, variant=args.variant
    )

    # Load VAE (Usually frozen/pretrained)
    vae = AutoencoderKL.from_pretrained(
        args.vae_path or args.pretrained_model_name_or_path,
        subfolder="vae", revision=args.revision, variant=args.variant
    )

    # Load Transformer
    if args.model_config:
        print(f"Initializing Lumina2Transformer2DModel from config: {args.model_config}")
        # Filter out non-init args if any, usually config dict matches init
        # The config provided has keys like "_class_name" which we should ignore
        config = {k: v for k, v in args.model_config.items() if not k.startswith("_")}
        transformer = Lumina2Transformer2DModel(**config)
    elif args.pretrained_model_name_or_path:
        transformer = Lumina2Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", 
            revision=args.revision, variant=args.variant
        )
    else:
        raise ValueError("Must provide either --pretrained_model_name_or_path or --model_config")

    return noise_scheduler, tokenizer, text_encoder, vae, transformer
