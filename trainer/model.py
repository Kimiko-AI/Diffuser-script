import torch
import importlib
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_class_from_string(class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)

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
        args.text_encoder_path or args.pretrained_model_name_or_path, subfolder = "tokenizer"
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.text_encoder_path or args.pretrained_model_name_or_path, subfolder = "text_encoder"
    )

    # Load VAE (Usually frozen/pretrained)
    vae = AutoencoderKL.from_pretrained(
        args.vae_path or args.pretrained_model_name_or_path, subfolder = "vae"
    )

    # Determine Transformer Class
    transformer_class_path = getattr(args, "transformer_class_path", "diffusers.ZImageTransformer2DModel")
    TransformerClass = get_class_from_string(transformer_class_path)

    # Load Transformer
    if args.model_config:
        print(f"Initializing {TransformerClass.__name__} from config: {args.model_config}")
        # Filter out non-init args if any, usually config dict matches init
        # The config provided has keys like "_class_name" which we should ignore
        config = {k: v for k, v in args.model_config.items() if not k.startswith("_")}
        transformer = TransformerClass(**config)
    elif args.pretrained_model_name_or_path:
        transformer = TransformerClass.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", 
            revision=args.revision, variant=args.variant
        )
    else:
        raise ValueError("Must provide either --pretrained_model_name_or_path or --model_config")

    return noise_scheduler, tokenizer, text_encoder, vae, transformer
