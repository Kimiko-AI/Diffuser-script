import torch
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_sana_components(args, device=None, weight_dtype=torch.float32):
    """
    Loads components specifically for Sana model.
    """
    device_map = None
    if device:
        # device_map expects a string 'cuda:0' etc. or a dict.
        # If device is a torch.device, str(device) works.
        device_map = {"": str(device)}

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
    )

    text_encoder = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        device_map=device_map,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True if device_map else False
    )

    # Load VAE (AutoencoderDC)
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        device_map=device_map,
        torch_dtype=torch.float32, # Usually kept in fp32
        low_cpu_mem_usage=True if device_map else False
    )

    # Load Transformer (SanaTransformer2DModel)
    model_config = getattr(args, "model_config", None)
    if model_config:
        # If training from scratch with config
        print(f"Initializing SanaTransformer2DModel from config: {model_config}")
        config = {k: v for k, v in model_config.items() if not k.startswith("_")}
        transformer = SanaTransformer2DModel(**config)
    else:
        transformer = SanaTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="transformer", 
            low_cpu_mem_usage=True,
            device_map=device_map,
            torch_dtype=torch.float32
        )

    return noise_scheduler, tokenizer, text_encoder, vae, transformer
