# Diffuser-script

A flexible training framework for Diffusion Transformers (Z-Image, Sana, DecoDiT).

## Quick Start

1. **Install:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure:**
   Edit `config/config.yaml` or `config/decodit_xl.yaml`. Set `data_url` to your WebDataset shards.

3. **Train:**
   ```bash
   torchrun --nproc_per_node=GPU_COUNT train.py --config config/config.yaml
   ```

## Features

- **Architectures:** Support for Z-Image, Sana, and DecoDiT.
- **Data Loading:** WebDataset with dynamic bucketing or "Fast Mode" random crops.
- **DecoDiT:** Pixel-space coordinate-aware training with (x, y, w, h) conditioning.
- **Optimization:** Mixed precision (bf16/fp16) and gradient norm logging.
