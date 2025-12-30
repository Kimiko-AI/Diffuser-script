# Diffuser-script

A flexible training framework for Diffusion Transformers (Z-Image, Sana, SR-DiT).

## Quick Start

1. **Install:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure:**
   Edit `config/config.yaml` or `config/sr_dit_xl.yaml`. Set `data_url` to your WebDataset shards.

3. **Train:**
   ```bash
   torchrun --nproc_per_node=GPU_COUNT train.py --config config/config.yaml
   ```

## Features

- **Architectures:** Support for Z-Image, Sana, and SR-DiT.
- **Data Loading:** WebDataset with dynamic bucketing or "Fast Mode" random crops.
- **SR-DiT:** Coordinate-aware training with (x, y, w, h) conditioning.
- **Optimization:** Mixed precision (bf16/fp16) and gradient norm logging.