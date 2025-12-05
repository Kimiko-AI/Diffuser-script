# ZImage Training Script

This repository contains a training script for ZImage/Lumina2 based models using Diffusers and WebDataset.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install a specific version of PyTorch compatible with your CUDA version.*

2.  **Configuration:**
    Edit `config/config.yaml` to match your environment.
    
    **Key configurations to change:**
    *   `data_url`: Path to your WebDataset tar files (e.g., `C:/data/shards/{00000..00010}.tar` or a generic URL).
    *   `output_dir`: Directory where checkpoints and logs will be saved.
    *   `pretrained_model_name_or_path`: If fine-tuning, set this. If training from scratch, leave `null` and ensure `model_config` is set.

## Training

Run the training script using `accelerate`:

```bash
accelerate launch train.py --config config/config.yaml
```

## Features

*   **Dynamic Bucketing:** The data loader automatically groups images of similar aspect ratios into buckets to minimize padding and maximize efficiency. You can configure the base resolution and bucket step size in `config.yaml`.
*   **WebDataset Support:** Efficient streaming of large datasets.
*   **Configurable Model:** Supports training from scratch or fine-tuning, defined via YAML.
*   **Mixed Precision:** Supports fp16 and bf16 (configured via `accelerate config`).

## Directory Structure

*   `train.py`: Main training entry point.
*   `trainer/`: Contains model wrappers, dataset logic, and utilities.
*   `config/`: Configuration files.
