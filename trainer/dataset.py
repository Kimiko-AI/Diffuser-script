import webdataset as wds
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from functools import partial

# --- Configuration ---
BUCKET_SIZES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
]
# Pre-calculate ratios for faster lookups
BUCKET_RATIOS = np.array([w / h for w, h in BUCKET_SIZES])

def assign_bucket_index(width, height):
    ratio = width / height
    idx = (np.abs(BUCKET_RATIOS - ratio)).argmin()
    return idx

def resize_to_bucket(image, bucket_idx):
    target_w, target_h = BUCKET_SIZES[bucket_idx]
    return image.resize((target_w, target_h), Image.BICUBIC)

# --- 1. Preprocessing Function ---
def transform_sample(sample):
    # Adapting to common WDS formats (jpg/png/webp)
    # Using 'pil' decoding in WDS pipeline usually yields standard keys or 'jpg', 'png' etc.
    # We check for common image keys
    
    image = None
    for key in ["jpg", "png", "webp", "jpeg"]:
        if key in sample:
            image = sample[key]
            break
    
    if image is None:
        raise ValueError("No image found in sample")

    # Handle prompts
    # Assuming 'json' or 'txt' or 'caption'
    prompt = ""
    if "json" in sample:
        json_data = sample["json"]
        # User specific tag logic from previous file
        if isinstance(json_data, dict):
            rating = json_data.get("rating", [])
            character_tags = json_data.get("character_tags", [])
            general_tags = json_data.get("general_tags", [])
            # If lists, join them. If strings, concat.
            parts = []
            if isinstance(rating, list): parts.extend(rating)
            else: parts.append(str(rating))
            if isinstance(character_tags, list): parts.extend(character_tags)
            else: parts.append(str(character_tags))
            if isinstance(general_tags, list): parts.extend(general_tags)
            else: parts.append(str(general_tags))
            
            prompt = " ".join(map(str, parts))[:512]
        else:
            prompt = str(json_data)
    elif "txt" in sample:
        prompt = sample["txt"]
    elif "caption" in sample:
        prompt = sample["caption"]

    return {
        "image": image,
        "prompts": prompt,
        "key": sample.get("__key__", "unknown")
    }

# --- 2. The Bucket Batcher ---
def bucket_batcher(data_stream, batch_size=1):
    buckets = [[] for _ in BUCKET_SIZES]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    for sample in data_stream:
        try:
            image = sample["image"]
            w, h = image.size
            b_idx = assign_bucket_index(w, h)
            
            image_resized = resize_to_bucket(image, b_idx)
            image_tensor = to_tensor(image_resized)

            buckets[b_idx].append({
                "pixels": image_tensor,
                "prompts": sample["prompts"]
            })

            if len(buckets[b_idx]) >= batch_size:
                batch = buckets[b_idx]
                yield {
                    "pixels": torch.stack([x["pixels"] for x in batch]),
                    "prompts": [x["prompts"] for x in batch]
                }
                buckets[b_idx] = []

        except Exception as e:
            print(f"Skipping sample due to error: {e}")
            continue

# --- 3. The Pipeline Builder ---
def get_wds_loader(url_pattern, batch_size, num_workers=4, is_train=True):
    dataset = wds.WebDataset(
        url_pattern, 
        resampled=True, 
        handler=wds.warn_and_continue,
        nodesplitter=wds.split_by_node,
        shardshuffle=True
    )

    if is_train:
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.compose(wds.split_by_worker)
    dataset = dataset.decode("pil", handler=wds.warn_and_continue)
    dataset = dataset.map(transform_sample, handler=wds.warn_and_continue)
    dataset = dataset.compose(partial(bucket_batcher, batch_size=batch_size))

    loader = wds.WebLoader(
        dataset,
        batch_size=None, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return loader