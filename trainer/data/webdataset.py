import webdataset as wds
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from functools import partial


def generate_buckets(base_width, base_height, step_size=32):
    # Generate bucket sizes based on the formula (BASE_WIDTH - STEP_SIZE * n, BASE_HEIGHT + STEP_SIZE * n)
    # where n ranges from -4 to 4 (inclusive).
    n_range = range(-4, 5)  # n from -4 to 4 inclusive

    bucket_sizes = []
    for n in n_range:
        width = base_width - (step_size * n)
        height = base_height + (step_size * n)
        if width > 0 and height > 0:  # Ensure dimensions are positive
            bucket_sizes.append((width, height))

    # Sort the bucket sizes
    bucket_sizes = sorted(list(set(bucket_sizes)))

    # Pre-calculate ratios for faster lookups
    bucket_ratios = np.array([w / h for w, h in bucket_sizes])

    return bucket_sizes, bucket_ratios


def assign_bucket_index(width, height, bucket_ratios):
    ratio = width / height
    idx = (np.abs(bucket_ratios - ratio)).argmin()
    return idx


def resize_to_bucket(image, bucket_idx, bucket_sizes):
    target_w, target_h = bucket_sizes[bucket_idx]
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
    full_prompt = ""

    if "json" in sample:
        json_data = sample["json"]
        # User specific tag logic from previous file
        if isinstance(json_data, dict):
            parts = []
            full_parts = []

            # Check for new structure (Pixiv/Tagger format)
            if "tags" in json_data and isinstance(json_data["tags"], list):
                all_general = []
                all_character = []
                all_rating = []

                for tag_entry in json_data["tags"]:
                    if "tags" in tag_entry and isinstance(tag_entry["tags"], dict):
                        t = tag_entry["tags"]

                        def extract_names(tag_list):
                            if isinstance(tag_list, list):
                                return [str(item["name"]) for item in tag_list if
                                        isinstance(item, dict) and "name" in item]
                            return []

                        all_general.extend(extract_names(t.get("general", [])))
                        all_character.extend(extract_names(t.get("character", [])))
                        all_rating.extend(extract_names(t.get("rating", [])))

                # Full Prompt Construction
                full_parts.extend(all_rating)
                full_content_tags = all_character + all_general
                np.random.shuffle(full_content_tags)
                full_parts.extend(full_content_tags)
                full_prompt = " ".join(full_parts)[:512]

                # Dropped Prompt Construction
                parts.extend(all_rating)

                # 1. Aggressive: 40% chance to drop ALL character tags
                if all_character and np.random.random() < 0.4:
                    all_character = []

                parts.extend(all_character)

                # 2. Aggressive: General tags processing
                np.random.shuffle(all_general)
                # Keep fewer tags: random between 1 and len
                if len(all_general) > 1:
                    keep_count = np.random.randint(1, len(all_general) + 1)
                    all_general = all_general[:keep_count]

                parts.extend(all_general)

            # Fallback to old structure
            else:
                rating = json_data.get("rating", [])
                character_tags = json_data.get("character_tags", [])
                general_tags = json_data.get("general_tags", [])

                # Helper to process tag lists
                def process_tags(tags):
                    if isinstance(tags, list):
                        return [str(t) for t in tags]
                    return [str(tags)]

                # Add rating (usually kept at start)
                parts.extend(process_tags(rating))
                full_parts.extend(process_tags(rating))

                char_parts = process_tags(character_tags)
                gen_parts = process_tags(general_tags)

                # Full Prompt
                all_tags_full = char_parts + gen_parts
                np.random.shuffle(all_tags_full)
                full_parts.extend(all_tags_full)
                full_prompt = " ".join(full_parts)[:512]

                # Dropped Prompt
                # Aggressive drop logic
                if char_parts and np.random.random() < 0.4:
                    char_parts = []

                parts.extend(char_parts)

                np.random.shuffle(gen_parts)
                if len(gen_parts) > 1:
                    keep_count = np.random.randint(1, len(gen_parts) + 1)
                    gen_parts = gen_parts[:keep_count]

                parts.extend(gen_parts)

            prompt = " ".join(parts)[:512]
        else:
            prompt = str(json_data)
            full_prompt = prompt
    elif "txt" in sample:
        prompt = sample["txt"]
        full_prompt = prompt
    elif "caption" in sample:
        prompt = sample["caption"]
        full_prompt = prompt
    return {
        "image": image,
        "prompts": prompt,
        "full_prompts": full_prompt,
        "key": sample.get("__key__", "unknown")
    }


# --- 2. The Bucket Batcher ---
def bucket_batcher(data_stream, batch_size=1, bucket_sizes=None, bucket_ratios=None):
    if bucket_sizes is None or bucket_ratios is None:
        raise ValueError("Bucket configuration (sizes and ratios) must be provided to bucket_batcher.")

    buckets = [[] for _ in bucket_sizes]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    for sample in data_stream:
        try:
            image = sample["image"]
            w, h = image.size
            b_idx = assign_bucket_index(w, h, bucket_ratios)

            image_resized = resize_to_bucket(image, b_idx, bucket_sizes)
            image_tensor = to_tensor(image_resized)

            buckets[b_idx].append({
                "pixels": image_tensor,
                "prompts": sample["prompts"],
                "full_prompts": sample["full_prompts"]
            })

            if len(buckets[b_idx]) >= batch_size:
                batch = buckets[b_idx]
                yield {
                    "pixels": torch.stack([x["pixels"] for x in batch]),
                    "prompts": [x["prompts"] for x in batch],
                    "full_prompts": [x["full_prompts"] for x in batch]
                }
                buckets[b_idx] = []

        except Exception as e:
            continue


# --- 3. The Pipeline Builder ---
def get_wds_loader(url_pattern, batch_size, num_workers=4, is_train=True, base_resolution=256, bucket_step_size=32):
    # Determine base width and height
    if isinstance(base_resolution, (list, tuple)):
        base_w, base_h = base_resolution
    else:
        base_w, base_h = base_resolution, base_resolution

    bucket_sizes, bucket_ratios = generate_buckets(base_w, base_h, bucket_step_size)

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

    # Pass the generated bucket config to the batcher
    dataset = dataset.compose(
        partial(bucket_batcher, batch_size=batch_size, bucket_sizes=bucket_sizes, bucket_ratios=bucket_ratios)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return loader