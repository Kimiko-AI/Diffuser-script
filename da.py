import os
import tarfile
import json
import io
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm


def parse_xml_captions(predicted: str):
    if not predicted:
        return None

    text = predicted.strip()
    text = re.sub(r"^```xml\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None

    captions = {}
    for tag in ("detailed", "long", "short"):
        node = root.find(tag)
        if node is not None and node.text:
            captions[tag] = node.text.strip()

    return captions or None


def load_caption_map(folder1_tar):
    """
    Build: id -> captions
    """
    result = {}

    with tarfile.open(folder1_tar, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue

            f = tar.extractfile(member)
            if not f:
                continue

            try:
                data = json.load(f)
            except Exception:
                continue

            original = data.get("original")
            predicted = data.get("predicted")
            if original is None or predicted is None:
                continue

            captions = parse_xml_captions(predicted)
            if captions:
                key = str(original).lstrip("0")
                result[key] = captions

    return result


def find_preview_sample(folder1, folder2):
    """
    Find and return one (shard, filename, merged_json) tuple
    without writing anything.
    """
    shards = sorted(f for f in os.listdir(folder2) if f.endswith(".tar"))

    for shard in shards:
        tar1 = os.path.join(folder1, shard)
        tar2 = os.path.join(folder2, shard)
        if not os.path.exists(tar1):
            continue

        caption_map = load_caption_map(tar1)
        if not caption_map:
            continue

        with tarfile.open(tar2, "r") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue

                file_id = os.path.splitext(os.path.basename(member.name))[0].lstrip("0")
                if file_id not in caption_map:
                    continue

                f = tar.extractfile(member)
                if not f:
                    continue

                data = json.load(f)
                caps = caption_map[file_id]

                merged = dict(data)
                if "detailed" in caps:
                    merged["caption_detailed"] = caps["detailed"]
                if "long" in caps:
                    merged["caption_long"] = caps["long"]
                if "short" in caps:
                    merged["caption_short"] = caps["short"]

                return shard, member.name, merged

    return None


def merge_all(folder1, folder2):
    shards = sorted(f for f in os.listdir(folder2) if f.endswith(".tar"))

    for shard in tqdm(shards, desc="Merging shards", unit="shard"):
        tar1 = os.path.join(folder1, shard)
        tar2 = os.path.join(folder2, shard)
        if not os.path.exists(tar1):
            continue

        caption_map = load_caption_map(tar1)
        if not caption_map:
            continue

        tmp = tar2 + ".tmp"

        with tarfile.open(tar2, "r") as src, tarfile.open(tmp, "w") as dst:
            for member in src:
                f = src.extractfile(member)
                if not f:
                    continue

                if member.isfile() and member.name.endswith(".json"):
                    file_id = os.path.splitext(
                        os.path.basename(member.name)
                    )[0].lstrip("0")

                    try:
                        data = json.load(f)
                    except Exception:
                        dst.addfile(member, f)
                        continue

                    caps = caption_map.get(file_id)
                    if caps:
                        if "detailed" in caps:
                            data["caption_detailed"] = caps["detailed"]
                        if "long" in caps:
                            data["caption_long"] = caps["long"]
                        if "short" in caps:
                            data["caption_short"] = caps["short"]

                    out = json.dumps(data, ensure_ascii=False).encode("utf-8")
                    info = tarfile.TarInfo(member.name)
                    info.size = len(out)
                    dst.addfile(info, io.BytesIO(out))
                else:
                    dst.addfile(member, f)

        os.replace(tmp, tar2)


if __name__ == "__main__":
    folder_1 = "/workspace/shinon/t2i/captions"
    folder_2 = "/workspace/shinon/t2i/anime/train"

    preview = find_preview_sample(folder_1, folder_2)
    if preview is None:
        raise SystemExit("No mergeable samples found.")

    shard, filename, merged_json = preview
    print("\n" + "=" * 80)
    print("PREVIEW (NO FILES MODIFIED)")
    print(f"Shard: {shard}")
    print(f"File : {filename}")
    print(json.dumps(merged_json, indent=2, ensure_ascii=False))
    print("=" * 80)

    resp = input("Proceed with overwriting folder 2 shards? [y/N]: ").strip().lower()
    if resp != "y":
        raise SystemExit("Aborted. No files were modified.")

    merge_all(folder_1, folder_2)
    print("Merge completed successfully.")
