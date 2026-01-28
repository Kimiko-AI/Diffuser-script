import os
import tarfile
import json
import io
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- Updated Paths ---
FOLDER_2 = "/workspace/shinon/t2i/captions"       # Folder with {id}.webp, {id}.json
FOLDER_1 = "/workspace/shinon/t2i/anime/train/"   # Folder with {hash}.json (after renaming)
OUTPUT   = "./webdataset_merged"                 # Result folder

def parse_xml_captions(predicted_str):
    """Parses the XML block from the 'predicted' field."""
    captions = {}
    clean_xml = re.sub(r"^```xml\s*|\s*```$", "", predicted_str.strip(), flags=re.MULTILINE)
    try:
        root = ET.fromstring(clean_xml)
        for tag in ['detailed', 'long', 'short']:
            node = root.find(tag)
            captions[tag] = node.text.strip() if (node is not None and node.text) else None
    except ET.ParseError:
        return None
    return captions

def build_metadata_map(tar_path):
    """Maps original IDs from folder_2 to their parsed captions."""
    meta_map = {}
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if member.name.endswith('.json'):
                    f = tar.extractfile(member)
                    if f:
                        data = json.load(f)
                        original_id = data.get('original')
                        predicted_raw = data.get('predicted')
                        if original_id and predicted_raw:
                            parsed = parse_xml_captions(predicted_raw)
                            if parsed:
                                meta_map[str(original_id)] = parsed
    except Exception:
        pass
    return meta_map

def run_merge():
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    tars = sorted([f for f in os.listdir(FOLDER_1) if f.endswith('.tar')])
    
    # --- STEP 1: PREVIEW CHECK ---
    print("--- Running Preview Check ---")
    test_tar = tars[0]
    p1, p2 = os.path.join(FOLDER_1, test_tar), os.path.join(FOLDER_2, test_tar)
    
    if os.path.exists(p2):
        test_map = build_metadata_map(p2)
        sample_id = next(iter(test_map))
        print(f"Sample Found: ID {sample_id} in {test_tar}")
        print(f"Parsed Captions Preview: {json.dumps(test_map[sample_id], indent=2)}")
        
        confirm = input("\nDoes the parsed data look correct? Type 'yes' to start full merge: ")
        if confirm.lower() != 'yes':
            print("Merge cancelled.")
            return
    else:
        print(f"Warning: Could not find matching {test_tar} in Folder 2 for preview.")

    # --- STEP 2: FULL MERGE ---
    for tar_name in tqdm(tars, desc="Shards"):
        path1 = os.path.join(FOLDER_1, tar_name)
        path2 = os.path.join(FOLDER_2, tar_name)
        out_path = os.path.join(OUTPUT, tar_name)

        if not os.path.exists(path2):
            continue

        id_map = build_metadata_map(path2)

        with tarfile.open(path1, 'r') as src, tarfile.open(out_path, 'w') as dst:
            for member in src:
                f_obj = src.extractfile(member)
                if f_obj is None: continue

                if member.name.endswith('.json'):
                    file_id = os.path.splitext(os.path.basename(member.name))[0]
                    data = json.load(f_obj)
                    
                    if file_id in id_map:
                        new_caps = id_map[file_id]
                        data.update({
                            'caption_detailed': new_caps['detailed'],
                            'caption_long': new_caps['long'],
                            'caption_short': new_caps['short']
                        })
                    
                    # Save merged JSON
                    new_bytes = json.dumps(data, indent=2).encode('utf-8')
                    new_info = tarfile.TarInfo(name=member.name)
                    new_info.size = len(new_bytes)
                    dst.addfile(new_info, io.BytesIO(new_bytes))
                else:
                    # Save Image (.webp)
                    dst.addfile(member, f_obj)

if __name__ == "__main__":
    run_merge()