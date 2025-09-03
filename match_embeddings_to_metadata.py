import os
import json
import numpy as np


embedding_folder = "embeddings/audio_midi_cqt5_ps_v5_after_attention"
metadata_json_path = "new_clean_data.json"  
with open(metadata_json_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata_keys = {key.lower().replace("_", " ").replace("-", " "): key for key in metadata.keys()}

def clean_string(name):
    return (
        name.lower()
        .replace("_post.npy", "")
        .replace("_", " ")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )

metadata_keys = {clean_string(key): key for key in metadata.keys()}

matched_data = []

for filename in os.listdir(embedding_folder):
    if not filename.endswith("_post.npy"):
        continue
    file_path = os.path.join(embedding_folder, filename)
    cleaned_name = clean_string(filename.replace("_post.npy", ""))

    matched_key = metadata_keys.get(cleaned_name)

    if not matched_key:
        for key in metadata_keys:
            if cleaned_name in key or key in cleaned_name:
                matched_key = metadata_keys[key]
                break

    if matched_key:
        embedding = np.load(file_path)
        matched_data.append({
            "filename": filename,
            "embedding_shape": embedding.shape,
            "metadata": metadata[matched_key]
        })
    else:
        print(f"[!] No match found for: {filename}")

metadata_keys = {clean_string(key): key for key in metadata.keys()}

matched_data = []

for filename in os.listdir(embedding_folder):
    if not filename.endswith("_post.npy"):
        continue
    file_path = os.path.join(embedding_folder, filename)
    cleaned_name = clean_string(filename.replace("_post.npy", ""))

    matched_key = metadata_keys.get(cleaned_name)

    if not matched_key:
        for key in metadata_keys:
            if cleaned_name in key or key in cleaned_name:
                matched_key = metadata_keys[key]
                break

    if matched_key:
        embedding = np.load(file_path)
        matched_data.append({
            "filename": filename,
            "embedding_shape": embedding.shape,
            "metadata": metadata[matched_key]
        })
    else:
        print(f"[!] No match found for: {filename}")

matched_data = []

for filename in os.listdir(embedding_folder):
    if not filename.endswith("_post.npy"):
        continue
    file_path = os.path.join(embedding_folder, filename)
    cleaned_name = clean_string(filename)

    matched_key = metadata_keys.get(cleaned_name)
    
    #fuzzy matching
    if not matched_key:
        for key in metadata_keys:
            if cleaned_name in key or key in cleaned_name:
                matched_key = metadata_keys[key]
                break

    if matched_key:
        embedding = np.load(file_path)
        matched_data.append({
            "filename": filename,
            "embedding_shape": embedding.shape,
            "metadata": metadata[matched_key]
        })
    else:
        print(f"[!] No match found for: {filename}")

print(f"\n✅ Matched {len(matched_data)} embeddings.")
print(f"First match:\n{matched_data[0] if matched_data else 'None'}")

import json
with open("matched_embeddings_after_attention_metadata.json", "w", encoding="utf-8") as f:
    json.dump(matched_data, f, indent=2, ensure_ascii=False)
