import os
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *

DATASET_NAME = "asvpesd"
SAMPLE_RATE = 16000

def parse_filename(filename):
    # Define the meaning of each part of the filename
    modalities = {"03": "audio-only"}
    vocal_channels = {"01": "speech", "02": "non-speech"}
    intensity = {"01": "normal", "02": "high"}
    actor_gender = lambda x: "male" if int(x) % 2 == 0 else "female"
    age_groups = {"01": "above 65", "02": "between 20~64", "03": "under 20", "04": "baby"}
    sources = {"01": "website, youtube", "02": "website, youtube", "03": "movies"}
    languages = {"01": "zh", "02": "en", "04": "fr",}
    parts = filename.split('.')[0].split('-')
    return {
        "id": filename.replace(".wav", ''),
        "age": age_groups.get(parts[6], "Unknown"),
        "gd": actor_gender(parts[5]),
        "lang": languages.get(parts[8], "Russian and others") if len(parts) > 8 else "Russian and others",
        "modality": modalities.get(parts[0], "Unknown"),
        "vocal": vocal_channels.get(parts[1], "Unknown"),
        "emotion": parts[2],
        "intensity": intensity.get(parts[3], "Unknown"),
        "statement": parts[4],
        "actor": parts[5],
        "source": sources.get(parts[7], "Unknown"),
    }
    
def process_asvpesd(
    dataset_path, output_base_dir="data/asvpesd", output_format: str | list = "jsonl"
):

    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)

    data = {}
    emotion_freq = defaultdict(int)

    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if file.lower().endswith(".wav")
    ]

    for file_path in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
        file_path = get_relative_audio_path(file_path)
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        if sample_rate != SAMPLE_RATE:
            print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")

        parsed_info = parse_filename(os.path.basename(file_path))
        sid = f"{DATASET_NAME}-{parsed_info['id'].replace('_', '-')}"

        data[sid] = {
            "audio": file_path,
            "emotion": parsed_info["emotion"],
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": parsed_info["actor"],
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
        }
        emotion_freq[parsed_info["emotion"]] += 1
    
    if output_format == "mini_format" or "mini_format" in output_format:
        write_mini_format(data, output_base_dir)

    if output_format == "jsonl" or "jsonl" in output_format:
        jsonl_file_path = os.path.join(output_base_dir, f"{DATASET_NAME}.jsonl")
        write_jsonl(data, jsonl_file_path, DATASET_NAME)

    if output_format == "json" or "json" in output_format:
        json_file_path = os.path.join(output_base_dir, f"{DATASET_NAME}.json")
        write_json(data, json_file_path, DATASET_NAME)

    if output_format == "split" or "split" in output_format:
        write_folds(data, output_base_dir, DATASET_NAME)
    
    
if __name__ == "__main__":
    process_asvpesd("downloads/asvpesd", output_format=["mini_format", "jsonl", "json", "split"])