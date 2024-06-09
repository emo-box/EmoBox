import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

SAMPLE_RATE = 16000
DATASET_NAME = "resd"



def process_resd(
    dataset_path, output_base_dir="data/resd", output_format: str | list = "jsonl"
):
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)

    data = {}
    emotion_freq = defaultdict(int)

    all_files = []
    
    train = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    test = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    
    for idx, row in train.iterrows():
        file_path = os.path.join(dataset_path, "train", row["path"])
        all_files.append((file_path, row["emotion"], "train"))
        
    for idx, row in test.iterrows():
        file_path = os.path.join(dataset_path, "test", row["path"])
        all_files.append((file_path, row["emotion"], "test"))

    data = {}
    emotion_freq = defaultdict(int)
    
    for file_path, emotion, split in tqdm(all_files):
        if not os.path.exists(file_path):
            file_path = file_path.replace('.avi', '.mp4')
        
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            total_fail_files += 1
            print(f"Error processing file {file_path}: {e}")
            continue
            
        sid = f"{DATASET_NAME}-{file_path.split('/')[-1].replace('.avi', '')}"
        data[sid] = {
            "audio": file_path,
            "emotion": emotion,
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": file_path.split('/')[-1].replace('.avi', ''),
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
            "split": split,
        }
        emotion_freq[emotion] += 1
    

    if output_format == "mini_format" or "mini_format" in output_format:
        write_mini_format(data, output_base_dir)

    if output_format == "jsonl" or "jsonl" in output_format:
        # Handle single-file JSON or JSONL output for the entire dataset
        jsonl_file_path = os.path.join(output_base_dir, f"{DATASET_NAME}.jsonl")
        write_jsonl(data, jsonl_file_path, DATASET_NAME)

    if output_format == "json" or "json" in output_format:
        # Handle single-file JSON output for the entire dataset
        json_file_path = os.path.join(output_base_dir, f"{DATASET_NAME}.json")
        write_json(data, json_file_path, DATASET_NAME)

    if output_format == "split" or "split" in output_format:
        os.makedirs(os.path.join(output_base_dir, "fold_1"), exist_ok=True)
        # Handle split JSONL output
        for split in ["train", "test"]:
            split_data = {sid: d for sid, d in data.items() if d["split"] == split}
            split_jsonl_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_{split}_fold_1.jsonl")
            split_json_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_{split}_fold_1.json")
            write_jsonl(split_data, split_jsonl_file_path, DATASET_NAME)
            write_json(split_data, split_json_file_path, DATASET_NAME)
            
    print(f"Emotion frequency: {emotion_freq}")


if __name__ == "__main__":
    process_resd(
        "downloads/resd", output_format=["mini_format", "jsonl", "json", "split"]
    )
