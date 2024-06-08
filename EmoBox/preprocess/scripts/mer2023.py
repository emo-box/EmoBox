import os
import re
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "mer2023"
SAMPLE_RATE = 44100


def process_m3ed(
    dataset_path, output_base_dir="data/mer2023", output_format: str | list = "jsonl"
):

    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    
    file_set = set()
    total_fail_files = 0
    all_files = []
    
    train_label = pd.read_csv(os.path.join(dataset_path, "mer2023train", "train-label.csv"))
    test1_label = pd.read_csv(os.path.join(dataset_path, "test-labels", "test1-label.csv"))
    test2_label = pd.read_csv(os.path.join(dataset_path, "test-labels", "test2-label.csv"))
    test3_label = pd.read_csv(os.path.join(dataset_path, "test-labels", "test3-label.csv"))
    
    for idx, row in train_label.iterrows():
        if row["name"] in file_set:
            print(f"File {row['name']} already exists")
            continue
        file_path = os.path.join(dataset_path, "mer2023train", "train", row["name"]+".avi")
        all_files.append((file_path, row["discrete"], "train"))
        file_set.add(row["name"])
    
    for idx, row in test1_label.iterrows():
        if row["name"] in file_set:
            print(f"File {row['name']} already exists")
            continue
        file_path = os.path.join(dataset_path, "test1", row["name"]+".avi")
        all_files.append((file_path, row["discrete"], "test1"))
        file_set.add(row["name"])
        
    for idx, row in test2_label.iterrows():
        if row["name"] in file_set:
            print(f"File {row['name']} already exists")
            continue
        file_path = os.path.join(dataset_path, "test2", row["name"]+".avi")
        all_files.append((file_path, row["discrete"], "test2"))
        file_set.add(row["name"])
    
    for idx, row in test3_label.iterrows():
        if row["name"] in file_set:
            print(f"File {row['name']} already exists")
            continue
        file_path = os.path.join(dataset_path, "mer2023train", "test3", row["name"]+".avi")
        all_files.append((file_path, row["discrete"], "test3"))
        file_set.add(row["name"])
    
    

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
        # for split in ["train", "test1", "test2", "test3"]:
        #     split_data = {sid: d for sid, d in data.items() if d["split"] == split}
        #     split_jsonl_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_{split}_fold_1.jsonl")
        #     split_json_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_{split}_fold_1.json")
        #     write_jsonl(split_data, split_jsonl_file_path, DATASET_NAME)
        #     write_json(split_data, split_json_file_path, DATASET_NAME)
        
        train_data = {sid: d for sid, d in data.items() if d["split"] == "train"}
        test_data = {sid: d for sid, d in data.items() if d["split"] == "test1"}
        
        train_jsonl_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_train_fold_1.jsonl")
        train_json_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_train_fold_1.json")
        write_jsonl(train_data, train_jsonl_file_path, DATASET_NAME)
        write_json(train_data, train_json_file_path, DATASET_NAME)
        
        test_jsonl_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_test_fold_1.jsonl")
        test_json_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_test_fold_1.json")
        write_jsonl(test_data, test_jsonl_file_path, DATASET_NAME)
        write_json(test_data, test_json_file_path, DATASET_NAME)
        
            
    print(f"Total fail files: {total_fail_files}")
    print(f"Emotion frequency: {emotion_freq}")
    
if __name__ == "__main__":
    process_m3ed(
        "downloads/mer2023", output_format=["mini_format", "jsonl", "json", "split"]
    )
