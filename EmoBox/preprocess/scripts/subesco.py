import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "subesco"
# SAMPLE_RATE = 44100 or 48000


def parse_filename(filename):
    # Remove file extension and get the base name
    base_name = filename.split('.')[0].lower()

    # Split the filename by underscores
    parts = base_name.split('_')

    # Construct the dictionary
    gender = 'Female' if parts[0] == 'f' else 'Male'
    parsed_info = {
        'gd': gender,
        'lang': 'bn',
        'spk': f"{gender}-{parts[1]}",
        'spk_name': parts[2],
        'transcript_id': f"{parts[3]}_{parts[4]}",
        'emotion': parts[5],
        'take_number': parts[6]
    }

    return parsed_info
    
def process_subesco(
    dataset_path, output_base_dir="data/subesco", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    speaker_set = set()
    
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if file.lower().endswith(".wav")
    ]
    
    # Processing files with a progress bar
    for file_path in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
        # if sample_rate != SAMPLE_RATE:
        #     print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")

        parsed_info = parse_filename(os.path.basename(file_path))
        sid = f"{DATASET_NAME}-{file_path.split('/')[-1].replace('.wav', '')}"
        data[sid] = {
            "audio": file_path,
            "emotion": parsed_info["emotion"],
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": parsed_info["spk"],
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
        }
        emotion_freq[parsed_info["emotion"]] += 1
        speaker_set.add(parsed_info["spk"])
        
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
        write_folds(data, output_base_dir, DATASET_NAME)

    print(f"Emotion frequency: {emotion_freq}")
    print(f"Total speakers: {len(speaker_set)}")

if __name__ == "__main__":
    process_subesco(
        "downloads/subesco", output_format=["mini_format", "jsonl", "json", "split"]
    )