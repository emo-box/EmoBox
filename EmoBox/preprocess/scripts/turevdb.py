import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "turevdb"
SAMPLE_RATE = 44100
    
def process_turevdb(
    dataset_path, output_base_dir="data/turevdb", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    all_files = []
    
    for emotion in os.listdir(os.path.join(dataset_path, "Sound Source")):
        for file in os.listdir(os.path.join(dataset_path, "Sound Source", emotion)):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(dataset_path, "Sound Source", emotion, file)
                all_files.append((file_path, emotion, file.split("_")[0]))
    

    # Processing files with a progress bar
    for file_path, emotion, spk in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
        if sample_rate != SAMPLE_RATE:
            print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")

        sid = f"{DATASET_NAME}-{file_path.split('/')[-1].replace('.webm', '')}"
        data[sid] = {
            "audio": file_path,
            "emotion": emotion,
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": spk,
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
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
        write_folds(data, output_base_dir, DATASET_NAME)

    print(f"Emotion frequency: {emotion_freq}")


if __name__ == "__main__":
    process_turevdb(
        "downloads/turevdb", output_format=["mini_format", "jsonl", "json", "split"]
    )