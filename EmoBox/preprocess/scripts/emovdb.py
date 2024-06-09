import os
import re
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "emovdb"
# SAMPLE_RATE = 16000 and 44100

def process_emovdb(
    dataset_path, output_base_dir="data/emovdb", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(dataset_path))
        for file in files
        if file.lower().endswith(".wav")
    ]
    
    pbar = tqdm(desc=f"Processing {DATASET_NAME} files", total=len(all_files))

    # Processing files with a progress bar
    for folder in os.listdir(dataset_path):
        # check is a folder
        if not os.path.isdir(os.path.join(dataset_path, folder)):
            continue
        spk = folder.split('_')[0]
        emotion = folder.split('_')[1]
        emotion = emotion if emotion != 'Disgusted' else 'Disgust'
        for file in os.listdir(os.path.join(dataset_path, folder)):
            if file.endswith(".wav"):
                file_path = os.path.join(dataset_path, folder, file)
                text_id = file.split("_")[-1].replace('.wav', '')
                waveform, sample_rate = load_audio(file_path)
                if waveform is None:
                    continue
                # if sample_rate != SAMPLE_RATE:
                #     print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")

                sid = f"{DATASET_NAME}-{spk}-{emotion}-{text_id}"
                data[sid] = {
                    # relative path from dataset_path
                    "audio": file_path,
                    "emotion": emotion,
                    "channel": 1,
                    "sid": sid,
                    "sample_rate": sample_rate,
                    "num_frame": waveform.size(1),
                    "spk": spk,
                    "start_time": 0,
                    "end_time": waveform.size(1) / sample_rate,
                    "duration": waveform.size(1) / sample_rate,
                }
                pbar.update(1)
                emotion_freq[emotion] += 1
    pbar.close()
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
    process_emovdb(
        "downloads/emovdb", output_format=["mini_format", "jsonl", "json", "split"]
    )
