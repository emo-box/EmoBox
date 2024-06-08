import os
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *

DATASET_NAME = "ased"
SAMPLE_RATE = 16000


def parse_filename(filename):
    # Remove file extension
    name = filename.split(".")[0]

    # Mapping for emotions
    emotions = {"n": "Neutral", "f": "Fear", "h": "Happy", "s": "Sad", "a": "Angry"}

    # Split the filename
    parts = filename.split("-")

    # Extracting information
    emotion_code = parts[0][0]  # First character of the first part
    folder_number = parts[0][1]  # Second character of the first part
    sentence_number = parts[1]
    repetition_number = parts[2]
    gender_code = parts[3]
    speaker_number = parts[4].split(".")[0]  # Remove file extension

    # Mapping codes to meaningful information
    emotion = emotion_code
    gender = "Female" if gender_code == "01" else "Male"

    # Construct the dictionary
    parsed_info = {
        "id": name,
        "gd": gender,
        "lang": "am",
        "spk": speaker_number,
        "gd": gender,
        "emotion": emotion,
    }
    return parsed_info


def process_ased(
    dataset_path, output_base_dir="data/ased", output_format: str | list = "jsonl"
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

    # Processing files with a progress bar
    for file_path in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
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
            "spk": parsed_info["spk"],
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
            "lang": parsed_info["lang"],
        }
        emotion_freq[parsed_info["emotion"]] += 1

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
    process_ased(
        "downloads/ased", output_format=["mini_format", "jsonl", "json", "split"]
    )
