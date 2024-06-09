import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "polish"
SAMPLE_RATE = 44100


def parse_filename(filename):
    # Remove the file extension and any leading/trailing slashes
    name_without_extension = filename.strip('/').rstrip('.wav')

    # Split the filename by underscores to extract the parts
    parts = name_without_extension.split('_')

    # Extract the components from the filename
    speaker,emotion_code = parts[0].split('.')
    word = parts[1]
    repetition = parts[2]

    # Map the emotion code to the emotion
    emotion = emotion_code

    # Construct the dictionary
    parsed_info = {
        "id": name_without_extension,
        'spk': speaker,
        'emotion': emotion,
        'transcript': word,
        'Repetition': repetition
    }

    return parsed_info


def process_polish(
    dataset_path, output_base_dir="data/polish", output_format: str | list = "jsonl"
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
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        parsed_info = parse_filename(os.path.basename(file_path))
        sid = f"{DATASET_NAME}-{parsed_info['id']}"

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

    if "mini_format" in output_format:
        write_mini_format(data, output_base_dir)

    if "jsonl" in output_format:
        write_jsonl(
            data, os.path.join(output_base_dir, f"{DATASET_NAME}.jsonl"), DATASET_NAME
        )

    if "json" in output_format:
        write_json(
            data, os.path.join(output_base_dir, f"{DATASET_NAME}.json"), DATASET_NAME
        )

    if "split" in output_format:
        write_folds(data, output_base_dir, DATASET_NAME)

    print(f"Emotion frequency: {emotion_freq}")


if __name__ == "__main__":
    process_polish(
        "downloads/polish", output_format=["mini_format", "jsonl", "json", "split"]
    )
