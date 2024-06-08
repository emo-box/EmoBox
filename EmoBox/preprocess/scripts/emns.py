import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "emns"
SAMPLE_RATE = 48000


def parse_filename(file, df):
    str = f"wavs/{file}"
    row = df.loc[df["audio_recording"] == str]

    if row.empty:
        print(f"File {file} does not exist in metadata")
        return
    f_id = row["id"].values[0]
    age = row["age"].values[0]
    gender = row["gender"].values[0]
    emotion = row["emotion"].values[0]
    level = row["level"].values[0]
    utterance = row["utterance"].values[0]
    description = row["description"].values[0]

    return {
        "id": f_id,
        "age": age,
        "gd": gender,
        "spk": 3,
        "emotion": emotion if emotion != "Surprised" else "Surprise",
        "level": level,
        "transcript": utterance,
        "description": description
    }
    
def process_emns(
    dataset_path, output_base_dir="data/emns", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(dataset_path, "cleaned_webm"))
        for file in files
        if file.lower().endswith(".webm")
    ]
    
    metadata_df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"), sep="|")

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

        parsed_info = parse_filename(os.path.basename(file_path), metadata_df)
        sid = f"{DATASET_NAME}-{file_path.split('/')[-1].replace('.webm', '')}"
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
    process_emns(
        "downloads/emns", output_format=["mini_format", "jsonl", "json", "split"]
    )