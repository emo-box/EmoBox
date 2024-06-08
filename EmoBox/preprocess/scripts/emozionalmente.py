import os
from collections import defaultdict
import sys
from pandas import read_csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "emozionalmente"
#SAMPLE_RATE = 16000 or 44100

def process_emozionalmente(
    dataset_path, output_base_dir="data/emozionalmente", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if file.lower().endswith(".wav")
    ]
    
    meta_data = read_csv(os.path.join(dataset_path, "metadata", "samples.csv"))
    spk_data = read_csv(os.path.join(dataset_path, "metadata", "users.csv"))
    

    # Processing files with a progress bar
    for file_path in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
        waveform, sample_rate = load_audio(file_path)
        num_frame = waveform.size(1)

        # if sample_rate != SAMPLE_RATE:
        #     print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")

        file = file_path.split('/')[-1]
        audio_info = meta_data.loc[meta_data["file_name"] == file]
        spk_info = spk_data.loc[spk_data["username"] == audio_info["actor"].values[0]]
        parsed_info = {
            "id":file.replace('.wav', ''),
            "lang": "it",
            "spk": audio_info["actor"].values[0],
            "transcript": audio_info["sentence"].values[0],
            "emotion":audio_info["emotion_expressed"].values[0],
            "gd": spk_info["gender"].values[0],
            "age": spk_info["age"].values[0],
        }
            
        sid = f"{DATASET_NAME}-{file.replace('.wav', '').replace('_', '-')}"
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
    process_emozionalmente(
        "downloads/emozionalmente", output_format=["mini_format", "jsonl", "json", "split"]
    )