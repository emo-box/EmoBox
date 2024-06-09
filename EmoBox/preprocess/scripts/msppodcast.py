import os
import re
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "msppodcast"
SAMPLE_RATE = 16000


def process_msppodcast(
    dataset_path, output_base_dir="data/msppodcast", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)

    data = {}
    emotion_freq = defaultdict(int)
    

    with open(os.path.join(dataset_path, "label", "labels_concensus.json"), "r") as f:
        label_dict = json.load(f)

    all_files = [
        (file_name, labels, dataset_path) for file_name, labels in label_dict.items()
    ]

    # Processing files with a progress bar
    for file_name, labels, dataset_path in tqdm(
        all_files, desc=f"Processing {DATASET_NAME} files"
    ):
        file_path = os.path.join(dataset_path, "Audio", file_name)
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
        if sample_rate != SAMPLE_RATE:
            print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")
        
        # transcript_path = os.path.join(dataset_path, "Transcripts", file_name.replace(".wav", ".txt"))
        # with open(transcript_path, 'r') as f:
        #     transcript = ";".join(f.readlines()).replace("\n", "")
        
        sid = f"{DATASET_NAME}-{file_name.split('.')[0]}"
        speaker_id, emotion, gender, dataset = labels['SpkrID'], labels["EmoClass"], labels["Gender"], labels["Split_Set"]
        
        data[sid] = {
            "audio": file_path,
            "emotion": emotion,
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": speaker_id,
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
            "dset": dataset,
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
        output_fold_dir = os.path.join(output_base_dir, f"fold_{1}")
        os.makedirs(output_fold_dir, exist_ok=True)
        # write_folds(data, output_base_dir, DATASET_NAME)
        train_data = {k: v for k, v in data.items() if data[k]["dset"] == "Train"}
        dev_data = {k: v for k, v in data.items() if data[k]["dset"] == "Validation"}
        test_data = {k: v for k, v in data.items() if data[k]["dset"] == "Test1"} 
        
        write_jsonl(
            train_data, os.path.join(output_fold_dir, f"{DATASET_NAME}_train_fold_1.jsonl"), DATASET_NAME
        )
        write_jsonl(
            dev_data, os.path.join(output_fold_dir, f"{DATASET_NAME}_valid_fold_1.jsonl"), DATASET_NAME
        )
        write_jsonl(
            test_data, os.path.join(output_fold_dir, f"{DATASET_NAME}_test_fold_1.jsonl"), DATASET_NAME
        )

        write_json(
            train_data, os.path.join(output_fold_dir, f"{DATASET_NAME}_train_fold_1.json"), DATASET_NAME
        )
        write_json(
            dev_data, os.path.join(output_fold_dir, f"{DATASET_NAME}_valid_fold_1.json"), DATASET_NAME
        )
        write_json(
            test_data, os.path.join(output_fold_dir, f"{DATASET_NAME}_test_fold_1.json"), DATASET_NAME
        ) 
        

    print(f"Emotion frequency: {emotion_freq}")


if __name__ == "__main__":
    process_msppodcast(
        "downloads/msppodcast", output_format=["mini_format", "jsonl", "json", "split"]
    )
