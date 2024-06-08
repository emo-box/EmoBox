import os
import re
from collections import defaultdict
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "m3ed"
SAMPLE_RATE = 16000


def process_m3ed(
    dataset_path, output_base_dir="data/m3ed", output_format: str | list = "jsonl"
):
    spk_set = set()
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    
    with open(os.path.join(dataset_path, "annotation.json"), "r") as f:
        meta_info = json.load(f)
        
    with open(os.path.join(dataset_path, "splitInfo", "movie_list_test.txt"), "r") as f:
        test_movie_set = set([i.replace("\n", '') for i in f.readlines()])
        
    with open(os.path.join(dataset_path, "splitInfo", "movie_list_train.txt"), "r") as f:
        train_movie_set = set([i.replace("\n", '') for i in f.readlines()])
    
    with open(os.path.join(dataset_path, "splitInfo", "movie_list_val.txt"), "r") as f:
        val_movie_set = set([i.replace("\n", '') for i in f.readlines()])

    data = {}
    emotion_freq = defaultdict(int)
    
    total_utterances = sum(len(clip["Dialog"]) for movie_name in meta_info for clip in meta_info[movie_name].values())
    progress_bar = tqdm(total=total_utterances, desc=f"Processing {DATASET_NAME} files")
    
    total_fail_files = 0
    
    for movie_name in meta_info:
        if movie_name in test_movie_set:
            split = "test"
        elif movie_name in train_movie_set:
            split = "train"
        elif movie_name in val_movie_set:
            split = "valid"
        else:
            print(f"Movie {movie_name} not found in any split")
            
        for _, clip in meta_info[movie_name].items():
            spk_info = clip["SpeakerInfo"]
            for utterance_id, utterance_info in clip["Dialog"].items():
                spk_id = utterance_info["Speaker"]
                emotion = utterance_info["EmoAnnotation"]["final_main_emo"]
                file_name = f"{spk_id}_{utterance_id}.wav"
                file_path = os.path.join(dataset_path, "modality_speech", file_name)
                
                if not os.path.exists(file_path):
                    print(f"File {file_path} not found")
                    total_fail_files += 1
                    continue
                
                waveform, sample_rate = load_audio(file_path)
                if waveform is None:
                    total_fail_files += 1
                    continue
                num_frame = waveform.size(1)
                
                sid = f"{DATASET_NAME}-{utterance_id}"
                spk = f"{movie_name}_{spk_info[spk_id]['Name']}"
                spk_set.add(spk)
                data[sid] = {
                    # relative path from dataset_path
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
                    "lang": "zh",
                    "split": split,
                }
                emotion_freq[emotion] += 1
                progress_bar.update(1)
    
    progress_bar.close()

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
        for split in ["train", "valid", "test"]:
            split_data = {sid: d for sid, d in data.items() if d["split"] == split}
            split_jsonl_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_{split}_fold_1.jsonl")
            split_json_file_path = os.path.join(output_base_dir, "fold_1", f"{DATASET_NAME}_{split}_fold_1.json")
            write_jsonl(split_data, split_jsonl_file_path, DATASET_NAME)
            write_json(split_data, split_json_file_path, DATASET_NAME)
            
    print(f"Total fail files: {total_fail_files}")
    print(f"Emotion frequency: {emotion_freq}")
    print(f"Total speakers: {len(spk_set)}")
if __name__ == "__main__":
    process_m3ed(
        "downloads/m3ed", output_format=["mini_format", "jsonl", "json", "split"]
    )
