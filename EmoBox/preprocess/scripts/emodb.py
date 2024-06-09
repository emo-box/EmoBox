import os
import re
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "emodb"
SAMPLE_RATE = 16000


def process_emodb(
    dataset_path, output_base_dir="data/emodb", output_format: str | list = "jsonl"
):

    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    
    file_set = set()
    total_fail_files = 0
    all_files = []
    
    emotion_list_dir = os.path.join(dataset_path, "lists", "emotions")
    spkear_list_dir = os.path.join(dataset_path, "lists", "speakers")
    speaker_dict = {}
    
    for emotion in ["anger", "boredem", "disgust", "fear", "happy", "neutral", "sad"]:
        with open(os.path.join(emotion_list_dir, emotion+".txt"), "r") as f:
            for line in f:
                file_path = os.path.join(dataset_path, "wav", line.strip())
                all_files.append((file_path, emotion))
                
    for file in os.listdir(spkear_list_dir):
        with open(os.path.join(spkear_list_dir, file), "r") as f:
            for line in f:
                speaker_dict[line.strip()] = file.split(".")[0]
                
    fold_dict = {}
    
    for fold in range(1, 6):
        with open(os.path.join(dataset_path, "lists", "cv_k_fold" + str(fold)), "r") as f:
            for line in f:
                file_path = os.path.join(dataset_path, "wav", line.strip())
                fold_dict[file_path] = fold

    data = {}
    emotion_freq = defaultdict(int)
    
    for file_path, emotion in tqdm(all_files):        
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            total_fail_files += 1
            print(f"Error processing file {file_path}: {e}")
            continue
            
        sid = f"{DATASET_NAME}-{file_path.split('/')[-1].replace('.wav', '')}"
        data[sid] = {
            "audio": file_path,
            "emotion": emotion,
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": speaker_dict[file_path.split('/')[-1]],
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
        # Handle split JSONL output
        for fold in range(1, 6):
            os.makedirs(os.path.join(output_base_dir, f"fold_{fold}"), exist_ok=True)
            train_data = {}
            test_data = {}
            
            for sid, d in data.items():
                if fold_dict[d["audio"]] == fold:
                    test_data[sid] = d
                else:
                    train_data[sid] = d
        
            train_jsonl_file_path = os.path.join(output_base_dir, f"fold_{fold}", f"{DATASET_NAME}_train_fold_{fold}.jsonl")
            test_jsonl_file_path = os.path.join(output_base_dir, f"fold_{fold}", f"{DATASET_NAME}_test_fold_{fold}.jsonl")
            write_jsonl(train_data, train_jsonl_file_path, DATASET_NAME)
            write_jsonl(test_data, test_jsonl_file_path, DATASET_NAME)
            
            train_json_file_path = os.path.join(output_base_dir, f"fold_{fold}", f"{DATASET_NAME}_train_fold_{fold}.json")
            test_json_file_path = os.path.join(output_base_dir, f"fold_{fold}", f"{DATASET_NAME}_test_fold_{fold}.json")
            write_json(train_data, train_json_file_path, DATASET_NAME)
            write_json(test_data, test_json_file_path, DATASET_NAME)
            
            
    print(f"Total fail files: {total_fail_files}")
    print(f"Emotion frequency: {emotion_freq}")
    
if __name__ == "__main__":
    process_emodb(
        "downloads/emodb", output_format=["mini_format", "jsonl", "json", "split"]
    )
