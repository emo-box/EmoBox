import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "esd"
SAMPLE_RATE = 16000

def parse_text_file(file_path):
    """
    Parses a text file and extracts each line as a tuple of (id, text, sentiment).

    :param file_path: Path to the text file
    :return: A list of tuples with the extracted information
    """
    tuple_list = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split("\t")  # Split each line by tab
            if len(parts) == 3:
                # match chinese part3 to english part3
                if parts[2] == "中立":
                    parts[2] = "Neutral"
                elif parts[2] == "快乐":
                    parts[2] = "Happy"
                elif parts[2] == "伤心":
                    parts[2] = "Sad"
                elif parts[2] == "生气":
                    parts[2] = "Angry"
                elif parts[2] == "惊喜":
                    parts[2] = "Surprise"
                tuple_list.append((parts[0] + ".wav", parts[1], parts[2]))

    return tuple_list


def process_esd(
    dataset_path, output_base_dir="data/esd", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    file_list = []
    
    speakers = [f"{i:04}" for i in range(1, 21)]
    for spk in speakers:
        for file_name, transcript, emotion in parse_text_file(os.path.join(dataset_path, spk, f"{spk}.txt")):
            file_path = os.path.join(dataset_path, spk, emotion, file_name)
            file_list.append((file_path, spk, emotion))
            
    for file_path, spk, emotion in tqdm(file_list, desc=f"Processing {DATASET_NAME} files"):
        waveform, sample_rate = load_audio(file_path)
        num_frame = waveform.size(1)
        
        seg_id = f"{DATASET_NAME}-{file_path.split('/')[-1].replace('.wav', '').replace('_', '-')}"
        
        data[seg_id] = {
            "audio": file_path,
            "emotion": emotion,
            "channel": 1,
            "sid": seg_id,
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
    process_esd("downloads/esd", output_format=["mini_format", "jsonl", "json", "split"])