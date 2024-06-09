import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "mesd"
SAMPLE_RATE = 48000

def parse_filename(filename):
    voice_types = {'F': 'Female', 'M': 'Male', 'C': 'Child'}
    corpus_types = {'A': 'Corpus A', 'B': 'Corpus B'}
    
    # Remove the file extension
    name_without_extension = filename.split('.')[0]

    # Split the filename by underscores
    emotion = name_without_extension.split('_')[0]
    voice_type_letter = name_without_extension.split('_')[1]
    corpus_letter = name_without_extension.split('_')[2]
    word = " ".join(name_without_extension.split('_')[3:])

    # Map the type of voice and word corpus to their full meaning
    voice_type = voice_types.get(voice_type_letter, 'Unknown')
    word_corpus = corpus_types.get(corpus_letter, 'Unknown')
    
    # Construct the dictionary
    parsed_info = {
        "id": name_without_extension,
        'gd': voice_type,
        'emotion': emotion,
        # 'Word Corpus': word_corpus,
        'transcript': word
    }

    return parsed_info

def process_mesd(
    dataset_path, output_base_dir="data/mesd", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(dataset_path, "Mexican Emotional Speech Database (MESD)"))
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
        sid = f"{DATASET_NAME}-{parsed_info['id']}"
        data[sid] = {
            "audio": file_path,
            "emotion": parsed_info["emotion"],
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": parsed_info["gd"],
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
    process_mesd(
        "downloads/mesd", output_format=["mini_format", "jsonl", "json", "split"]
    )