import os
from collections import defaultdict
import sys
from pandas import read_csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "jlcorpus"
SAMPLE_RATE = 44100

def parse_filename(filename):
    # Remove file extension
    name = filename.split('.')[0]

    # Split the filename by underscores
    parts = name.split('_')

    # Extract the individual components
    gender = 'male' if 'male' in parts[0] else 'female'
    speaker_id = parts[0]  # Assumes the speaker ID is always the last character

    emotion = parts[1]
    # Separate Sentence ID and Session ID
    sentence_id = ''.join(filter(lambda x: not x.isdigit(), parts[2]))
    session_id = ''.join(filter(str.isdigit, parts[2]))

    # Construct the dictionary
    parsed_info = {
        "id": name,
        'gd': gender,
        'lang': 'en',
        'spk': speaker_id,
        'emotion': emotion,
        'sentence_id': sentence_id,
        'session_id': session_id
    }
    return parsed_info


def process_jlcorpus(
    dataset_path, output_base_dir="data/jlcorpus", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(dataset_path, "Raw JL corpus (unchecked and unannotated)/JL(wav+txt)"))
        for file in files
        if file.lower().endswith(".wav")
    ]
    

    # Processing files with a progress bar
    for file_path in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
        file_path = get_relative_audio_path(file_path)
        waveform, sample_rate = load_audio(file_path)
        num_frame = waveform.size(1)

        if sample_rate != SAMPLE_RATE:
            print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")

        parsed_info = parse_filename(file_path)
        sid = f"{DATASET_NAME}-{file_path.split('/')[-1].replace('.wav', '').replace('_', '-')}"
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
    process_jlcorpus(
        "downloads/jlcorpus", output_format=["mini_format", "jsonl", "json", "split"]
    )