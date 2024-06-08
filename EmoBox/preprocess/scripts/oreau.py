import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "oreau"
SAMPLE_RATE = 44100


def parse_filename(filename):
    """
    Parses a filename from the Oréau database with the format XXYYYS.wav
    and returns a dictionary with the extracted information.

    - XX: Speaker identifier
    - YYY: Utterance identifier
    - S: Emotional Style (with specific meanings for each letter)
    - .wav: Sample format

    For example: '12ABCN.wav' ->
    {
        'Speaker Identifier': '12',
        'Utterance Identifier': 'ABC',
        'Emotional Style': 'Neutral'
    }
    """
    # Define the emotional styles mapping


    # Remove the file extension
    name_without_extension = filename.split('.')[0]

    # Extract the components from the filename
    speaker_id = name_without_extension[:2]
    utterance_id = name_without_extension[2:5]
    emotional_style_letter = name_without_extension[5]
    emotional_style = emotional_style_letter

    # Construct the dictionary
    parsed_info = {
        "id": name_without_extension,
        'spk': speaker_id,
        'transcript_id': utterance_id,
        'emotion': emotional_style
    }

    return parsed_info

def process_oreau(
    dataset_path, output_base_dir="data/oreau", output_format: str | list = "jsonl"
):
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(dataset_path, "OréauFR_02"))
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
        sid = f"{DATASET_NAME}-{parsed_info['id'].replace(' ', '-')}"
        data[sid] = {
            "audio": file_path.replace("OréauFR_02", "OréauFR_02"),
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
    process_oreau(
        "downloads/oreau", output_format=["mini_format", "jsonl", "json", "split"]
    )