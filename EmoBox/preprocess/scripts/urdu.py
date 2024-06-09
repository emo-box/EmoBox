import os
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "urdu"
SAMPLE_RATE = 44100


def parse_filename(filename):
    """
    Parse the filename from the URDU dataset and return a dictionary with details.

    :param filename: The filename in the format 'SGX_FXX_EYY'
    :return: A dictionary with the speaker's gender, ID, file number, and emotion
    """
    # Split the filename into its components
    parts = filename.split('_')
    if len(parts) != 3:
        raise ValueError("Filename format should be 'SGX_FXX_EYY'")

    # Extracting details from each part
    speaker_gender = 'Male' if parts[0][1] == 'M' else 'Female'
    speaker_id = parts[0]
    file_number = parts[1][1:]

    # Mapping emotion codes to emotions
    emotions = {'A': 'Angry', 'H': 'Happy', 'N': 'Neutral', 'S': 'Sad'}
    emotion_code = parts[2][0]
    emotion = emotions.get(emotion_code, 'Unknown')

    # Constructing the result dictionary
    result = {
        'id': filename.replace(".wav", '').replace("_", "-"),
        'gd': speaker_gender,
        'lang': 'urd',
        'spk': speaker_id,
        #'File Number': file_number,
        'emotion': emotion_code
    }

    return result
    
def process_urdu(
    dataset_path, output_base_dir="data/urdu", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    speaker_set = set()
    data = {}
    emotion_freq = defaultdict(int)
    
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
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
        sid = f"{DATASET_NAME}-{file_path.split('/')[-1].replace('.wav', '')}"
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
        
        speaker_set.add(parsed_info["spk"])

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
    process_urdu(
        "downloads/urdu", output_format=["mini_format", "jsonl", "json", "split"]
    )