import os
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm


DATASET_NAME = "enterface"
SAMPLE_RATE = 16000

def parse_filename(filename):
    parts = filename.replace(".avi", "").split("_")
    subject_id = parts[0]
    emotion = parts[1]
    #sentence_id = parts[2]
    if emotion == "3":
        emotion = parts[2]
    
    return {
        "id": filename.replace(".wav", ""),
        "subject_id": subject_id,
        "emotion": emotion,
        #"sentence_id": sentence_id,
    }

def process_enterface(
    dataset_path, output_base_dir="data/enterface", output_format: str | list = "jsonl"
):
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    data = {}
    emotion_freq = defaultdict(int)
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if file.lower().endswith(".avi")
    ]
    
    
    for file_path in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        parsed_info = parse_filename(os.path.basename(file_path))
        sid = f"{DATASET_NAME}-{parsed_info['id']}"

        if parsed_info["emotion"] == "3":
            print(1)
        data[sid] = {
            "audio": file_path,
            "emotion": parsed_info["emotion"],
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": parsed_info["subject_id"],
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
        }
        emotion_freq[parsed_info["emotion"]] += 1

    if "mini_format" in output_format:
        write_mini_format(data, output_base_dir)

    if "jsonl" in output_format:
        write_jsonl(
            data, os.path.join(output_base_dir, f"{DATASET_NAME}.jsonl"), DATASET_NAME
        )

    if "json" in output_format:
        write_json(
            data, os.path.join(output_base_dir, f"{DATASET_NAME}.json"), DATASET_NAME
        )

    if "split" in output_format:
        write_folds(data, output_base_dir, DATASET_NAME)

    print(f"Emotion frequency: {emotion_freq}")


if __name__ == "__main__":
    process_enterface(
        "downloads/enterface", output_format=["mini_format", "jsonl", "json", "split"]
    )