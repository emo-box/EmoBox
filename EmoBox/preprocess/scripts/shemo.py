import os
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

SAMPLE_RATE = 44100
DATASET_NAME = "shemo"


def process_shemo(
    dataset_path, output_base_dir="data/shemo", output_format: str | list = "jsonl"
):
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)

    data = {}
    emotion_freq = defaultdict(int)
    
    with open(os.path.join(dataset_path, "shemo.json")) as f:
        meta_data = json.load(f)


    for file in tqdm(meta_data, desc=f"Processing {DATASET_NAME} files"):
        file_path = os.path.join(dataset_path, meta_data[file]['gender'], file + ".wav")
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        seg_id = f"{DATASET_NAME}-{file}"

        data[seg_id] = {
            "audio": file_path,
            "emotion": meta_data[file]['emotion'].lower(),
            "channel": 1,
            "sid": seg_id,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": meta_data[file].pop("speaker_id"),
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
        }
        emotion_freq[meta_data[file]['emotion'].lower()] += 1
        
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
    process_shemo(
        "downloads/shemo", output_format=["mini_format", "jsonl", "json", "split"]
    )
