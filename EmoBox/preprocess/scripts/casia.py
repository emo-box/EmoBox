import os
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

SAMPLE_RATE = 16000
DATASET_NAME = "casia"


def process_casia(
    dataset_path, output_base_dir="data/casia", output_format: str | list = "jsonl"
):
    os.makedirs(output_base_dir, exist_ok=True)
    data = {}
    
    emotion_freq = defaultdict(int)
    
    all_files = []
    for speaker in os.listdir(dataset_path):
        # check if the speaker is a directory
        if not os.path.isdir(os.path.join(dataset_path, speaker)):
            continue
        for emotion in os.listdir(os.path.join(dataset_path, speaker)):
            # check if the emotion is a directory
            if not os.path.isdir(os.path.join(dataset_path, speaker, emotion)):
                continue
            for file in os.listdir(os.path.join(dataset_path, speaker, emotion)):
                if file.endswith(".wav"):
                    file_path = os.path.join(dataset_path, speaker, emotion, file)
                    all_files.append((file_path, speaker, emotion))
                    
    for file_path, speaker, emotion in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
        
        sid = f"{DATASET_NAME}-{os.path.basename(file_path).replace('.wav', '')}-{speaker}-{emotion}"
        
        data[sid] = {
            "audio": file_path,
            "emotion": emotion,
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": speaker,
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
        }
        emotion_freq[emotion] += 1
        
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
    process_casia(
        "downloads/casia", output_format=["mini_format", "jsonl", "json", "split"]
    )
