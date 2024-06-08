import os
from collections import defaultdict
import sys
from pandas import read_csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "ravdess"
SAMPLE_RATE = 48000


def parse_file_name(filename):

    parts = filename.split(".")[0].split("-")
    (
        modality,
        vocal_channel,
        emotion,
        emotional_intensity,
        statement,
        repetition,
        actor,
    ) = parts

    modalities = {"01": "full-AV", "02": "video-only", "03": "audio-only"}
    vocal_channels = {"01": "speech", "02": "song"}
    emotions = {
        "01": "Neutral",
        "02": "Calm",
        "03": "Happy",
        "04": "Sad",
        "05": "Angry",
        "06": "Fearful",
        "07": "Disgust",
        "08": "Surprised",
    }
    emotional_intensities = {"01": "normal", "02": "strong"}
    statements = {
        "01": "Kids are talking by the door",
        "02": "Dogs are sitting by the door",
    }
    repetitions = {"01": "1st repetition", "02": "2nd repetition"}
    actor_gender = "Male" if int(actor) % 2 != 0 else "Female"

    return {
        "id": filename.replace(".wav", ""),
        "Modality": modalities[modality],
        "Vocal Channel": vocal_channels[vocal_channel],
        "emotion": emotions[emotion],
        "Emotional Intensity": emotional_intensities[emotional_intensity],
        "Statement": statements[statement],
        "Repetition": repetitions[repetition],
        "spk": actor,
        "Gender": actor_gender,
    }


def process_ravdess(
    dataset_path, output_base_dir="data/ravdess", output_format: str | list = "jsonl",
):
    os.makedirs(output_base_dir, exist_ok=True)

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
        waveform, sample_rate = load_audio(file_path)
        num_frame = waveform.size(1)

        if sample_rate != SAMPLE_RATE:
            print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")

        parsed_info = parse_file_name(os.path.basename(file_path))

            
        sid = f"{DATASET_NAME}-{parsed_info['id']}"
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
    process_ravdess(
        "downloads/ravdess", output_format=["mini_format", "jsonl", "json", "split"]
    )