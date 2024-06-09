import os
import re
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *

from tqdm import tqdm

DATASET_NAME = "aesdd"
SAMPLE_RATE = 44100


def parse_filename(filename):
    emotion_code = filename[0]
    utterance_number = int(filename[1:3])
    speaker_number = (
        (
            re.search(r"\((\d+)\)", filename).group(1)
            if re.search(r"\((\d+)\)", filename)
            else "Unknown"
        ),
    )

    return {
        "id": filename.replace(".wav", "")
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", ""),
        "spk": speaker_number[0].replace("0", ""),
        "lang": "el",
        "emotion": emotion_code,
        "utterance_number": utterance_number,
    }


def process_aesdd(
    dataset_path, output_base_dir="data/aesdd", output_format: str | list = "jsonl"
):

    os.makedirs(output_base_dir, exist_ok=True)
    emotion_freq = defaultdict(int)
    data = {}

    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if file.lower().endswith(".wav")
    ]

    for file_path in tqdm(all_files, desc="Processing AESDD files"):
        file_path = get_relative_audio_path(file_path)
        waveform, sample_rate = load_audio(file_path)
        if waveform is None:
            continue
        if sample_rate != SAMPLE_RATE:
            print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")
        num_frame = waveform.size(1)
        metadata = parse_filename(file_path.split("/")[-1])
        sid = f"{DATASET_NAME}-{metadata['id']}"

        data[sid] = {
            "audio": file_path,
            "emotion": metadata["emotion"],
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": num_frame,
            "spk": metadata["spk"],
            "start_time": 0,
            "end_time": num_frame / sample_rate,
            "duration": num_frame / sample_rate,
        }
        emotion_freq[metadata["emotion"]] += 1

    if output_format == "mini_format" or "mini_format" in output_format:
        write_mini_format(data, output_base_dir)

    if output_format == "jsonl" or "jsonl" in output_format:
        jsonl_file_path = os.path.join(output_base_dir, f"{DATASET_NAME}.jsonl")
        write_jsonl(data, jsonl_file_path, DATASET_NAME)

    if output_format == "json" or "json" in output_format:
        json_file_path = os.path.join(output_base_dir, f"{DATASET_NAME}.json")
        write_json(data, json_file_path, DATASET_NAME)

    if output_format == "split" or "split" in output_format:
        write_folds(data, output_base_dir, DATASET_NAME)


if __name__ == "__main__":
    process_aesdd(
        dataset_path="downloads/aesdd",
        output_format=["mini_format", "jsonl", "json", "split"],
    )
