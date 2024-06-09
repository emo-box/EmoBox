import os
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *

DATASET_NAME = "meld"
SAMPLE_RATE = 48000


def parse_filename(filename, df, dataset_type):
    parts = filename.replace("final_videos_test", "").replace(".mp4", "").split("_")
    dia_id = parts[0].replace("dia", "")
    utt_id = parts[1].replace("utt", "")
    row = df.loc[
        (df["Dialogue_ID"] == int(dia_id)) & (df["Utterance_ID"] == int(utt_id))
    ]
    if row.empty:
        print(f"File {filename} does not exist")
        return
    spk = row["Speaker"].values[0]
    emotion = row["Emotion"].values[0]
    transcript = row["Utterance"].values[0]
    return {
        "id": f"{filename.replace('.mp4', '')}-{dataset_type}",
        "lang": "en",
        "dia_id": dia_id,
        "utt_id": utt_id,
        "spk": spk.replace(" ", "-"),
        "emotion": emotion,
        "transcript": transcript.replace(" ", "-"),
    }


def process_meld(
    dataset_path, output_base_dir="data/meld", output_format: str | list = "jsonl"
):

    # Load dataframes
    train_df = pd.read_csv(os.path.join(dataset_path, "train", "train_sent_emo.csv"))
    dev_df = pd.read_csv(os.path.join(dataset_path, "dev_sent_emo.csv"))
    test_df = pd.read_csv(os.path.join(dataset_path, "test_sent_emo.csv"))

    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    data = {}

    progress_bar = tqdm(
        total=len(train_df) + len(dev_df) + len(test_df), desc="Processing MELD"
    )

    # Process each dataset separately
    for df, dataset_type, dataset_subpath in [
        (train_df, "train", "train/train_splits"),
        (dev_df, "dev", "dev_splits_complete"),
        (test_df, "test", "output_repeated_splits_test"),
    ]:
        for _, row in df.iterrows():
            filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            file_path = os.path.join(dataset_path, dataset_subpath, filename)
            waveform, sample_rate = load_audio(file_path)
            if waveform is None:
                continue

            if sample_rate != SAMPLE_RATE:
                print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")
            
            sid = f"{DATASET_NAME}-dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}-{dataset_type}"
            metadata = parse_filename(filename, df, dataset_type)
            if metadata:
                data[sid] = {
                    "audio": file_path,
                    "emotion": metadata["emotion"],
                    "sid": sid,
                    "spk": metadata["spk"],
                    "start_time": 0,
                    "end_time": waveform.size(1) / sample_rate,
                    "sample_rate": sample_rate,
                    "duration": waveform.size(1) / sample_rate,
                    "channel": 1 if waveform.size(0) == 2 else 3,
                    "dset": dataset_type,
                }
            progress_bar.update(1)
    progress_bar.close()

    if output_format == "mini_format" or "mini_format" in output_format:
        write_mini_format(data, output_base_dir)
    if output_format == "jsonl" or "jsonl" in output_format:
        # Handle single-file JSON or JSONL output for the entire dataset
        jsonl_file_path = os.path.join(output_base_dir, "meld.jsonl")
        write_jsonl(data, jsonl_file_path, "meld")
    if output_format == "json" or "json" in output_format:
        # Handle single-file JSON output for the entire dataset
        json_file_path = os.path.join(output_base_dir, "meld.json")
        write_json(data, json_file_path, "meld")
    if output_format == "split" or "split" in output_format:
        output_fold_dir = os.path.join(output_base_dir, f"fold_{1}")
        os.makedirs(output_fold_dir, exist_ok=True)

        train_data = {k: v for k, v in data.items() if v["dset"] == "train"}
        valid = {k: v for k, v in data.items() if v["dset"] == "dev"}
        test_data = {k: v for k, v in data.items() if v["dset"] == "test"}

        write_jsonl(
            train_data, os.path.join(output_fold_dir, "meld_train_fold_1.jsonl"), "meld"
        )
        write_jsonl(
            valid, os.path.join(output_fold_dir, "meld_valid_fold_1.jsonl"), "meld"
        )
        write_jsonl(
            test_data, os.path.join(output_fold_dir, "meld_test_fold_1.jsonl"), "meld"
        )

        write_json(
            train_data, os.path.join(output_fold_dir, "meld_train_fold_1.json"), "meld"
        )
        write_json(
            valid, os.path.join(output_fold_dir, "meld_valid_fold_1.json"), "meld"
        )
        write_json(
            test_data, os.path.join(output_fold_dir, "meld_test_fold_1.json"), "meld"
        )


if __name__ == "__main__":
    # Example usage
    process_meld(
        "downloads/meld", output_format=["mini_format", "jsonl", "json", "split"]
    )
