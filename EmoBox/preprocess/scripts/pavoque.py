import os
import re
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm
import yaml

DATASET_NAME = "pavoque"
SAMPLE_RATE = 44100

def process_pavoque(dataset_path, output_base_dir="data/pavoque", out_audio_dir="processed_data/pavoque", output_format: str | list = "jsonl"):
    """
    Process the Pavoque dataset and generate output in specified format.

    Parameters:
    - dataset_path: Path to the raw Pavoque dataset.
    - output_base_dir: Base directory for the processed output.
    - output_format: The format for the output data ("json", "jsonl", "mini", "split").

    Returns:
    - None
    """
    os.makedirs(out_audio_dir, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)
    
    data = {}
    emotion_freq = defaultdict(int)
    
    for files in os.listdir(dataset_path):
        if not files.lower().endswith('.yaml'):
            continue
        file_path = os.path.join(dataset_path, files)
        # load the yaml file
        with open(file_path, 'r', encoding='utf8') as stream:
            try:
                meta_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
        input_audio_path = file_path.replace('.yaml', '.flac')
        waveform, sample_rate = load_audio(input_audio_path)
        
        for d in tqdm(meta_data, desc=f"Processing {files}"):
            # get the parsed info
            parsed_info = {
                'id': f"{d['prompt']}-{d['style']}",
                "lang": "de", 
                "emotion": d["style"].lower(), 
                # "transcript": d["text"]
            }
            sid = f"pavoque-{d['prompt']}-{d['style']}"
            channel_id = 1
            speaker = "STEFAN-RÖTTIG"
            start_time = d['start']
            end_time = d['end']
            split_file_name = f"{sid}.wav"
            
            # split the audio
            output_audio_path = os.path.join(out_audio_dir, split_file_name)
            # split_audio_torchaudio(waveform, sample_rate, start_time, end_time, output_audio_path)
            
            data[sid] = {
                "audio": input_audio_path,
                "emotion": parsed_info["emotion"],
                "channel": channel_id,
                "sid": sid,
                "sample_rate": sample_rate,
                "num_frame": (end_time - start_time)*sample_rate,
                "spk": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
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
    process_pavoque(
        "downloads/pavoque", output_format=["mini_format", "jsonl", "json", "split"]
    )