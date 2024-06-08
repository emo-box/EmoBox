import os
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *

DATASET_NAME = "cafe"
SAMPLE_RATE = 48000

def parse_cafe_filename(filename):
    """
    Parses a CaFE database filename and extracts information as a dictionary.

    :param filename: The filename to parse (e.g., '01-C-1-1.wav')
    :return: A dictionary with the extracted information
    """
    # Assuming the filename format is 'AA-E-I-S.wav'
    parts = filename.split('-')
    if len(parts) != 4:
        raise ValueError("Filename format should be 'AA-E-I-S.wav'")

    # Extracting information
    actor_number = parts[0].rstrip('.wav')
    emotion_code = parts[1]
    intensity = parts[2]
    sentence_number = parts[3].rstrip('.wav')

    # Mapping for actor's age and gender
    actors_info = {
        '01': ('46', 'male'),
        '02': ('64', 'female'),
        '03': ('18', 'male'),
        '04': ('50', 'female'),
        '05': ('22', 'male'),
        '06': ('34', 'female'),
        '07': ('15', 'male'),
        '08': ('25', 'female'),
        '09': ('42', 'male'),
        '10': ('20', 'female'),
        '11': ('35', 'male'),
        '12': ('37', 'female')
    }

    # Mapping for sentences in French and English
    sentences = {
        '1': ('Un cheval fou dans mon jardin', 'One crazy horse in my garden'),
        '2': ('Deux ânes aigris au pelage brun', 'Two embittered donkeys with a brown coat'),
        '3': ('Trois cygnes aveugles au bord du lac', 'Three blind swans by the lake'),
        '4': ('Quatre vieilles truies éléphantesques', 'Four gigantic old sows'),
        '5': ('Cinq pumas fiers et passionnés', 'Five proud and passionate pumas'),
        '6': ('Six ours aimants domestiqués', 'Six domesticated caring bears')
    }

    actor_age, gender = actors_info.get(actor_number, ('Unknown', 'Unknown'))
    french_sentence, english_sentence = sentences.get(sentence_number, ('Unknown', 'Unknown'))

    # Constructing the result dictionary
    result = {
        "id": filename.replace(".wav", '').replace(".aiff", ''),
        'age': actor_age,
        'lang': 'fr',
        'gd': gender,
        'spk': actor_number,
        'emotion': emotion_code.lower(),
        'intensity': 'low' if intensity == '1' else 'high',
        'sentence_id': sentence_number,
        'transcription': french_sentence,
        'english_translation': english_sentence
    }

    return result

def process_cafe(dataset_path, output_base_dir="data/cafe", output_format: str | list = "jsonl"):
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)

    data = {}
    emotion_freq = defaultdict(int)

    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if file.lower().endswith(".wav")
    ]

    for file_path in tqdm(all_files, desc=f"Processing {DATASET_NAME} files"):
        file_path = get_relative_audio_path(file_path)
        try:
            waveform, sample_rate = load_audio(file_path)
            num_frame = waveform.size(1)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        if sample_rate != SAMPLE_RATE:
            print(f"Sample rate of {file_path} is not {SAMPLE_RATE}")

        parsed_info = parse_cafe_filename(os.path.basename(file_path))
        sid = f"{DATASET_NAME}-{parsed_info['id'].replace('_', '-')}"

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
    process_cafe(
        "downloads/cafe", output_format=["mini_format", "jsonl", "json", "split"]
    )
    