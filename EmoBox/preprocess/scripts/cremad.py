import os
import re
from collections import defaultdict
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "cremad"
SAMPLE_RATE = 16000


def parse_filename(filename, spk_table):
    """
    Parses the filename based on the specified conventions and extracts information.

    :param filename: The filename to parse
    :return: A dictionary with the extracted information
    """
    # Split the filename into its components
    parts = filename.split('_')
    if len(parts) < 4:
        raise ValueError("Filename format is incorrect.")

    # Extracting information from each part
    actor_id = parts[0]
    actor_info = spk_table.loc[spk_table['ActorID'] == int(actor_id)]

    sentence_code = parts[1]
    emotion_code = parts[2]
    emotion_level_code = parts[3].split('.')[0]  # Split again to remove file extension

    # Mapping for sentences
    sentences = {
        'IEO': "It's eleven o'clock",
        'TIE': "That is exactly what happened",
        'IOM': "I'm on my way to the meeting",
        'IWW': "I wonder what this is about",
        'TAI': "The airplane is almost full",
        'MTI': "Maybe tomorrow it will be cold",
        'IWL': "I would like a new alarm clock",
        'ITH': "I think I have a doctor's appointment",
        'DFA': "Don't forget a jacket",
        'ITS': "I think I've seen this before",
        'TSI': "The surface is slick",
        'WSI': "We'll stop in a couple of minutes"
    }

    # Mapping for emotion levels
    emotion_levels = {
        'LO': 'low',
        'MD': 'medium',
        'HI': 'high',
        'XX': 'unspecified'
    }

    sentence = sentences.get(sentence_code, 'Unknown Sentence')
    emotion = emotion_code
    emotion_level = emotion_levels.get(emotion_level_code, 'Unknown Emotion Level')

    # Constructing the result dictionary
    result = {
        'id': filename.replace(".wav", ''),
        'age': actor_info['Age'].values[0],
        'lang': 'en',
        'gd': actor_info['Sex'].values[0],
        'spk': actor_id,
        'race': actor_info["Race"].values[0],
        'Sentence': sentence,
        'emotion': emotion,
        'emotion_level': emotion_level
    }

    return result


def process_cremad(
    dataset_path, output_base_dir="data/cremad", output_format: str | list = "jsonl"
):
    """
    Preprocesses the CREMA-D dataset and saves it to the specified output directory.

    :param dataset_path: The path to the dataset
    :param output_base_dir: The base directory to save the preprocessed data
    :param output_format: The format to save the preprocessed data in
    """
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)

    # Load the speaker table
    spk_table = pd.read_csv(os.path.join(dataset_path, "VideoDemographics.csv"))

    data = {}
    emotion_freq = defaultdict(int)

    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(os.path.join(dataset_path, "AudioWAV"))
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

        parsed_info = parse_filename(os.path.basename(file_path), spk_table)
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
            "lang": parsed_info["lang"],
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
    process_cremad(
        "downloads/cremad", output_format=["mini_format", "jsonl", "json", "split"]
    )
