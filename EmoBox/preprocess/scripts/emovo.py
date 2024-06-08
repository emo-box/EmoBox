import os
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "emovo"
SAMPLE_RATE = 48000

def parse_filename(filename):
    """
    Parses a filename based on the specified Italian emotional speech dataset conventions.

    :param filename: The filename to parse
    :return: A dictionary with the extracted information
    """
    # Assuming the filename format is 'emotion-actor-phrase.wav'
    parts = filename.split('-')
    if len(parts) != 3:
        raise ValueError("Filename format is incorrect.")

    # Extracting information from each part
    emotion_code = parts[0]
    actor_id = parts[1]
    phrase_type_code = parts[2].rstrip('.wav')

    # Mapping for emotions (Italian to English)
    emotions = {
        'neu': 'Neutral',
        'dis': 'Disgust',
        'gio': 'joy',
        'pau': 'Fear',
        'rab': 'Anger',
        'sor': 'Surprise',
        'tri': 'Sad'
    }

    actors_info = {
        'M1': ('male', 30),
        'M2': ('male', 27),
        'M3': ('male', 30),
        'F1': ('female', 28),
        'F2': ('female', 23),
        'F3': ('female', 25)
    }

    # Mapping for phrase types
    sentences = {
        'b1': ("Gli operai si alzano presto.", "The workers get up early."),
        'b2': ("I vigili sono muniti di pistola.", "The guards are equipped with a gun."),
        'b3': ("La cascata fa molto rumore.", "The waterfall makes a lot of noise."),
        'l1': ("L’autunno prossimo Tony partirà per la Spagna: nella prima metà di ottobre.",
               "Next autumn Tony will leave for Spain: in the first half of October."),
        'l2': ("Ora prendo la felpa di là ed esco per fare una passeggiata.",
               "Now I take the sweatshirt from there and go out for a walk."),
        'l3': (
            "Un attimo dopo s’è incamminato...ed è inciampato.", "A moment later he started walking...and stumbled."),
        'l4': ("Vorrei il numero telefonico del Signor Piatti.", "I would like Mr. Piatti's phone number."),
        'n1': ("La casa forte vuole col pane.", "The strong house wants with bread."),
        'n2': ("La forza trova il passo e l’aglio rosso.", "The force finds the step and the red garlic."),
        'n3': ("Il gatto sta scorrendo nella pera", "The cat is running in the pear"),
        'n4': ("Insalata pastasciutta coscia d’agnello limoncello.", "Pasta salad lamb thigh limoncello."),
        'n5': ("Uno quarantatré dieci mille cinquantasette venti.", "One forty-three ten thousand fifty-seven twenty."),
        'd1': ("Sabato sera cosa farà?", "What will you do Saturday night?"),
        'd2': ("Porti con te quella cosa?", "Do you carry that thing with you?")
    }

    # emotion = emotions.get(emotion_code, 'Unknown Emotion')
    italian_sentence, english_sentence = sentences.get(phrase_type_code, ('Unknown Sentence', 'Unknown Translation'))
    gender, age = actors_info.get(actor_id, ('Unknown', 'Unknown'))

    # Constructing the result dictionary
    result = {
        'id': filename.replace(".wav", ''),
        'lang': 'it',
        'age': age,
        'gd': gender,
        'spk': actor_id,
        'emotion': emotion_code,
        'transcript': italian_sentence,
        'translated_english_sentence': english_sentence
    }

    return result


def process_emovo(
    dataset_path, output_base_dir="data/emovo", output_format: str | list = "jsonl"
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
    process_emovo(
        "downloads/emovo", output_format=["mini_format", "jsonl", "json", "split"]
    )
