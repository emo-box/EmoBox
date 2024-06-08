import os
import re
from collections import defaultdict
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocess_utils import *
from tqdm import tqdm

DATASET_NAME = "iemocap"
SAMPLE_RATE = 16000


def load_utter_info(input_file):
    """
    Extracts utterance information from the inputFile based on a predefined pattern.
    """
    pattern = re.compile(
        r"[\[]*[0-9]*[.][0-9]*[ -]*[0-9]*[.][0-9]*[\]][\t][a-z0-9_]*[\t][a-z]{3}[\t][\[][0-9]*[.][0-9]*[, "
        r"]+[0-9]*[.][0-9]*[, ]+[0-9]*[.][0-9]*[\]]",
        re.IGNORECASE,
    )
    with open(input_file, "r") as file:
        data = file.read().replace("\n", " ")
    result = pattern.findall(data)
    out = []
    for i in result:
        a = i.replace("[", "")
        b = a.replace(" - ", "\t")
        c = b.replace("]", "")
        x = c.replace(", ", "\t")
        out.append(x.split("\t"))
    return out


def load_session(path_session):
    """
    Loads session data, returning a list of utterances with their paths, labels, and session details.
    """
    path_emo = os.path.join(path_session, "dialog/EmoEvaluation/")
    path_wav_folder = os.path.join(path_session, "sentences/wav/")
    improvised_utterance_list = []
    for emoFile in os.listdir(path_emo):
        if os.path.isfile(os.path.join(path_emo, emoFile)):
            for utterance in load_utter_info(os.path.join(path_emo, emoFile)):
                path = os.path.join(
                    path_wav_folder, utterance[2][:-5], utterance[2] + ".wav"
                )
                label = utterance[3]
                if emoFile[7] != "i" and utterance[2][7] == "s":
                    improvised_utterance_list.append([path, label, utterance[2][18]])
                else:
                    improvised_utterance_list.append([path, label, utterance[2][15]])
    return improvised_utterance_list


def process_iemocap(
    dataset_path, output_base_dir="data/iemocap", output_format: str | list = "jsonl"
):
    """
    Process the IEMOCAP dataset and generate output in specified format.

    Parameters:
    - dataset_path: Path to the raw IEMOCAP dataset.
    - output_base_dir: Base directory for the processed output.
    - output_format: The format for the output data ("json", "jsonl", "mini", "split").

    Returns:
    - None
    """

    os.makedirs(output_base_dir, exist_ok=True)
    session_paths = [os.path.join(dataset_path, f"Session{i}") for i in range(1, 6)]

    data = {}

    utt_list = [
        (utterance, session_path)
        for session_path in session_paths
        for utterance in load_session(session_path)
    ]

    for utterance in tqdm(
        utt_list, desc=f"Processing {DATASET_NAME}", unit="utterance", leave=False
    ):
        (wav_file, emotion, _), session_path = utterance
        if emotion not in ["neu", "hap", "ang", "sad", "exc"]:
            continue
        waveform, sample_rate = load_audio(wav_file)
        if waveform is None or sample_rate is None:
            print(f"Error with file {wav_file}")
            continue
        uuid = os.path.basename(wav_file).split(".")[0]
        sid = f"{DATASET_NAME}-{uuid}"
        data[sid] = {
            "audio": wav_file,
            "emotion": emotion,
            "channel": 1,
            "sid": sid,
            "sample_rate": sample_rate,
            "num_frame": waveform.size(1),
            "spk": uuid.split("_")[0],
            "start_time": 0,
            "end_time": waveform.size(1) / sample_rate,
            "duration": waveform.size(1) / sample_rate,
            "session": session_path.split("/")[-1],
            "uuid": uuid,
        }

    if output_format == "mini_format" or "mini_format" in output_format:
        write_mini_format(data, output_base_dir)

    if output_format == "jsonl" or "jsonl" in output_format:
        # Handle single-file JSON or JSONL output for the entire dataset
        jsonl_file_path = os.path.join(output_base_dir, f"{DATASET_NAME}.jsonl")
        write_jsonl(data, jsonl_file_path, f"{DATASET_NAME}")

    if output_format == "json" or "json" in output_format:
        # Handle single-file JSON output for the entire dataset
        json_file_path = os.path.join(output_base_dir, f"{DATASET_NAME}.json")
        write_json(data, json_file_path, f"{DATASET_NAME}")

    if output_format == "split" or "split" in output_format:
        # Process and split data into folds
        for test_session_index in range(5):
            fold_number = test_session_index + 1
            output_fold_dir = os.path.join(output_base_dir, f"fold_{fold_number}")
            os.makedirs(output_fold_dir, exist_ok=True)

            train_jsonl_data = {}
            test_jsonl_data = {}
            emotion_freq = defaultdict(int)

            for key in data:
                if data[key]["session"] == f"Session{test_session_index + 1}":
                    test_jsonl_data[key] = data[key]
                else:
                    train_jsonl_data[key] = data[key]
                    emotion_freq[data[key]["emotion"]] += 1

            train_jsonl_file_path = os.path.join(
                output_fold_dir, f"{DATASET_NAME}_train_fold_{fold_number}.jsonl"
            )
            test_jsonl_file_path = os.path.join(
                output_fold_dir, f"{DATASET_NAME}_test_fold_{fold_number}.jsonl"
            )

            train_json_file_path = os.path.join(
                output_fold_dir, f"{DATASET_NAME}_train_fold_{fold_number}.json"
            )
            test_json_file_path = os.path.join(
                output_fold_dir, f"{DATASET_NAME}_test_fold_{fold_number}.json"
            )

            write_json(train_jsonl_data, train_json_file_path, DATASET_NAME)
            write_json(test_jsonl_data, test_json_file_path, DATASET_NAME)

            write_jsonl(train_jsonl_data, train_jsonl_file_path, DATASET_NAME)
            write_jsonl(test_jsonl_data, test_jsonl_file_path, DATASET_NAME)


if __name__ == "__main__":
    process_iemocap(
        "downloads/iemocap", output_format=["mini_format", "jsonl", "json", "split"]
    )
