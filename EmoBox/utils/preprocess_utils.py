import json
import os
import torchaudio
from collections import defaultdict, Counter
import random
from scipy import special

def validate_jsonl_data(output_jsonl_file_path, jsonl_data):
    """Check if JSONL data is the same as existing data in file."""
    with open(output_jsonl_file_path, "r") as jsonl_file:
        existing_jsonl_data = [json.loads(line) for line in jsonl_file]
    # TODO (hz): This is not complete. I will need to comlete this function.
    return existing_jsonl_data == jsonl_data

def get_relative_audio_path(file_path):
    parts = file_path.split("/")
    downloads_index = parts.index("downloads")
    relative_path = "/".join(parts[downloads_index:])
    return relative_path

def load_audio(file_path):
    """Load audio file and return the waveform and sample rate."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def split_audio_torchaudio(waveform, sample_rate, start_time, end_time, output_path):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    split_waveform = waveform[:, start_sample:end_sample]
    torchaudio.save(output_path, split_waveform, sample_rate)

def write_mini_format(data, output_base_dir):
    """Output the data in the mini format."""
    audio_map_file = os.path.join(output_base_dir, "data.seg.audmap")
    stm_file = os.path.join(output_base_dir, "data.seg.wrd.stm")
    freq_file = os.path.join(output_base_dir, "data.seg.wrd.freq")

    audio_map = open(audio_map_file, 'w')
    stm_file = open(stm_file, 'w')
    freq_file = open(freq_file, 'w')
    emotion_freq = defaultdict(int)

    for key, value in data.items():
        audio_map_output = f"{key} {value['channel']} {value['start_time']} {value['end_time']} {value['audio']}\n"
        audio_map.write(audio_map_output)

        parsed_info = {
            "id": key,
            "lang": "en",
            "spk": value['spk'],
            "emotion": value['emotion'],
        }

        parsed_info_str = ",".join([f"{k}={v}" for k, v in parsed_info.items()])
        meta_info = f"<{parsed_info_str}>"
        stm_output = f"{key} {value['channel']} {value['spk']} {value['start_time']} {value['end_time']} {meta_info} {value['emotion']}\n"
        stm_file.write(stm_output)

        emotion_freq[value['emotion']] += 1

    max_key_length = max(len(key) for key in emotion_freq.keys())
    space_between = 2  # Space between the key and the value

    for emotion_freq_key, emotion_freq_value in emotion_freq.items():
        # Write the emotion frequency to the file
        line = f"{emotion_freq_key.ljust(max_key_length + space_between)}{emotion_freq_value}\n"
        freq_file.write(line)

    stm_file.close()
    audio_map.close()
    freq_file.close()


def write_jsonl(data, output_jsonl_file_path, dataset_name):
    """Output the data in the JSONL format."""
    jsonl_data = []
    for key, value in data.items():
        jsonl_entry = {
            "key": data[key]["sid"],
            "dataset": dataset_name,
            "wav": data[key]["audio"],
            "type": "raw",
            "sample_rate": data[key]["sample_rate"],
            "num_frame": (data[key]["end_time"]-data[key]["start_time"]) * data[key]["sample_rate"],
            "task": "category",
            "length": data[key]["duration"],
            "emo": data[key]["emotion"],
            "channel": data[key]["channel"],
        }
        if dataset_name == "pavoque":
            jsonl_entry["start_time"] = data[key]["start_time"]
            jsonl_entry["end_time"] = data[key]["end_time"]
            
        jsonl_data.append(jsonl_entry)
    with open(output_jsonl_file_path, "w") as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry) + "\n")


def write_json(data, output_json_file_path, dataset_name):
    """Output the data in the JSON format."""
    json_data = {}
    for key, value in data.items():
        json_data[key] = {
            "wav": data[key]["audio"],
            "length": data[key]["duration"],
            "emo": data[key]["emotion"],
            "dataset": dataset_name,
            "channel": data[key]["channel"],
        }
        if dataset_name == "pavoque":
            json_data[key]["start_time"] = data[key]["start_time"]
            json_data[key]["end_time"] = data[key]["end_time"]
    with open(output_json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
        

def write_folds(data, output_base_dir, dataset_name):
    """Process and split data into folds."""
    speaker_data = defaultdict(list)
    speaker_emo_count = defaultdict(Counter)
    emo_count = defaultdict(list)
    
    for key in data:
        speaker_data[data[key]["spk"]].append(key)
        emo_count[data[key]["emotion"]].append(key)
        speaker_emo_count[data[key]["spk"]][data[key]["emotion"]] += 1
        
    # is the emotion distribution balanced
    speaker_emo_count_set = set()
    emotions = emo_count.keys()
    for speaker in speaker_emo_count:
        speaker_emo_count_set.add(tuple(speaker_emo_count[speaker][emotion] for emotion in emotions))
    is_balanced = len(speaker_emo_count_set) == 1
        
    print(f"Dataset: {dataset_name}, Speaker count: {len(speaker_data)}, Emotions: {emotions}, Balanced: {is_balanced}")
    
    # If the dataset has less than 4 speakers or the emotion distribution is not balanced, disregard speaker and stratified sampling 25% as test set 1fold
    if len(speaker_data) < 4 or is_balanced==False:
        os.makedirs(os.path.join(output_base_dir, "fold_1"), exist_ok=True)
        random.seed(0)
        # Disregard speaker, stratified sampling 25% as test set 1fold
        test_keys = []
        train_keys = []

        # Stratified sampling 25% as test set
        for emotion, keys in emo_count.items():
            # Calculate the number of keys to sample for this emotion
            num_test_keys = int(len(keys) * 0.25)
            
            # Randomly select the test keys for this emotion
            test_keys.extend(random.sample(emo_count[emotion], num_test_keys))
            
            # Add the remaining keys to the train set
            train_keys.extend([key for key in emo_count[emotion] if key not in test_keys])

        # Create the train and test data dictionaries
        train_data = {key: data[key] for key in train_keys}
        test_data = {key: data[key] for key in test_keys}

        # Define the file paths for train and test data
        train_jsonl_file_path = os.path.join(output_base_dir, "fold_1", f"{dataset_name}_train_fold_1.jsonl")
        test_jsonl_file_path = os.path.join(output_base_dir, "fold_1", f"{dataset_name}_test_fold_1.jsonl")

        train_json_file_path = os.path.join(output_base_dir, "fold_1", f"{dataset_name}_train_fold_1.json")
        test_json_file_path = os.path.join(output_base_dir, "fold_1", f"{dataset_name}_test_fold_1.json")

        # Write the train and test data to files
        write_json(train_data, train_json_file_path, dataset_name)
        write_json(test_data, test_json_file_path, dataset_name)

        write_jsonl(train_data, train_jsonl_file_path, dataset_name)
        write_jsonl(test_data, test_jsonl_file_path, dataset_name)
    
    # If the dataset has 4-6 speakers, use speaker stratified sampling to create 4-6 folds
    elif len(speaker_data) in [4, 5, 6]:
        for fold_number, (test_spk, test_keys) in enumerate(speaker_data.items()):
            os.makedirs(os.path.join(output_base_dir, f"fold_{fold_number + 1}"), exist_ok=True)
            train_keys = []
            for spk, keys in speaker_data.items():
                if spk != test_spk:
                    train_keys.extend(keys)
            
            train_data = {key: data[key] for key in train_keys}
            test_data = {key: data[key] for key in test_keys}
            
            train_jsonl_file_path = os.path.join(output_base_dir, f"fold_{fold_number + 1}", f"{dataset_name}_train_fold_{fold_number + 1}.jsonl")
            test_jsonl_file_path = os.path.join(output_base_dir, f"fold_{fold_number + 1}", f"{dataset_name}_test_fold_{fold_number + 1}.jsonl")
            
            train_json_file_path = os.path.join(output_base_dir, f"fold_{fold_number + 1}", f"{dataset_name}_train_fold_{fold_number + 1}.json")
            test_json_file_path = os.path.join(output_base_dir, f"fold_{fold_number + 1}", f"{dataset_name}_test_fold_{fold_number + 1}.json")
            
            write_json(train_data, train_json_file_path, dataset_name)
            write_json(test_data, test_json_file_path, dataset_name)
            
            write_jsonl(train_data, train_jsonl_file_path, dataset_name)
            write_jsonl(test_data, test_jsonl_file_path, dataset_name)
            
    elif len(speaker_data) > 6:
        folds = [[] for _ in range(len(speaker_data) // 4)]
        speakers = list(speaker_data.keys())
        random.shuffle(speakers) 
        for i, speaker in enumerate(speakers):
            folds[i % len(folds)].append(speaker)
        
        for fold_number, test_spks in enumerate(folds):
            os.makedirs(os.path.join(output_base_dir, f"fold_{fold_number + 1}"), exist_ok=True)
            
            train_data = {}
            test_data = {}
            
            for spk, keys in speaker_data.items():
                if spk in test_spks:
                    test_data.update({key: data[key] for key in keys})
                else:
                    train_data.update({key: data[key] for key in keys})
            
            train_jsonl_file_path = os.path.join(output_base_dir, f"fold_{fold_number + 1}", f"{dataset_name}_train_fold_{fold_number + 1}.jsonl")
            test_jsonl_file_path = os.path.join(output_base_dir, f"fold_{fold_number + 1}", f"{dataset_name}_test_fold_{fold_number + 1}.jsonl")
            
            test_json_file_path = os.path.join(output_base_dir, f"fold_{fold_number + 1}", f"{dataset_name}_test_fold_{fold_number + 1}.json")
            train_json_file_path = os.path.join(output_base_dir, f"fold_{fold_number + 1}", f"{dataset_name}_train_fold_{fold_number + 1}.json")
            
            write_jsonl(train_data, train_jsonl_file_path, dataset_name)
            write_jsonl(test_data, test_jsonl_file_path, dataset_name)
            
            write_json(train_data, train_json_file_path, dataset_name)
            write_json(test_data, test_json_file_path, dataset_name)
            

            
            
