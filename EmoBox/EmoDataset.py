import os
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
import logging
import torchaudio
SAMPLING_RATE=16000
logger = logging.getLogger(__name__)



"""
"Ses01M_impro01_F000": {
	key: "Ses01M_impro01_F000"
	dataset: "iemocap"
	audio: "Session1/sentences/wav/Ses01M_impro01/Ses01M_impro01_F000.wav"
    type: "raw" # raw, feature
	sample_rate: 16000
	num_frame: 32000
	task: "category" # category, valence, arousal
	label: "hap"
	} 
"Ses01M_impro01_F001": {...}
...
"""
def check_exists(data, data_dir, logger):
    new_data = []
    for instance in data:
        audio_path = instance['wav']
        
        if os.path.exists(audio_path):
            new_data.append(instance)
    logger.info(f'load in {len(data)} samples, only {len(new_data)} exists in data dir {data_dir}')        
    return new_data

def replace_label(data, label_map, logger):
    new_data = []
    for instance in data:
        emotion = instance['emo']
        label = label_map[emotion]
        instance['emo'] = label
        new_data.append(instance)
    return new_data    

def prepare_data(
    dataset,
    data_dir,
    meta_data_dir,
    label_map,
    meta_format='jsonl',
    fold=1,
    split_ratio=[80, 20],
    seed=12,
):
    # setting seeds for reproducible code.
    random.seed(seed)

    
    # find train/valid/test metadata files
    train_data_path = os.path.join(data_dir, dataset, 'fold'+fold, f'{dataset}_train_fold_{fold}.{meta_format}')
    test_data_path = os.path.join(data_dir, dataset, 'fold'+fold, f'{dataset}_test_fold_{fold}.{meta_format}')
    valid_data_path = os.path.join(data_dir, dataset, 'fold'+fold, f'{dataset}_valid_fold_{fold}.{meta_format}')
    
    # check existance
    assert os.path.exists(train_data_path)
    assert os.path.exists(test_data_path)
    official_valid = False
    if os.path.exists(valid_data_path):
        logger.info(f'using official valid data in {valid_data_path}')
        official_valid = True
    else:
        logger.info(f'since there is no official valid data, use random split for train valid split, with a ratio of {split_ratio}')    

    # load in train & test data
    train_data = []
    test_data = []
    with open(train_data_path) as f:
        for line in f:
            train_data.append(json.loads(f).strip())
    with open(test_data_path) as f:
        for line in f:
            test_data.append(json.loads(f).strip())
    if official_valid:
        with open(valid_data_path) as f:
            for line in f:
                valid_data.append(json.loads(f).strip())
            
            

    train_data = check_exists(train_data, data_dir, logger)
    test_data = check_exists(test_data, data_dir, logger)
    if official_valid:
        valid_data = check_exists(valid_data, data_dir, logger)
    else:
        train_data, valid_data = split_sets(train_data, split_ratio)
        
    num_train_data = len(train_data)
    num_valid_data = len(valid_data)
    num_test_samples = len(test_data)
    label_map = json.load(open(label_map))
    logger.info(f'Num. training samples {num_train_data}')
    logger.info(f'Num. valid samples {num_valid_data}')
    logger.info(f'Num. test samples {num_test_samples}')

    
    logger.info(f'Using label_map {label_map}')
    train_data = replace_label(train_data, label_map, logger)
    valid_data = replace_label(valid_data, label_map, logger)
    test_data = replace_label(test_data, label_map, logger)
    
    return train_data, valid_data, test_data

def split_sets(train_data, split_ratio):
    num_train_data = len(train_data)
    num_train_nodev_samples = int(num_train_data * split_ratio[0])

    sample_idx = np.arange(num_train_data)
    random.shuffle(sample_idx)
    train_nodev_data = [ train_data[idx] for idx in sample_idx[:num_train_nodev_samples]]
    valid_data = [train_data[idx] for idx in sample_idx[num_train_nodev_samples:]]
    
    return train_data, valid_data

def read_wav(data):
    wav_path = data['wav']
    channel = data['channel']
    dur = float(data['length'])
    if 'start_time' in data and 'end_time' in data:
        start_time = data['start_time']
        end_time = data['end_time']
    else:
        start_time = None
        end_time = None    
    if start_time is not None and end_time is not None:
        sample_rate = torchaudio.info(wav_path).sample_rate
        frame_offset = int(start_time * sample_rate)
        num_frames = int(end_time * sample_rate) - frame_offset
        wav, sr = torchaudio.load(wav_path, frame_offset = frame_offset, num_frames = num_frames)
    else:    
        wav, sr = torchaudio.load(wav_path)
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
    wav = wav.view(-1)    
    return wav 

class EmoDataset(Dataset):
    def __init__(self, dataset, data_dir, meta_data_dir, fold=1, split="train", meta_format = 'jsonl'):
        super().__init__()
        self.data_dir = data_dir
        train_data, valid_data, test_data = prepare_data(dataset, data_dir, meta_data_dir, meta_format = meta_format, fold = fold)
        if split == 'train':
            self.data_list = train_data
        elif split == 'valid':
            self.data_list = valid_data
        elif split == 'test':
            self.data_list = test_data
        else:
            raise Exception(f'does not support split {split}')        
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        audio = os.path.join(self.data_dir, data["wav"])
        if not os.path.exists(audio):
            raise FileNotFoundError(f"{audio} does not exist.")
        
        if data["type"] == "raw":
            
            audio = read_wav(data)
        elif data["type"] == "feature":
            audio = np.load(audio)
        else:
            raise ValueError(f"Unknown data type: {data['type']}")
        label = data['emo']        
        return{
            "audio": audio,
            "label": label,
            # other meta data can be added here
        }

        