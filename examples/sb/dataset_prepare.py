
import os
import sys
import re
import json
import random
import logging
from speechbrain.dataio.dataio import read_audio
import numpy as np

logger = logging.getLogger(__name__)
#SAMPLERATE = 16000
#NUMBER_UTT = 5531

def check_exists(data, feat_dir, logger):
    new_data = {}
    keys = data.keys()
    new_keys = []
    for key in data:
        sample = data[key]
        feat_path = os.path.join(feat_dir, key + '.npy')
        if os.path.exists(feat_path):
            new_keys.append(key)
    logger.info(f'load in {len(keys)} samples, only {len(new_keys)} exists in feature dir {feat_dir}')        
    new_data = {key:data[key] for key in new_keys}    
    return new_data

def prepare_data(
    train_annotation,
    valid_annotation,
    test_annotation,
    save_json_train,
    save_json_valid,
    save_json_test,
    label_map,
    feat_dir,
    split_ratio=[80, 20],
    seed=12,
):
    # setting seeds for reproducible code.
    random.seed(seed)

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return
    
    
    # split original train json into train and valid.
    train_data = json.load(open(train_annotation))
    test_data = json.load(open(test_annotation))
    
    train_data = check_exists(train_data, feat_dir, logger)
    test_data = check_exists(test_data, feat_dir, logger)

    num_train_data = len(train_data.keys())
    num_test_samples = len(test_data.keys())
    
    if os.path.isfile(valid_annotation):
        valid_data = json.load(open(valid_annotation))
        valid_data = check_exists(valid_data, feat_dir, logger)
    else:
        train_data, valid_data = split_sets(train_data, split_ratio)
        
    num_train_data = len(train_data.keys())
    num_valid_data = len(valid_data.keys())
    num_test_samples = len(test_data.keys())
    label_map = json.load(open(label_map))
    logger.info(f'Num. training samples {num_train_data}')
    logger.info(f'Num. valid samples {num_valid_data}')
    logger.info(f'Num. test samples {num_test_samples}')

    logger.info(f'Using {save_json_train} as training data')
    logger.info(f'Using {save_json_valid} as valid data')
    logger.info(f'Using {save_json_test} as test data')
    
    logger.info(f'Using label_map {label_map}')
    new_train_data = {}
    for key, values in train_data.items():
        emo = values['emo']
        values['emo'] = label_map[emo] 
        new_train_data[key] = values
    
    new_valid_data = {}
    for key, values in valid_data.items():
        emo = values['emo']
        values['emo'] = label_map[emo] 
        new_valid_data[key] = values
    
    new_test_samples = {}
    for key, values in test_data.items():
        emo = values['emo']
        values['emo'] = label_map[emo] 
        new_test_samples[key] = values

    with open(save_json_train, mode='w') as json_f:
        json.dump(new_train_data, json_f, indent=2)    
        json_f.close()
    
    with open(save_json_valid, mode='w') as json_f:
        json.dump(new_valid_data, json_f, indent=2)    
        json_f.close()

    with open(save_json_test, mode="w") as json_f:
        json.dump(new_test_samples, json_f, indent=2)
        json_f.close()


def split_sets(train_data, split_ratio):
    train_keys = list(train_data.keys())
    num_train_data = len(train_keys)
    num_train_nodev_samples = int(num_train_data * split_ratio[0])

    sample_idx = np.arange(num_train_data)
    random.shuffle(sample_idx)
    train_nodev_keys = [ train_keys[idx] for idx in sample_idx[:num_train_nodev_samples]]
    valid_keys = [train_keys[idx] for idx in sample_idx[num_train_nodev_samples:]]
    
    train_data = {key: train_data[key] for key in train_nodev_keys}
    valid_data= {key: train_data[key] for key in valid_keys}
    return train_data, valid_data
    


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True




