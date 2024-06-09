import torch
import os
import sys
import json
import numpy as np
import argparse
from tqdm import tqdm
import torchaudio
import torch.nn.functional as F
SAMPLING_RATE=16000

from fairseq_hubert import FairseqHubert

# We check if transformers is installed.
try:
    import transformers
    from transformers import AutoModel
    from transformers import Wav2Vec2Model, HubertModel, WavLMModel
    from transformers import Wav2Vec2Config, HubertConfig, WavLMConfig
    from transformers import AutoFeatureExtractor, AutoProcessor
    from transformers import Wav2Vec2ForPreTraining
    from transformers import Data2VecAudioModel, Data2VecAudioConfig
    from transformers import WhisperFeatureExtractor, WhisperForAudioClassification, WhisperConfig
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        _compute_mask_indices,
    )

except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

HF_models = {
    "wav2vec2": Wav2Vec2Model,
    "wavlm": WavLMModel,
    "data2vec": Data2VecAudioModel,
    "whisper": WhisperForAudioClassification
}

HF_config = {
    "wav2vec2": Wav2Vec2Config,
    "wavlm": WavLMConfig,
    "data2vec": Data2VecAudioConfig,
    "whisper": WhisperConfig

}

def load_model(source, model_path, device):
    # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
    if "wavlm" in source:
        config = HF_config.get("wavlm")
        model = HF_models.get("wavlm")
        processor = AutoFeatureExtractor.from_pretrained(source, cache_dir = model_path)
    elif "data2vec" in source:
        config = HF_config.get("data2vec")
        model = HF_models.get("data2vec")    
        processor = AutoFeatureExtractor.from_pretrained(source, cache_dir = model_path)
    elif 'wav2vec2' in source:
        config = HF_config.get("wav2vec2")
        model = HF_models.get("wav2vec2")
        processor = AutoFeatureExtractor.from_pretrained(source, cache_dir = model_path)
    elif 'whisper' in source:
        config = HF_config.get("whisper")
        model = HF_models.get("whisper")
        processor = WhisperFeatureExtractor.from_pretrained(source)

            
        
    model = model.from_pretrained(
        source, cache_dir=model_path
    )
    model.eval()
    model.to(device)
    return (model, processor)

def extract_whisper_feature(wav_path, channel, model, output_norm, all_layers,device, start_time = None, end_time = None):
    model, processor = model
    if start_time is not None and end_time is not None:
        sample_rate = torchaudio.info(wav_path).sample_rate
        frame_offset = int(start_time * sample_rate)
        num_frames = int(end_time * sample_rate) - frame_offset
        wav, sr = torchaudio.load(wav_path, frame_offset = frame_offset, num_frames = num_frames)
    else:    
        wav, sr = torchaudio.load(wav_path)
    channel = channel -1
    wav = wav[channel, :]
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
    wav = wav.view(-1)
    feat_len = int(wav.size(0) // 320)
    input_features = processor(wav, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features
    padding_mask = torch.ones_like(input_features)
    input_features = input_features.to(model.device)

    out = model(input_features, output_hidden_states = True)

    if all_layers:
        out = torch.stack(list(out.hidden_states), dim=0)
        out = out[:,:,:feat_len,:]
        norm_shape = out.shape[-3:]
    else:
        out = out.hidden_states[-1]
        out = out[:,:feat_len,:]
        norm_shape = out.shape

    if output_norm:
        out = F.layer_norm(out, norm_shape[1:])

    return out


def extract_fairseq_feature(wav_path, channel, model, output_norm, all_layers, device, start_time = None, end_time = None):
    if start_time is not None and end_time is not None:
        sample_rate = torchaudio.info(wav_path).sample_rate
        frame_offset = int(start_time * sample_rate)
        num_frames = int(end_time * sample_rate) - frame_offset
        wav, sr = torchaudio.load(wav_path, frame_offset = frame_offset, num_frames = num_frames)
    else:    
        wav, sr = torchaudio.load(wav_path)
    channel = channel -1
    wav = wav[channel, :]
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
    wav = wav.view(1, -1)    
    wav = wav.to(device)
    
    out = model.extract_features(wav)
    return out


def extract_huggingface_feature(wav_path, channel, model, output_norm, all_layers, device, start_time = None, end_time = None):
    if start_time is not None and end_time is not None:
        sample_rate = torchaudio.info(wav_path).sample_rate
        frame_offset = int(start_time * sample_rate)
        num_frames = int(end_time * sample_rate) - frame_offset
        wav, sr = torchaudio.load(wav_path, frame_offset = frame_offset, num_frames = num_frames)
    else:    
        wav, sr = torchaudio.load(wav_path)
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
    channel = channel -1
    wav = wav[channel, :]
    model, processor = model
    inputs = processor(wav, sampling_rate = SAMPLING_RATE, return_tensors="pt")
    inputs = inputs.to(device)

    out = model(**inputs, output_hidden_states = all_layers)

    if all_layers:
        out = torch.stack(list(out.hidden_states), dim=0)
        norm_shape = out.shape[-3:]
    else:
        out = out.last_hidden_state
        norm_shape = out.shape

    if output_norm:
        out = F.layer_norm(out, norm_shape[1:])

    return out
    
def get_source(model_name):
    if model_name.startswith('wavlm'):
        source = f'microsoft/{model_name}'
    elif model_name == 'wav2vec2-base':
        source = 'facebook/wav2vec2-base'
    elif model_name == 'wav2vec2-large':
        source = 'facebook/wav2vec2-large'
    elif model_name == 'hubert-base':
        source = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt'        
    elif model_name == 'hubert-large':
        source = 'https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt'    
    elif model_name == 'data2vec-base':
        source = 'facebook/data2vec-audio-base'
    elif model_name == 'data2vec-large':
        source = 'facebook/data2vec-audio-large'
    elif model_name == 'data2vec2-base':
        source = 'https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_libri.pt'
    elif model_name == 'data2vec2-large':
        source = 'https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_vox.pt'        
    elif model_name == 'whisper-large-v3':
        source = 'openai/whisper-large-v3'    
    else:
        source = f'facebook/{model_name}'
    return source



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type = str, required = True) 
    parser.add_argument('--model_path', type = str, required = True)
    parser.add_argument('--dump_dir', type = str, required = True)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--data', type =str, required = True)
    parser.add_argument('--all_layers', default = False, action = 'store_true')
    parser.add_argument('--output_norm', default = False, action = 'store_true')
    
    args = parser.parse_args()
    print(args)

    # load metadata
    f = open(args.data)
    data = json.load(f)
    f.close()
    
    seg_ids = data.keys()
    print(f'load in {len(seg_ids)} segments')

    # load models
    source = get_source(args.model_name)
    if 'hubert' in args.model_name:
        model = FairseqHubert(source, os.path.join(args.model_path, args.model_name+'.pt'), output_norm = args.output_norm, freeze = True, freeze_feature_extractor = True)    
        model.model.to(args.device)
        feat_func = extract_fairseq_feature
    elif 'whisper' in args.model_name:
        model = load_model(source, args.model_path, args.device)
        feat_func = extract_whisper_feature    
    else:        
        model = load_model(source, args.model_path, args.device)
        feat_func = extract_huggingface_feature
    
    # load speech ssl models
    for seg_id in tqdm(seg_ids):
        sample = data[seg_id]
        wav_path = sample['wav']
        channel = sample['channel']
        dur = float(sample['length'])
        if dur > 30. :
            print(f"SKIP {wav_path} because its duration is {dur}, which is too long!")
            continue

        if 'start_time' in sample and 'end_time' in sample:
            start_time = sample['start_time']
            end_time = sample['end_time']
        else:
            start_time = None
            end_time = None    
        assert os.path.exists(wav_path), f'{wav_path} does not exists on your disk'
        try:
            torchaudio.load(wav_path)
        except:
            print(f'ERROR!! wav file {wav_path} can not be loaded!')
            continue   
        feat = feat_func(wav_path, channel, model, args.output_norm, args.all_layers, args.device, start_time, end_time)
        feat = feat.data.cpu().numpy()
        if args.all_layers:
            feat = np.squeeze(feat, 1)
        else:
            feat = feat[0]    
        save_path = os.path.join(args.dump_dir, seg_id + '.npy')
        print(f'seg_id:{seg_id}\tfeat_shape:{feat.shape}\tsave_path:{save_path}')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        np.save(save_path, feat)





