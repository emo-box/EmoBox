import os
import numpy as np
import argparse
import sys
from dataclasses import dataclass
import soundfile as sf

def get_parser():
    parser = argparse.ArgumentParser(
        description="extract emotion2vec features for downstream tasks"
    )
    parser.add_argument('--source_path', help='location of source wav files', required=True)
    parser.add_argument('--target_path', help='location of target npy files', required=True)
    parser.add_argument('--fairseq_root', help='location of fairseq root', required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', required=True)
    parser.add_argument('--user_module_path', type=str, default='emotion2vec', help='user module path', required=True)
    parser.add_argument('--model_name', type=str, default='example/data2vec', help='pretrained model name', required=True)
    parser.add_argument('--layer', type=int, default=7, help='which layer to use', required=True)
    parser.add_argument('--dataset_name', type=str, default='iemocap', help='which dataset to use', required=True)
    parser.add_argument('--granularity', type=str, help='which granularity to use, frame or utterance', required=True)
    

    return parser

@dataclass
class UserDirModule:
    user_dir: str

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    source_dir = args.source_path
    target_dir = args.target_path
    fairseq_root = args.fairseq_root
    checkpoint_dir = args.checkpoint_dir
    user_module_path = args.user_module_path
    model_name = args.model_name
    layer = args.layer
    dataset_name = args.dataset_name

    sys.path.append(fairseq_root)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import fairseq
    from fairseq import checkpoint_utils
    model_path = UserDirModule(user_module_path)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.cuda()

    # DFS all wav files
    for dirpath, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if filename.endswith('.wav'):
                source_file_path = os.path.join(dirpath, filename)
                target_file_path = os.path.join(target_dir, os.path.relpath(source_file_path, source_dir))
                target_file_path = target_file_path.replace('.wav', '.npy')

                # make sure the target directory exists
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

                # extract features
                wav, sr = sf.read(source_file_path)
                if dataset_name == 'meld':
                    wav = wav[:, 2] if wav.shape[1] > 2 else wav[:, 0]
                assert sr == 16e3, "Sample rate should be 16kHz, but got {}".format(sr)
                with torch.no_grad():
                    source = torch.from_numpy(wav).float().cuda()
                    assert source.dim() == 1, "Souce dim should be 1, but got {}".format(source.dim())
                    if task.cfg.normalize:
                        source = F.layer_norm(source, source.shape)
                    source = source.view(1, -1)
                    feats = model.extract_features(source, padding_mask=None, layer=layer)
                    feats = feats['x'].squeeze(0).cpu().numpy()
                    if granularity == 'frame':
                        feats = feats
                    elif granularity == 'utterance':
                        feats = np.mean(feats, axis=0)
                    else:
                        raise ValueError("Unknown granularity: {}".format(args.granularity))
                    np.save(target_file_path, feats)


if __name__ == '__main__':
    main()