import os
import pandas as pd
import json
import re
from collections import defaultdict

pattern = re.compile(r'<(.*?)>')

rows_list = []


datasets = [d for d in os.listdir("data") if os.path.isdir(f"data/{d}")]

datasets.sort()

for dataset in datasets:
    dataset_path = f"data/{dataset}"
    
    with open(f"{dataset_path}/{dataset}.json") as f:
        data = json.load(f)
        
    total_duration = 0
    speakers = set()
    speakers_dict = {}
    emotions_count = defaultdict(int)
    
    for k, v in data.items():
        total_duration += v["length"]
        emotions_count[v["emo"]] += 1
        
    with open(f"{dataset_path}/data.seg.wrd.stm") as f:
        for line in f.readlines():
            match = pattern.search(line)
            if match:
                content = match.group(1)
                parts = content.split(',') 
                id_value = spk_value = ''
                for part in parts:
                    if part.startswith('id='):
                        id_value = part.split('=')[1]
                    elif part.startswith('spk='):
                        spk_value = part.split('=')[1]

                if id_value and spk_value:
                    speakers.add(spk_value)
                    speakers_dict[id_value] = spk_value
                else:
                    print(f"Error: {line}")

    row = {
        "dataset name": dataset,
        "language": "", 
        "Source": "",  
        "speaker": len(speakers),
        "emo & number": dict(emotions_count),
        "hours": total_duration / 3600,
        "utterance": len(data),
    }
    if dataset in ["mer2023"]:
        rows_list.append(row)
        continue
    for fold in range(1, 10):
        fold_path = f"{dataset_path}/fold_{fold}"
        if not os.path.exists(fold_path):
            break
        with open(f"{fold_path}/{dataset}_test_fold_{fold}.json") as f:
            fold_data = json.load(f)
        fold_speakers = set()
        fold_emotion_count = defaultdict(int)
        for k, v in fold_data.items():
            fold_speakers.add(speakers_dict[k])
            fold_emotion_count[v["emo"]] += 1
        
        fold_speakers_classes = f"{len(fold_speakers)} + {dict(fold_emotion_count)}"
        row[f"fold {fold} spk + classes"] = fold_speakers_classes
    row['fold number'] = fold - 1  # 最后一个存在的fold
    rows_list.append(row)

df = pd.DataFrame(rows_list)
df.sort_values(by='dataset name', inplace=True)

df.to_excel("metadata.xlsx", index=False)

for index, row in df.iterrows():
    print('\t'.join(str(x) for x in row.values))
