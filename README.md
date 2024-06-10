# EmoBox

# Quick start

```# Example:
from EmoBox import EmoDataset, EmoEval
for fold in folds:
	train = EmoDataset(user_data_dir, meta_data_dir, fold=1, split="train")
	val, test = xxx
	dataloader = xxx
	
	"""training"""
	pred = model(test)
	WA, UA, F1 = EmoEval
	WAs.append(WA)
	xxx ...
```
# Datasets
| Dataset | Language | Download link |
| :---: | :---: | :---: |
| 单元格1 | 单元格2 | 单元格3 |
| 单元格4 | 单元格5 | 单元格6 |


# Evaluation setup of track1 and track2

## track1

| Dataset | Num. of folds |
| :---: | :---: |
| 单元格1 | 单元格2 |
| 单元格4 | 单元格5 |


# Example recipe

We provide a speechbrain recipe for using EmoDataset and EmoEval

```
	cd examples/sb
	conda create -n conda-env -f environment.yml
	conda activate conda-env
	
	# extract features
	python3 speech_feature_extraction.py
	
	# run training & evaluation
	python3 train.py
```

# Reference



