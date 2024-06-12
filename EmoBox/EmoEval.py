import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score
eps=1e-12	# to prevent dividing by zero

def ComputePerformance(ref_id, hyp_id):
    score = dict()
 
    num = len(ref_id)
    score['num'] = num
    score['overallWA'] = accuracy_score(ref_id, hyp_id)
    score['overallUA'] = balanced_accuracy_score(ref_id, hyp_id)
    score['overallMicroF1'] = f1_score(ref_id, hyp_id, average = 'micro')
    score['overallMacroF1'] = f1_score(ref_id, hyp_id, average = 'macro')
    score['report'] = classification_report(ref_id, hyp_id)
    score['confusion'] = confusion_matrix(ref_id, hyp_id)
    return score
###### Write scores into file    
def WriteScore(f, score ):
    classification = 'Emotion Recognition'
    K = score['num']
    f.write('%sScoring -- Sample [%d], Overall UA     %.4f\n' % (classification, K, score['overallUA']))
    f.write('%sScoring -- Sample [%d], Overall WA     %.4f\n' % (classification, K, score['overallWA']))
    f.write('%sScoring -- Sample [%d], Overall Micro-F1     %.4f\n' % (classification, K, score['overallMicroF1']))
    f.write('%sScoring -- Sample [%d], Overall Macro-F1     %.4f\n' % (classification, K, score['overallMacroF1']))
    f.write('\n')
    f.write(score['report'])
    f.write('\n')
    confusion_matrix = score['confusion']
    f.write(f'confusion_matrix: \n {confusion_matrix}')


class EmoEval():
    def __init__(self, predictions, data_labels):
        """
            predictions: [{"key": "xxx", pred":1}, {"key":"yyy",pred":2},...]
            data_labels: [{"key":"xxx",label":1}, {"key":"yyy",label":3}, ...]
        """
        assert len(predictions) == len(data), f'the number of predictions shoud be equal to labels'
        for pred, label in zip(predictions, data_labels):
            pred_key = pred['key']
            label_key = label['key']
            assert pred_key == label_key, f'prediction and label should have the same key, while prediction has a key {pred_key}, label has a key {label_key}' 
        
        self.targets = []
        for instance in dataset:
            label = instance['label']
            self.targets.append(label)
        
        self.predictions = []

        for instance in predictions:
            pred = instance['prediction']
            self.predictions.append(preed)
    def compute_metrics(self,):
        scores = ComputePerformance(self.targets, self.predictions) 
        return scores
    
    def write_scores(self, path, scores)
        f = open(path, 'w')
        WriteScore(f, scores)
            
        

              


        
