import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score
eps=1e-12	# to prevent dividing by zero

def ComputePerformance(ref_id, hyp_id):
    score = dict()
 

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
    f.write('%sScoringv3 -- Epoch [%d], Sample [%d], Overall UA     %.4f\n' % (classification, epoch, K, score['overallUA']))
    f.write('%sScoringv3 -- Epoch [%d], Sample [%d], Overall WA     %.4f\n' % (classification, epoch, K, score['overallWA']))
    f.write('%sScoringv3 -- Epoch [%d], Sample [%d], Overall Micro-F1     %.4f\n' % (classification, epoch, K, score['overallMicroF1']))
    f.write('%sScoringv3 -- Epoch [%d], Sample [%d], Overall Macro-F1     %.4f\n' % (classification, epoch, K, score['overallMacroF1']))
    f.write('\n')
    f.write(score['report'])
    f.write('\n')
    confusion_matrix = score['confusion']
    f.write(f'confusion_matrix: \n {confusion_matrix}')


class EmoEval():
    def __init__(self, predictions, data):
        """
            predictions: [{"pred":'hap'}, {"pred":'ang'},...]
            data: [{"label":'hap'}, {"label":'sad'}, ...]
        """
        assert len(predictions) == len(data), f'the number of predictions shoud be equal to data'
        labels = []
        for instance in data:
            label = instance["label"]
            if label not in labels:
                labels.append(label)
        label2idx = {label:labels.index(label) for label in labels}
        
        self.targets = []
        for instance in dataset:
            label = instance['label']
            idx = self.label2idx['label']
            self.targets.append(idx)
        
        self.predictions = []

        for instance in predictions:
            pred = instance['prediction']
            if not pred in label2idx:
                raise Exception(f'prediction {pred} does not exists in dataset')  
            idx = self.label2idx[pred]
            self.predictions.append(idx)
    def compute_metrics(self,):
        scores = ComputePerformance(self.targets, self.predictions) 
        return scores
    
    def print_scores(self, path, scores)
        f = open(path, 'w')
        WriteScore(f, scores)
            
        

              


        
