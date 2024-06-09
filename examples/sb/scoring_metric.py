import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score


eps=1e-12	# to prevent dividing by zero


###### INTERFACE to speechbrain train_with_wav2vec.py

def go_through_error_metrics(error_metrics):
    targets_id = []
    prediction_id = []
    for batch_id, pred, emoid in error_metrics:
        targets_id.append(emoid)
        prediction_id.append(np.argmax(pred, 1))
    
    prediction_id = np.concatenate(prediction_id, axis = 0)
    targets_id = np.concatenate(targets_id, axis = 0)
    return prediction_id, targets_id    

def scoring_all(error_metrics):
    prediction_id, target_id = go_through_error_metrics(error_metrics)
    scores = ComputePerformance(target_id, prediction_id)
    num_samples = prediction_id.shape[0]
    return scores, num_samples

def scoring_ua_wa(error_metrics):
    scores, _ = scoring_all(error_metrics)
    wa = scores['overallWA']
    ua = scores['overallUA']
    micro_f1 = scores['overallMicroF1']
    macro_f1 = scores['overallMacroF1']
    return ua, wa, micro_f1, macro_f1

def output_score(scores, path, epoch, num_samples, dataset_name):        
    f = open(path,'w')
    WriteScore(f, scores, epoch, num_samples)
    f.close()

################## Functions to compute scores    


def ComputePerformance(ref_id, hyp_id):
    score = dict()
 

    score['overallUA'] = accuracy_score(ref_id, hyp_id)
    score['overallWA'] = balanced_accuracy_score(ref_id, hyp_id)
    score['overallMicroF1'] = f1_score(ref_id, hyp_id, average = 'micro')
    score['overallMacroF1'] = f1_score(ref_id, hyp_id, average = 'macro')
    score['report'] = classification_report(ref_id, hyp_id)
    score['confusion'] = confusion_matrix(ref_id, hyp_id)
    return score
###### Write scores into file    
def WriteScore(f, score, epoch, K):
    classification = 'Emotion Recognition'
    lbl = 'MPS-Podcast'
    f.write('DATASET -- %s\n' % lbl)
    f.write(f"{classification}, {epoch}, {K}\n")
    f.write('%sScoringv3 -- Epoch [%d], Sample [%d], Overall UA     %.4f\n' % (classification, epoch, K, score['overallUA']))
    f.write('%sScoringv3 -- Epoch [%d], Sample [%d], Overall WA     %.4f\n' % (classification, epoch, K, score['overallWA']))
    f.write('%sScoringv3 -- Epoch [%d], Sample [%d], Overall Micro-F1     %.4f\n' % (classification, epoch, K, score['overallMicroF1']))
    f.write('%sScoringv3 -- Epoch [%d], Sample [%d], Overall Macro-F1     %.4f\n' % (classification, epoch, K, score['overallMacroF1']))
    f.write('\n')
    f.write(score['report'])
    f.write('\n')
    confusion_matrix = score['confusion']
    f.write(f'confusion_matrix: \n {confusion_matrix}')

