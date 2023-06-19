import numpy as np

def determine_correctness(y_pred, y_true):
  """
  Given a list of predictions and a list of true values, create a list that contains for each position whether the prediction was correct.
  
  param y_pred: (list) containing predicted sentiment labels.
  param y_true: (list) containing actual sentiment labels.
  """
  correct = []
  i = 0
  while i < len(y_pred):
    if y_pred[i] == y_true[i]:
      correct.append(1)
    else:
      correct.append(0)
    i += 1
  return correct

def calc_accuracy_vanilla(scores):
  """
  Given a list with scores, determine the percentage of instances that was predicted correctly.
  
  param scores: (list) containing whether each prediction was correct.
  """
  correct = 0
  total = len(scores)
  for val in scores:
    if val == 1:
      correct += 1
  accuracy = round(correct/total*100, 4)
  return accuracy

def calc_base_metrics(y_pred, y_real, focus):
  """
  Calculate the TP, FP, TN and FN values with the focus on one specific class.
  
  param y_pred: (list) containing predicted sentiment labels.
  param y_real: (list) containing actual sentiment labels.
  param focus: (str) determines the class from which we evaluate.
  """
  metrics = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
  i = 0
  while i < len(y_pred):
      if focus == 'pos':
        if (y_pred[i] == 2) and (y_real[i] == 2):
            metrics['TP'] += 1
        elif (y_pred[i] != y_real[i]) and (y_real[i] == 2):
            metrics['FN'] += 1
        elif (y_pred[i] != y_real[i]) and (y_pred[i] == 2):
            metrics['FP'] += 1
        else:
            metrics['TN'] += 1
      elif focus == 'neu':
          if (y_pred[i] == 1) and (y_real[i] == 1):
              metrics['TP'] += 1
          elif (y_pred[i] != y_real[i]) and (y_real[i] == 1):
              metrics['FN'] += 1
          elif (y_pred[i] != y_real[i]) and (y_pred[i] == 1):
              metrics['FP'] += 1
          else:
              metrics['TN'] += 1
      else:
          if (y_pred[i] == 0) and (y_real[i] == 0):
              metrics['TP'] += 1
          elif (y_pred[i] != y_real[i]) and (y_real[i] == 0):
              metrics['FN'] += 1
          elif (y_pred[i] != y_real[i]) and (y_pred[i] == 0):
              metrics['FP'] += 1
          else:
              metrics['TN'] += 1
      i += 1
  return metrics

def calculate_class_metrics(base_metrics):
  """
  Calculate the precision, recall and specificity using the supplied base metrics (TP, TN, FP, FN)
  
  param base_metrics: (dict) containing the TP, FP, TN, FN values for a given class.
  """
  precision = base_metrics['TP'] / (base_metrics['TP'] + base_metrics['FP'])
  recall = base_metrics['TP'] / (base_metrics['TP'] + base_metrics['FN'])
  specificity = base_metrics['TN'] / (base_metrics['FP'] + base_metrics['TN'])

  class_metrics = {'Precision': precision, 'Recall': recall, 'Specificity': specificity}

  return class_metrics

def balanced_accuracy(pos, neu, neg):
  """
  Calculate the balanced accuracy by averaging the recall of the three classes.
  
  param pos: (dict) containing the precision and recall values for the positive class.
  param neu: (dict) containing the precision and recall values for the neutral class.
  param neg: (dict) containing the precision and recall values for the negative class.
  """
  bal_acc = (pos['Recall'] + neu['Recall'] + neg['Recall']) / 3
  return bal_acc

def calculate_f_measure(pos, neu, neg):
  """
  Calculate the Macro F1-score by using the formula supplied by Grandini et al., 2020.
  
  param pos: (dict) containing the precision and recall values for the positive class.
  param neu: (dict) containing the precision and recall values for the neutral class.
  param neg: (dict) containing the precision and recall values for the negative class.
  """
  MAP = (pos['Precision'] + neu['Precision'] + neg['Precision']) / 3
  MAR = (pos['Recall'] + neu['Recall'] + neg['Recall']) / 3

  f_measure = (2 * MAP * MAR) / (MAP**-1 + MAR**-1)

  return f_measure

def evaluate_performance(y_pred, y_true):
  """
  Calculates vanilla and balanced accuracy, precision, recall, specificity and f1-measure
  Returns all these values in a dictionary for easy access.
  
  param y_pred: (list) containing the predicted sentiment labels.
  param y_true: (list) containing the actual sentiment labels.
  """
  metrics = {}
  correct = determine_correctness(y_pred, y_true)
  
  accuracy = calc_accuracy_vanilla(correct)
  metrics['Accuracy'] = accuracy

  base_pos = calc_base_metrics(y_pred, y_true, 'pos')
  metrics['Base Positive'] = base_pos

  base_neu = calc_base_metrics(y_pred, y_true, 'neu')
  metrics['Base Neutral'] = base_neu

  base_neg = calc_base_metrics(y_pred, y_true, 'neg')
  metrics['Base Negative'] = base_neg

  adv_pos = calculate_class_metrics(base_pos)
  metrics['Advanced Positive'] = adv_pos

  adv_neu = calculate_class_metrics(base_neu)
  metrics['Advanced Neutral'] = adv_neu

  adv_neg = calculate_class_metrics(base_neg)
  metrics['Advanced Negative'] = adv_neg

  bal_acc = balanced_accuracy(adv_pos, adv_neu, adv_neg)
  metrics['Balanced Accuracy'] = bal_acc

  f_score = calculate_f_measure(adv_pos, adv_neu, adv_neg)
  metrics['F_Score'] = f_score

  return metrics
  
def confusion_matrix(y_pred, y_true):
  """
  Calculate the cells of the confusion matrix. Note that due to indexing this is the inverted matrix from the one in the document!
  (0,0) is (negative, negative), while normally (0,0) is (positive,positive)
  
  param y_pred: (list) containing the predicted sentiments.
  param y_true: (list) containing the actual sentiments.
  """
  table = np.zeros((3,3), dtype=int)
  i = 0

  while i < len(y_pred):
    x = int(y_pred[i])
    y = int(y_true[i])
    table[x][y] += 1
    i += 1

  return table
 