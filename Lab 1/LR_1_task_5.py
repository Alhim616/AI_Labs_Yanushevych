from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

print("Matrix sklearn.metrics:", confusion_matrix(
    df.actual_label.values, df.predicted_RF.values))

def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def yanushevych_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

print("Marix Yanushevych", yanushevych_confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print("")
print("")

assert np.array_equal(
    yanushevych_confusion_matrix(df.actual_label.values, df.predicted_RF.values), 
    confusion_matrix(df.actual_label.values, df.predicted_RF.values)
), 'yanushevych_confusion_matrix() is not correct for RF'

assert np.array_equal(
    yanushevych_confusion_matrix(df.actual_label.values, df.predicted_LR.values), 
    confusion_matrix(df.actual_label.values, df.predicted_LR.values)
), 'yanushevych_confusion_matrix() is not correct for LR'

from sklearn.metrics import accuracy_score
print("sklearn.metrics accuracy_score", accuracy_score(df.actual_label.values, df.predicted_RF.values))

def yanushevych_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

assert yanushevych_accuracy_score(df.actual_label.values, df.predicted_RF.values) == \
    accuracy_score(df.actual_label.values, df.predicted_RF.values), \
    'my_accuracy_score failed on RF'

assert yanushevych_accuracy_score(df.actual_label.values, df.predicted_LR.values) == \
    accuracy_score(df.actual_label.values, df.predicted_LR.values), \
    'my_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % (yanushevych_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR: %.3f' % (yanushevych_accuracy_score(df.actual_label.values, df.predicted_LR.values)))
print("")
print("")

from sklearn.metrics import recall_score
print("sklearn.metrics recall_score", recall_score(df.actual_label.values, df.predicted_RF.values))

def yanushevych_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall

assert yanushevych_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values), 'my_recall_score failed on RF'
assert yanushevych_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values), 'my_recall_score failed on LR'

print('Recall RF: %.3f' % (yanushevych_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f' % (yanushevych_recall_score(df.actual_label.values, df.predicted_LR.values)))
print("")
print("")

from sklearn.metrics import precision_score
print("sklearn.metrics precision_score", precision_score(df.actual_label.values, df.predicted_RF.values))

from sklearn.metrics import precision_score

def yanushevych_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0 
    return precision

assert yanushevych_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values), 'my_precision_score failed on RF'
assert yanushevych_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values), 'my_precision_score failed on LR'

print('Precision RF: %.3f' % (yanushevych_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision LR: %.3f' % (yanushevych_precision_score(df.actual_label.values, df.predicted_LR.values)))
print("")
print("")

from sklearn.metrics import f1_score
print("sklearn.metrics f1_score", f1_score(df.actual_label.values, df.predicted_RF.values))

from sklearn.metrics import f1_score

def yanushevych_f1_score(y_true, y_pred):
    recall = yanushevych_recall_score(y_true, y_pred)
    precision = yanushevych_precision_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

assert yanushevych_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values), 'my_f1_score failed on RF'
#assert yanushevych_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'my_f1_score failed on LR'

print('F1 RF: %.3f' % (yanushevych_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 LR: %.3f' % (yanushevych_f1_score(df.actual_label.values, df.predicted_LR.values)))
print("")
print("")

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % (yanushevych_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (yanushevych_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (yanushevych_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (yanushevych_f1_score(df.actual_label.values, df.predicted_RF.values)))

print('')

print('scores with threshold = 0.25')
print('Accuracy RF: %.3f' % (yanushevych_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f' % (yanushevych_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f' % (yanushevych_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f' % (yanushevych_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))

from sklearn.metrics import roc_curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

from sklearn.metrics import roc_auc_score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

import matplotlib.pyplot as plt
plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()