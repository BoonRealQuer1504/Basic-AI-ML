from __future__ import print_function
import numpy as np
from sklearn.metrics import accuracy_score

def acc(y_predictions, y_true):
    correct=np.sum(y_true==y_predictions)
    return float(correct/y_true.shape[0])


y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 2, 1])
print('Accuracy = ', acc(y_pred, y_true))

print("Accuracy = ",accuracy_score(y_true,y_pred))


