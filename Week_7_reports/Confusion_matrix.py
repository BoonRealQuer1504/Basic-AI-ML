from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
def confusion_matrix1(y_true,y_pred):
    N=np.unique(y_true).shape[0] #lay ra so gia tri phan biet cuar y_true --> so class can phan biet
    cm=np.zeros((N,N))
    for n in range(y_true.shape[0]):
        cm[y_true[n], y_pred[n]] += 1   # dem so cap gia tri thuc te mang class n, du doan mang class n voi n dong thoi tu 0-->2
    return cm 
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 1, 0, 2, 1, 1, 0, 2, 2, 1])


#function
cnf_matrix = confusion_matrix1(y_true, y_pred)
print('Confusion matrix:')
print(cnf_matrix)
#Sklearn
cnf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion matrix:')
print(cnf_matrix)

#normalized matrix 
normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
print('\nConfusion matrix (with normalizatrion:)')
print(normalized_confusion_matrix)
print('\nAccuracy:', np.diagonal(cnf_matrix).sum()/cnf_matrix.sum())