# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:35:39 2020

@author: bogdan
"""


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, f1_score


    

train_file = 'covid_train.csv'
validation_file = 'covid_validation.csv'

#columns = ['instituția sursă', 'sex', 'vârstă', 'dată debut simptome declarate',
 #          'simptome declarate', 'dată internare', 'simptome raportate la internare',
  #         'diagnostic și semne de internare', 'istoric de călătorie',
   #        'mijloace de transport folosite', 'confirmare contact cu o persoană infectată',
    #       'data rezultat testare', 'rezultat testare']

columns = ['simptome declarate', 'simptome raportate la internare',
           'diagnostic și semne de internare', 'istoric de călătorie',
           'mijloace de transport folosite', 'confirmare contact cu o persoană infectată',
           'rezultat testare']

# Specify the column names as the file does not have column header
df_train = pd.read_csv(train_file,names=columns)
df_validation = pd.read_csv(validation_file,names=columns)


X_train = df_train.iloc[:,:-1] # Features: 1st column onwards 
y_train = df_train.iloc[:,-1].ravel() # Target: 0th column


X_validation = df_validation.iloc[:,:-1]
y_validation = df_validation.iloc[:,-1].ravel()

# Encode Class Labels to integers -> label encoader merge doar pt o coloana
le = preprocessing.LabelEncoder()
le.fit(y_train)

y_t = le.transform(y_train)


# ordinal encoader merge aplicat pe mai multe coloane, il folosim pt X_train
oe = preprocessing.OrdinalEncoder()
oe.fit(X_train)

x_t = oe.transform(X_train)
# AICI SE VERIFICA DACA E OK NR DE ETICHETE

# se observa ca acum pe coloana pentru simptome generale datele sunt uniforme
# deoarece label encoader a generat doar 3 valori (exact cate label-uri am pus si noi)
# ps: a trebuit sa fac niste teste aici pan am nimerit toate caracterele in care se putea
# termina celula aia

#print(X_train['mijloace de transport folosite'][-50:-10])
#print('The values for encoded labels:')
#print(np.unique(oe.transform(X_train)[:, 9]))


clf = DecisionTreeClassifier(random_state=0)

clf.fit(x_t, y_t, sample_weight=None, check_input=True, X_idx_sorted=None)


le.fit(y_validation)
y_v = le.transform(y_validation)

oe.fit(X_validation)
x_v = oe.transform(X_validation)

# !! Predict
prediction = clf.predict(x_v)
#print(le.inverse_transform([prediction[0]]))


def metrics_generator(predicted, x1, y1, clf):
    plot_confusion_matrix(clf, x1, y1, display_labels=(['NEGATIV', 'POZITIV']))
    plt.show()
    print('SCORE: ' + str(clf.score(x1, y1)))
    print('Accuracy: ' + str(accuracy_score(y1, predicted)))
    print('Precision: ' + str(average_precision_score(y1, predicted)))
    print('Recall: ' + str(recall_score(y1, predicted)))
    print('F1: ' + str(f1_score(y1, predicted)))
    
    

if __name__ == '__main__':
    metrics_generator(prediction, x_v, y_v, clf)
        
        
        




