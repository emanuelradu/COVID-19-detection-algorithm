# -*- coding: utf-8 -*-
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


from data_preparation import prepare_test
from train import clf, metrics_generator


# INSERT TEST FILENAME HERE!!!
TEST_FILENAME = 'input_test.csv'

prepare_test(TEST_FILENAME)



columns = ['simptome declarate', 'simptome raportate la internare',
           'diagnostic și semne de internare', 'istoric de călătorie',
           'mijloace de transport folosite', 'confirmare contact cu o persoană infectată',
           'rezultat testare']

# Specify the column names as the file does not have column header
df_test = pd.read_csv('test.csv',names=columns, encoding = 'utf-8')
print(df_test.shape)


X = df_test.iloc[:,:-1] # Features: 1st column onwards 
y = df_test.iloc[:,-1].ravel() # Target: 0th column


# Encode Class Labels to integers -> label encoader merge doar pt o coloana
le = preprocessing.LabelEncoder()
le.fit(y)

y_encoded = le.transform(y)


# ordinal encoader merge aplicat pe mai multe coloane, il folosim pt X_train
oe = preprocessing.OrdinalEncoder()
oe.fit(X)

x_encoded = oe.transform(X)


prediction = clf.predict(x_encoded)


metrics_generator(prediction, x_encoded, y_encoded, clf)


