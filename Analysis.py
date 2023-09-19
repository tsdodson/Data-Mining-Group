import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus, graphviz
import random
import time
import os


col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
# load dataset
data = pd.read_csv("adult.csv",header=0, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'])
# data.to_csv('adult.csv', index=None)

missing_values = (data==' ?').sum()
# print(missing_values)

data = data[data['workclass'] != ' ?']
# print(data.head())

categorical = data.select_dtypes(include=['object'])

encoder = preprocessing.LabelEncoder()

data = data[data['occupation'] != ' ?']
data = data[data['native-country'] != ' ?']

categorical = categorical.apply(encoder.fit_transform)
# print(categorical.head())

data = data.drop(categorical.columns, axis = 1)
data = pd.concat([data, categorical], axis = 1)
# print(data.head())

# print(data.info())

data['salary'] = data['salary'].astype('category')

X = data.drop('salary', axis = 1)
y = data['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 99)

tree = DecisionTreeClassifier(max_depth = 5)
tree.fit(X_train, y_train)

ypred = tree.predict(X_test)
print(classification_report(y_test,ypred))
print(confusion_matrix(y_test,ypred))
print(accuracy_score(y_test,ypred))

feat = list(data.columns[1:])
# print(feat)

dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,feature_names=feat, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("dtree.png")