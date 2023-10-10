import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the dataset
data = pd.read_csv("adult.data", names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                                        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                                        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'],
                                        na_values = [' ?', '?'])

# Handle missing values (if any)
data.dropna(inplace=True)

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income'], drop_first=True)

# Separate features (X) and target variable (y)
X = data.drop('income_ >50K', axis=1)  # Features
y = data['income_ >50K']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Fit the model to the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

text = tree.export_text(decision_tree)
print(text) #print decision tree

print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", report)

