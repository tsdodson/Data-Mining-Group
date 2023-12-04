import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,silhouette_score,recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import export_graphviz,DecisionTreeClassifier
from sklearn import preprocessing, svm
from IPython.display import Image
from six import StringIO
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import keras_tuner
import keras
import time

def numtobinary(column):
    mean_value = column.mean()  # Calculate mean of the column
    return column.apply(lambda x: 1.0 if x > mean_value else 0.0)  # Transform to binary

    
def nnc(hp):
    
    model = keras.Sequential([
        layers.Dense(hp.Int("input_units",32,256,32), activation='relu', input_shape=(X_train.shape[1],)),
    ])
    
    for i in range(hp.Int("n_layers",1,4)):
        model.add(layers.Dense(hp.Int(f"conv_{i}_units",32,256,32), activation='relu'))
    
    model.add( layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if your labels are one-hot encoded
              metrics=['accuracy'])
    
    return model
    
    
df = pd.read_csv("adult.csv",header=0, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'])

#start of hot encoding
ohe = OneHotEncoder()

feature_array = ohe.fit_transform(df[['workclass','education','marital status','occupation','relationship','race','sex','native-country','salary']]).toarray()

feature_labels = ohe.categories_

feature_labels = np.hstack(feature_labels)

pd.DataFrame(feature_array,columns=feature_labels)

dfPostOneHot = pd.DataFrame(feature_array,columns=feature_labels)
#end of hot encoding

#start transforming binary
numCols = df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']]

#creates column that changes numerical value to 0 or 1 depending on if greater then mean
dfPostBinaryTrans = numCols.apply(numtobinary)

means = numCols.mean()

i = 0
for col in dfPostBinaryTrans.columns:
    dfPostBinaryTrans.rename(columns={col: str(col) + " is > " + str(round(means[i]))},inplace=True)
    i = i + 1
#end of transforming binary

#create final array that has both transformations
dfFinal = pd.concat([dfPostBinaryTrans,dfPostOneHot],axis=1)

X = dfFinal.iloc[:,0:104]
y = dfFinal.iloc[:,104:105]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

tuner = keras_tuner.RandomSearch(
    nnc,
    objective = "val_accuracy",
    max_trials = 10,
    executions_per_trial = 5
)

tuner.search(x = X_train,
             y = y_train,
             epochs = 3,
             batch_size = 64,
             validation_data=(X_test,y_test))

import pickle
with open(f"tuner_{int(time.time())}.pkl","wb") as f:
    pickle.dump(tuner,f)
    